from torch._functorch.aot_autograd import AOTConfig, create_aot_dispatcher_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from functools import wraps
from typing import Callable, Optional, Any
import torch
from typing import List, Dict
from torch.library import Library
from torch._dynamo.utils import to_fake_tensor
from torch.autograd import Function
from torch import Tensor

from torch._prims_common import CUDARngStateHelper
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
import torch.nn as nn
import torch.nn.functional as F
from torch._functorch.aot_autograd import aot_module_simplified
import torch.optim as optim
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import create_aot_dispatcher_function
from torch._inductor.lowering import register_lowering, fallbacks, add_needs_realized_inputs
from torch._inductor.ir import TensorBox, FallbackKernel, Layout, IRNode
from torch._inductor.virtualized import V
from torch._inductor.scheduler import Scheduler
from torch._functorch._aot_autograd.schemas import (
    OutputType,
    SubclassCreationMeta,
)
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.aot_autograd import AOTConfig, create_aot_dispatcher_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from functools import wraps
from typing import Callable, Optional, Protocol
import torch
from typing import List
from torch._prims_common import CUDARngStateHelper
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
import torch.nn as nn
import torch.nn.functional as F
from torch._functorch.aot_autograd import aot_module_simplified
import torch.optim as optim
import torch.utils._pytree as pytree
from torch._functorch.aot_autograd import create_aot_dispatcher_function
from torch._inductor.lowering import register_lowering, fallbacks, add_needs_realized_inputs
from torch._inductor.ir import TensorBox, FallbackKernel, Layout, IRNode
from torch._inductor.virtualized import V
from torch._inductor.scheduler import Scheduler
from torch._functorch._aot_autograd.schemas import (
    OutputType,
    SubclassCreationMeta,
)
from torch._subclasses import FakeTensor
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._functorch.partitioners import default_partition
from torch._inductor.utils import BoxedBool, InputType
from torch._inductor.cudagraph_utils import BoxedDeviceIndex
import torch.distributed as dist
import torch.distributed._functional_collectives as ft_c
from torch.distributed._functional_collectives import all_to_all_inplace
from torch.fx import GraphModule
from torch.fx.node import Node
from torch.fx.passes.shape_prop import ShapeProp

from collections.abc import KeysView, Sequence

BS_Shape = None
NH_Shape = None
World_Size = None

class AOTDispatchCompiler(Protocol):
    """
    Represents a fw or bw_compiler passed to AOTAutograd.
    """

    def __call__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
    ) -> Any:
        ...

def wrapper():
    # for the case of aot_module_simplified
    original_aot_module_simplified = torch._functorch.aot_autograd.aot_module_simplified
    def patch_aot_module_simplified(
        mod: nn.Module,
        args,
        fw_compiler: AOTDispatchCompiler,
        bw_compiler: Optional[AOTDispatchCompiler] = None,
        partition_fn: Callable = default_partition,
        decompositions: Optional[dict] = None,
        keep_inference_input_mutations=False,
        inference_compiler: Optional[AOTDispatchCompiler] = None,
        cudagraphs: Optional[BoxedBool] = None,
        boxed_forward_device_index: Optional[BoxedDeviceIndex] = None,
        ignore_shape_env: bool = False,
    ) -> nn.Module:
        global World_Size
        World_Size = dist.get_world_size()
        world_size = World_Size
        rank = dist.get_rank()
        
        # get seq length and partitioned start, end
        global BS_Shape
        if BS_Shape is None:
            batch_size, seq_length = args[0].shape
            BS_Shape = (batch_size, seq_length)
        B, S = BS_Shape # S here is full sequence
            
        # grab N, H info    
        global NH_Shape
        if NH_Shape is None:
            for node in mod.graph.nodes:
                if node.name == "attn_output":
                    _, num_heads, _, head_dim = node.args[0].meta['example_value'].shape
                    NH_Shape = (num_heads, head_dim)
                    break
        N, H = NH_Shape
        
        chunk_size = S // world_size
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size
        
        # replace seq to partitioned_seq
        def replace_constant_in_args(obj, old_value, new_value):
            if isinstance(obj, (list, tuple)):
                return type(obj)(replace_constant_in_args(o, old_value, new_value) for o in obj)
            elif isinstance(obj, dict):
                return {k: replace_constant_in_args(v, old_value, new_value) for k, v in obj.items()}
            elif isinstance(obj, slice):
                return slice(
                    replace_constant_in_args(obj.start, old_value, new_value),
                    replace_constant_in_args(obj.stop, old_value, new_value),
                    replace_constant_in_args(obj.step, old_value, new_value)
                )
            elif obj == old_value:
                return new_value
            else:
                return obj
        
        # modify the graph
        idx = 0
        for node in mod.graph.nodes:
            if idx == 0:
                with mod.graph.inserting_after(node):
                    new_node = mod.graph.create_node('call_function', torch.ops.ulysses.shard.default, (node, B, start_idx, end_idx, ), {}, name=None)
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node, B, start_idx, end_idx, )
            # constant
            S_partitioned = S // World_Size
            if node.name == "view" or node.name == "view_1" or node.name == "view_2" or node.name == "reshape_1":
                arg_idx = 0
                for arg in node.args:
                    if not isinstance(arg, Node):
                        new_arg = replace_constant_in_args(arg, S, S_partitioned)
                        node.update_arg(arg_idx, new_arg)
                    arg_idx += 1
            if node.name == "position_ids":
                old_arg = node.args[0]
                with mod.graph.inserting_before(node):
                    new_arg = mod.graph.create_node(
                        op=old_arg.op,
                        target=old_arg.target,
                        args=(0, S_partitioned),
                        kwargs=old_arg.kwargs,
                        name=None
                    )
                    node.replace_input_with(old_arg, new_arg) 
            if node.name == "attn_output":
                qkv = list(node.args[:3])
                node_names = ["query", "key", "value"]
                for i, old_arg in enumerate(qkv):
                    with mod.graph.inserting_after(old_arg):
                        new_arg = mod.graph.create_node('call_function', torch.ops.ulysses.all_to_all_qkv.default, (old_arg, B, S, N, H, world_size, node_names[i]), {}, name=None)
                        old_arg.replace_all_uses_with(new_arg)
                        new_arg.args = (old_arg, B, S, N, H, world_size, node_names[i],)
                with mod.graph.inserting_after(node):
                    new_node = mod.graph.create_node('call_function', torch.ops.ulysses.all_to_all_out.default, (node, B, S, N, H, world_size, ), {}, name=None)
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node, B, S, N, H, world_size, )
            idx += 1
        mod.recompile()
        return original_aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations,
                                            inference_compiler, cudagraphs)
    torch._functorch.aot_autograd.aot_module_simplified = patch_aot_module_simplified
    
    # for the case of aot_module
    def patch_aot_function(
        fn: Callable,
        fw_compiler: Callable,
        bw_compiler: Optional[Callable] = None,
        partition_fn: Callable = None,
        decompositions: Optional[dict] = None,
        num_params_buffers: int = 0,
        keep_inference_input_mutations: bool = False,
        inference_compiler: Optional[Callable] = None,
        *,
        dynamic=False,
        enable_log=True,
    ) -> Callable:
        if bw_compiler is None:
            bw_compiler = fw_compiler
        if inference_compiler is None:
            inference_compiler = fw_compiler

        aot_config = AOTConfig(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            inference_compiler=inference_compiler,
            partition_fn=partition_fn,
            decompositions=decompositions,
            num_params_buffers=num_params_buffers,
            aot_id=0,
            keep_inference_input_mutations=keep_inference_input_mutations,
            dynamic_shapes=dynamic,
            aot_autograd_arg_pos_to_source=None,
            is_export=False,
            no_tangents=False,
            enable_log=enable_log,
        )
        cached_res = None

        @wraps(fn)
        def returned_function(*args, **kwargs):
            nonlocal cached_res
            flat_args = tree_flatten((args, kwargs))[0]
            if cached_res is None:
                from torch._functorch.aot_autograd import create_tree_flattened_fn, construct_fake_mode, process_inputs

                flat_fn, out_spec = create_tree_flattened_fn(fn, args, kwargs)
                fake_mode, shape_env = construct_fake_mode(flat_args, aot_config)
                fake_flat_args = process_inputs(flat_args, aot_config, fake_mode, shape_env)

                compiled_fn, _ = create_aot_dispatcher_function(
                    flat_fn,
                    fake_flat_args,
                    aot_config,
                    fake_mode,
                    shape_env,
                )
                cached_res = (compiled_fn, out_spec)

            compiled_fn, out_spec = cached_res
            out = compiled_fn(flat_args)
            return out_spec.unflatten(out)
        return returned_function
    torch._functorch.aot_autograd.aot_function = patch_aot_function
    
@torch.library.custom_op("ulysses::shard", mutates_args=())
def shard(input_tensor: torch.Tensor, B: int, start_idx: int, end_idx: int) -> torch.Tensor:
    sharded = input_tensor[:, start_idx:end_idx]
    return sharded.clone()
@shard.register_fake
def _(input_tensor, B, start_idx, end_idx):
    fake_tensor = input_tensor.new_empty((B, end_idx - start_idx))
    return fake_tensor

@torch.library.custom_op("ulysses::all_to_all_qkv", mutates_args=())
def all_to_all_qkv(input_tensor: torch.Tensor, B: int, S: int, N: int, H: int, world_size: int, node_name: str) -> torch.Tensor:
    # b, n, s, h
    rank = dist.get_rank()
    input_t = input_tensor.reshape([B, world_size, N // world_size, S // world_size, H]).contiguous()
    input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 2, 0, 3, 4).contiguous()  
    output = output.reshape(B, N // world_size, S, H).contiguous()
    return output
@torch.library.register_fake("ulysses::all_to_all_qkv")
def _(input_tensor, B, S, N, H, world_size, node_name: str):
    fake_tensor = input_tensor.new_empty((B, N // world_size, S, H))
    return fake_tensor
def all_to_all_qkv_setup_context(ctx, inputs, output) -> Tensor:
    input_tensor, B, S, N, H, world_size, node_name = inputs
    ctx.saved_data = (B, S, N, H, world_size, node_name)
def all_to_all_qkv_backward(ctx, grad):
    B, S, N, H, world_size, node_name = ctx.saved_data
    input_t = grad.reshape([B, N // world_size, world_size, S // world_size, H])
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 0, 2, 3, 4).contiguous() 
    output = output.reshape(B, N, S // world_size, H)
    return (output, None, None, None, None, None, None)
torch.library.register_autograd(
    "ulysses::all_to_all_qkv", all_to_all_qkv_backward, setup_context=all_to_all_qkv_setup_context
)

@torch.library.custom_op("ulysses::all_to_all_out", mutates_args=())
def all_to_all_out(input_tensor: torch.Tensor, B: int, S: int, N: int, H: int, world_size: int) -> torch.Tensor:
    # b, n, s, h
    rank = dist.get_rank()
    input_t = input_tensor.reshape([B, N // world_size, world_size, S // world_size, H]).contiguous()
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 0, 2, 3, 4).contiguous() 
    output = output.reshape(B, N, S // world_size, H).contiguous()
    return output
@torch.library.register_fake("ulysses::all_to_all_out")
def _(input_tensor, B, S, N, H, world_size):
    fake_tensor = input_tensor.new_empty((B, N, S // world_size, H))
    return fake_tensor
def all_to_all_out_setup_context(ctx, inputs, output) -> Tensor:
    input_tensor, B, S, N, H, world_size = inputs
    ctx.saved_data = (B, S, N, H, world_size)
def all_to_all_out_backward(ctx, grad):
    B, S, N, H, world_size = ctx.saved_data
    input_t = grad.reshape([B, world_size, N // world_size, S // world_size, H]).contiguous()
    input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 2, 0, 3, 4).contiguous()  
    output = output.reshape(B, N // world_size, S, H).contiguous() 
    return (output, None, None, None, None, None, None)
torch.library.register_autograd(
    "ulysses::all_to_all_out", all_to_all_out_backward, setup_context=all_to_all_out_setup_context
)