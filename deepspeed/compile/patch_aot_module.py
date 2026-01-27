from torch._functorch.aot_autograd import AOTConfig, create_aot_dispatcher_function
from torch.utils._pytree import tree_flatten, tree_unflatten
from functools import wraps
from typing import Callable, Optional
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
from typing import Callable, Optional
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
from .patch_compiled_func import patch_compiled_func, unpatch_compiled_func, get_backward_inputs
from torch.utils.checkpoint import CheckpointPolicy

from .util import log_graph_0

BS_Shape = None
NH_Shape = None
World_Size = None

import operator
def extract_mlp_pattern_from_silu(silu_node: torch.fx.Node) -> list[torch.fx.Node] | None:
    if silu_node.op != "call_function":
        return None
    if silu_node.target != torch.nn.functional.silu:
        return None

    gate_proj_node = silu_node.args[0]
    if not isinstance(gate_proj_node, torch.fx.Node):
        return None
    if gate_proj_node.op != "call_function" or gate_proj_node.target != torch._C._nn.linear:
        return None

    mul_nodes = [user for user in silu_node.users if user.op == "call_function" and user.target == operator.mul]

    for mul_node in mul_nodes:
        up_proj_candidate = [arg for arg in mul_node.args if arg != silu_node][0]
        if not isinstance(up_proj_candidate, torch.fx.Node):
            continue
        if up_proj_candidate.op != "call_function" or up_proj_candidate.target != torch._C._nn.linear:
            continue

        down_proj_candidates = [
            user for user in mul_node.users
            if user.op == "call_function" and user.target == torch._C._nn.linear
        ]
        for down_proj_node in down_proj_candidates:
            return [
                gate_proj_node,
                silu_node,
                up_proj_candidate,
                mul_node,
                down_proj_node
            ]

    return None


def wrapper():
    # for the case of aot_module_simplified
    unpatch_compiled_func()
    original_aot_module_simplified = torch._functorch.aot_autograd.aot_module_simplified
    def patch_aot_module_simplified(
        mod: nn.Module,
        args,
        fw_compiler: Callable,
        bw_compiler: Optional[Callable] = None,
        partition_fn: Callable = default_partition,
        decompositions: Optional[Dict] = None,
        keep_inference_input_mutations=False,
        inference_compiler: Optional[Callable] = None,
        cudagraphs: Optional[BoxedBool] = None,
    ) -> nn.Module:
        # global World_Size
        World_Size = dist.get_world_size()
        world_size = World_Size
        rank = dist.get_rank()
        
        # get seq length and partitioned start, end
        global BS_Shape
        if BS_Shape is None:
            print(args[0].shape, flush=True)
            batch_size, seq_length = args[0].shape
            BS_Shape = (batch_size, seq_length * World_Size)
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
        for node in mod.graph.nodes:
            # constant
            S_partitioned = S // World_Size
            if "causal_mask" in node.name:
                arg_idx = 0
                for arg in node.args:
                    if not isinstance(arg, Node):
                        new_arg = replace_constant_in_args(arg, S_partitioned, S)
                        node.update_arg(arg_idx, new_arg)
                    arg_idx += 1
            # # position ids
            if node.name == "position_ids":
                old_arg = node.args[0]
                start = rank * S_partitioned
                end = (rank + 1) * S_partitioned
                with mod.graph.inserting_before(node):
                    new_arg = mod.graph.create_node(
                        op=old_arg.op,
                        target=old_arg.target,
                        args=(start, end),
                        kwargs=old_arg.kwargs,
                        name=None
                    )
                    node.replace_input_with(old_arg, new_arg)
            # constant
            if node.name == "reshape":
                old_arg = node.args[0]
                with mod.graph.inserting_before(node):
                    new_arg = mod.graph.create_node(
                        op=old_arg.op,
                        target=old_arg.target,
                        args=(0, S),
                        kwargs=old_arg.kwargs,
                        name=None
                    )
                    node.replace_input_with(old_arg, new_arg) 
            import torch.nn.functional as F
            if node.target == F.scaled_dot_product_attention:
                qkv = list(node.args[:3])
                file_names = ["query", "key", "value"]
                for i, old_arg in enumerate(qkv):
                    with mod.graph.inserting_after(old_arg):
                        new_arg = mod.graph.create_node('call_function', torch.ops.ulysses.all_to_all_qkv.default, (old_arg, B, S, N, H, world_size, file_names[i], ), {}, name=None)
                        old_arg.replace_all_uses_with(new_arg)
                        new_arg.args = (old_arg, B, S, N, H, world_size, file_names[i], )
                with mod.graph.inserting_after(node):
                    new_node = mod.graph.create_node('call_function', torch.ops.ulysses.all_to_all_out.default, (node, B, S, N, H, world_size, ), {}, name=None)
                    node.replace_all_uses_with(new_node)
                    new_node.args = (node, B, S, N, H, world_size, )
        mod.recompile()

        # for node in mod.graph.nodes:
        #     res = extract_mlp_pattern_from_silu(node)
        #     if res is not None:
        #         for mlp_node in res:
        #             mlp_node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

        # mod.recompile()
        log_graph_0(mod, filename="original_after")
        return original_aot_module_simplified(mod, args, fw_compiler, bw_compiler, partition_fn, decompositions, keep_inference_input_mutations,
                                            inference_compiler, cudagraphs)
    torch._functorch.aot_autograd.aot_module_simplified = patch_aot_module_simplified
    
@torch.library.custom_op("ulysses::all_to_all_qkv", mutates_args=())
def all_to_all_qkv(input_tensor: torch.Tensor, B: int, S: int, N: int, H: int, world_size: int, file_name: str) -> torch.Tensor:
    # b, n, s, h
    rank = dist.get_rank()
    input_t = input_tensor.reshape([B, world_size, N // world_size, S // world_size, H]).contiguous()
    input_t = input_t.permute(1, 0, 3, 2, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 0, 2, 3, 4).contiguous()  
    output = output.reshape(B, S, N // world_size, H).contiguous() 
    output = output.transpose(1, 2).contiguous() 
    rank = dist.get_rank()
    torch.cuda.synchronize()
    dist.barrier()
    return output
@torch.library.register_fake("ulysses::all_to_all_qkv")
def _(input_tensor, B, S, N, H, world_size, file_name):
    fake_tensor = input_tensor.new_empty((B, N // world_size, S, H))
    return fake_tensor
def all_to_all_qkv_setup_context(ctx, inputs, output) -> Tensor:
    input_tensor, B, S, N, H, world_size, file_name = inputs
    ctx.saved_data = (B, S, N, H, world_size)
def all_to_all_qkv_backward(ctx, grad):
    B, S, N, H, world_size = ctx.saved_data
    input_t = grad.reshape([B, N // world_size, world_size, S // world_size, H]).contiguous()
    input_t = input_t.permute(2, 0, 3, 1, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 2, 0, 3, 4).contiguous() 
    output = output.reshape(B, S // world_size, N, H).contiguous()
    output = output.transpose(1, 2).contiguous()
    return (output, None, None, None, None, None, None)
torch.library.register_autograd(
    "ulysses::all_to_all_qkv", all_to_all_qkv_backward, setup_context=all_to_all_qkv_setup_context
)

@torch.library.custom_op("ulysses::all_to_all_out", mutates_args=())
def all_to_all_out(input_tensor: torch.Tensor, B: int, S: int, N: int, H: int, world_size: int) -> torch.Tensor:
    # b, n, s, h
    input_t = input_tensor.reshape([B, N // world_size, world_size, S // world_size, H]).contiguous()
    input_t = input_t.permute(2, 0, 3, 1, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 2, 0, 3, 4).contiguous() 
    output = output.reshape(B, S // world_size, N, H).contiguous()
    output = output.transpose(1, 2).contiguous()
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
    input_t = input_t.permute(1, 0, 3, 2, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=dist.group.WORLD)
    output = output.permute(1, 0, 2, 3, 4).contiguous()  
    output = output.reshape(B, S, N // world_size, H).contiguous()
    output = output.transpose(1, 2).contiguous() 
    return (output, None, None, None, None, None)
torch.library.register_autograd(
    "ulysses::all_to_all_out", all_to_all_out_backward, setup_context=all_to_all_out_setup_context
)
