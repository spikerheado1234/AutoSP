# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch
import torch.utils._pytree as pytree

from torch._functorch.aot_autograd import create_aot_dispatcher_function
from torch._inductor.lowering import register_lowering, fallbacks, add_needs_realized_inputs
from torch._inductor.ir import TensorBox, FallbackKernel, Layout, IRNode
from torch._inductor.virtualized import V
from torch._inductor.scheduler import Scheduler
import torch.distributed as dist

from .util import get_input_nodes
from .graph_param import DSGraphParamManager
from torch.fx.experimental.symbolic_shapes import ShapeEnv
import torch._inductor as inductor_compile
import os

from torch.distributed._functional_collectives import all_to_all_inplace
from torch.fx.node import Node

original_create_aot_dispatcher_function = create_aot_dispatcher_function
original_compile = inductor_compile.compile

## Helper utils for ZeRo-1 support. ##
GROUP_REGISTRY = {} # int -> dist.ProcessGroup

def register_groups(groups):
    """groups: List[List[int]], e.g. [[0,1],[2,3]]"""
    for gid, ranks in enumerate(groups):
        if gid not in GROUP_REGISTRY:
            GROUP_REGISTRY[gid] = dist.new_group(ranks)

def get_group(gid: int):
    return GROUP_REGISTRY[gid] if gid is not None else dist.group.WORLD

def patch_create_aot_dispatcher_function(graph_id: int, z3_partition: bool, make_fw_graph, make_bw_graph, real_inputs,
                                         param_indices, param_manager):

    def wrapper_create_aot_dispatcher_function(
        flat_fn,
        fake_flat_args,
        aot_config,
        fake_mode,
        shape_env,
    ):

        def patch_compiler(original_compiler, bwd):

            def wrapped_compiler(gm, fake_inputs):
                dc_compiler = make_bw_graph if bwd else make_fw_graph
                dc_compiler(gm, fake_inputs)

                if z3_partition:
                    # Inductor validates input size estimated by the first trace, where ds tensor is materialized.
                    # We need to patch the input tensors to avoid the validation error.
                    patched_inputs = []
                    if bwd:
                        param_nodes_bw, _ = param_manager[graph_id].get_bwd_mapping(gm.graph)
                        param_names = [n.name for n in param_nodes_bw]
                    else:
                        param_names = param_manager[graph_id].param_names
                    input_nodes = get_input_nodes(gm.graph)

                    for in_node, in_v in zip(input_nodes, fake_inputs):
                        ds_param = in_node.name in param_names
                        if ds_param:
                            from torch._subclasses.fake_tensor import is_fake
                            from torch._dynamo.utils import to_fake_tensor
                            assert is_fake(in_v), f"Input {in_v} should be fake tensor"
                            patched_inputs.append(
                                to_fake_tensor(torch.empty([0], dtype=in_v.dtype, device=in_v.device), in_v.fake_mode))
                        else:
                            patched_inputs.append(in_v)

                    patched_inputs = tuple(patched_inputs)
                else:
                    patched_inputs = fake_inputs

                return original_compiler(gm, patched_inputs)

            return wrapped_compiler

        def wrap_partition_fn(partition_fn):

            def wrapped_partition_fn(*args, **kwargs):
                fw_module, bw_module = partition_fn(*args, **kwargs)

                # get parameter names
                pm = DSGraphParamManager(fw_module.graph, real_inputs, param_indices)

                def fix_placeholder_meta(graph):
                    for n in graph.nodes:
                        if n.op == "placeholder" and n.name in pm.param_names:
                            n.meta["val"] = torch.empty([0], dtype=n.meta["val"].dtype, device=n.meta["val"].device)

                fix_placeholder_meta(fw_module.graph)
                fix_placeholder_meta(bw_module.graph)

                return fw_module, bw_module

            return wrapped_partition_fn

        if z3_partition:
            aot_config.partition_fn = wrap_partition_fn(aot_config.partition_fn)

        aot_config.fw_compiler = patch_compiler(aot_config.fw_compiler, False)
        aot_config.bw_compiler = patch_compiler(aot_config.bw_compiler, True)
        aot_config.inference_compiler = aot_config.fw_compiler

        return original_create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)

    torch._functorch.aot_autograd.create_aot_dispatcher_function = wrapper_create_aot_dispatcher_function

def patch_create_aot_dispatcher_function_ulysses():    
    def wrapper_create_aot_dispatcher_function(
        flat_fn,
        fake_flat_args,
        aot_config,
        fake_mode,
        shape_env,
    ):
        def wrap_partition_fn(partition_fn):
            def wrapped_partition_fn(*args, **kwargs):
                fw_module, bw_module = partition_fn(*args, **kwargs)
                return fw_module, bw_module
            return wrapped_partition_fn
        aot_config.partition_fn = wrap_partition_fn(aot_config.partition_fn)
        return original_create_aot_dispatcher_function(flat_fn, fake_flat_args, aot_config, fake_mode, shape_env)
    torch._functorch.aot_autograd.create_aot_dispatcher_function = wrapper_create_aot_dispatcher_function


def register_custom_ops():

    def fallback_handler_no_reuse(kernel,
                                  never_reuse_input,
                                  never_reuse_output,
                                  force_free_input,
                                  add_to_fallback_set=True):
        if add_to_fallback_set:
            fallbacks.add(kernel)

        def handler(*args, **kwargs):

            def wrap_tensors(x):
                out = TensorBox.create(x) if isinstance(x, torch._inductor.ir.IRNode) else x
                if out is not None and never_reuse_output:
                    V.graph.never_reuse_buffers.add(out.get_name())
                return out

            class CustomDCKernel(FallbackKernel):

                def __init__(self, op, *args, **kwargs):
                    super().__init__(op, *args, **kwargs)

                    def add_to_never_reuse(x):
                        if isinstance(x, IRNode):
                            assert hasattr(x, "get_name"), f"x doesn't have get_name {x.__class__}"
                            V.graph.never_reuse_buffers.add(x.get_name())

                    if never_reuse_input:
                        pytree.tree_map(add_to_never_reuse, args)

                def get_var_name_for_arg(self, arg: str):
                    if arg.isidentifier():
                        return arg

                    import re
                    match = re.match(r"reinterpret_tensor\((\w+),", arg)
                    if match:
                        return match.group(1)
                    return None

                def codegen(self, wrapper):
                    if not force_free_input:
                        return super().codegen(wrapper)

                    kernel = self.op_overload
                    self.codegen_comment(wrapper)
                    args = [*self.codegen_args(), *self.codegen_kwargs()]

                    V.graph.wrapper_code.generate_fallback_kernel(self, args)
                    if isinstance(self.layout, Layout):
                        self.codegen_size_asserts(wrapper)

                    var_name = self.get_var_name_for_arg(args[0])
                    if var_name:
                        wrapper.writeline(f"{var_name} = None")

                    self.codegen_unbacked_symbol_defs(wrapper)

            kernel_cls = CustomDCKernel if force_free_input else FallbackKernel
            return pytree.tree_map(wrap_tensors, kernel_cls.create(kernel, *args, **kwargs))

        return handler

    def register_fallback_no_reuse(op_overload,
                                   never_reuse_input=False,
                                   never_reuse_output=False,
                                   force_free_input=False):
        add_needs_realized_inputs(op_overload)
        return register_lowering(op_overload, type_promotion_kind=None)(fallback_handler_no_reuse(
            op_overload,
            never_reuse_input=never_reuse_input,
            never_reuse_output=never_reuse_output,
            force_free_input=force_free_input))

    # Inductor tries to reuse output buffer when possible. We need to disable this behavior for some custom ops.
    # -> It seems that memory region is still reused in some cases. So we clone the inputs for some ops.
    register_fallback_no_reuse(torch.ops.dc.allgather_param.default, never_reuse_input=False, never_reuse_output=True)
    register_fallback_no_reuse(torch.ops.dc.wait_allgather.default, never_reuse_input=True, never_reuse_output=True)
    register_fallback_no_reuse(torch.ops.dc.release_param.default, never_reuse_input=True, never_reuse_output=False)
    register_fallback_no_reuse(torch.ops.dc.reduce_grad.default,
                               never_reuse_input=True,
                               never_reuse_output=True,
                               force_free_input=True)
    register_fallback_no_reuse(torch.ops.dc.free_tensors.default, never_reuse_input=True, never_reuse_output=True)

    if not hasattr(Scheduler, "is_dc_patched") or not Scheduler.is_dc_patched:
        Scheduler.is_dc_patched = True
        Scheduler.dead_node_elimination = lambda _: None

BS_Shape = None
NH_Shape = None
sp_size = int(os.getenv('SP_SIZE'))
B = None
S = None
N = None
H = None

GROUP_REGISTRY = {} # int -> dist.ProcessGroup

def register_groups(groups):
    """groups: List[List[int]], e.g. [[0,1],[2,3]]"""
    for gid, ranks in enumerate(groups):
        if gid not in GROUP_REGISTRY:
            GROUP_REGISTRY[gid] = dist.new_group(ranks)

def patch_compile_fx(gm, example_inputs, options=None):

    rank = dist.get_rank() % sp_size

    group_listing = []
    offset = 0
    for _ in range(int(os.getenv('DP_SIZE'))):
        group_listing.append([i + offset for i in range(sp_size)])
        offset += sp_size

    register_groups(group_listing)

    group_id = (dist.get_rank() // sp_size)  # 0 or 1

    # get seq length and partitioned start, end
    global BS_Shape
    global B, S
    if BS_Shape is None:
        batch_size, seq_length = example_inputs[0].shape
        BS_Shape = (batch_size, seq_length * sp_size)
        B, S = BS_Shape 
    
    # grab N, H info    
    global NH_Shape
    global N, H
    if NH_Shape is None:
        for node in gm.graph.nodes:
            if node.name == "attn_output":
                _, num_heads, _, head_dim = node.args[0].meta['example_value'].shape
                NH_Shape = (num_heads, head_dim)
                break
        if NH_Shape:
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
    for node in gm.graph.nodes:
        # constant
        S_partitioned = S // sp_size
        if "causal_mask" in node.name:
            arg_idx = 0
            for arg in node.args:
                if not isinstance(arg, Node):
                    new_arg = replace_constant_in_args(arg, S_partitioned, S)
                    node.update_arg(arg_idx, new_arg)
                arg_idx += 1
         # position ids
        if node.name == "position_ids":
            old_arg = node.args[0]
            start = rank * S_partitioned
            end = (rank + 1) * S_partitioned
            with gm.graph.inserting_before(node):
                new_arg = gm.graph.create_node(
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
            with gm.graph.inserting_before(node):
                new_arg = gm.graph.create_node(
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
            for i, old_arg in enumerate(qkv):
                with gm.graph.inserting_after(old_arg):
                    new_arg = gm.graph.create_node('call_function', torch.ops.ulysses.all_to_all_qkv.default, (old_arg, B, S, N, H, sp_size, group_id, str(i)), {}, name=None)
                    old_arg.replace_all_uses_with(new_arg)
                    new_arg.args = (old_arg, B, S, N, H, sp_size, group_id, str(i))
            with gm.graph.inserting_after(node):
                new_node = gm.graph.create_node('call_function', torch.ops.ulysses.all_to_all_out.default, (node, B, S, N, H, sp_size, group_id, ), {}, name=None)
                node.replace_all_uses_with(new_node)
                new_node.args = (node, B, S, N, H, sp_size, group_id,)
    gm.recompile()
    return original_compile(gm, example_inputs, options)

@torch.library.custom_op("ulysses::all_to_all_qkv", mutates_args=())
def all_to_all_qkv(input_tensor: torch.Tensor, B: int, S: int, N: int, H: int, sp_size: int, group_id: int, name: str) -> torch.Tensor:
    # b, n, s, h
    group_ = get_group(group_id)
    rank = dist.get_rank()

    torch.save(input_tensor, f'qkv_prior_{name}_{rank}.pt')

    input_t = input_tensor.reshape([B, sp_size, N // sp_size, S // sp_size, H]).contiguous()
    input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=group_)
    output = output.permute(1, 2, 0, 3, 4).contiguous()  
    output = output.reshape(B, N // sp_size, S, H).contiguous()
    return output
@torch.library.register_fake("ulysses::all_to_all_qkv")
def _(input_tensor, B, S, N, H, sp_size, group_id, name: str):
    fake_tensor = input_tensor.new_empty((B, N // sp_size, S, H))
    return fake_tensor
def all_to_all_qkv_setup_context(ctx, inputs, output) -> torch.Tensor:
    input_tensor, B, S, N, H, sp_size, group_id, name = inputs
    ctx.saved_data = (B, S, N, H, sp_size, group_id)
def all_to_all_qkv_backward(ctx, grad):
    B, S, N, H, sp_size, group_id = ctx.saved_data
    group_ = get_group(group_id)
    input_t = grad.reshape([B, N // sp_size, sp_size, S // sp_size, H]).contiguous()
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous() # [sp_size, B, N // sp_size, S // sp_size, H]
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=group_)
    output = output.permute(1, 0, 2, 3, 4).contiguous() # [B, sp_size, N // sp_size, S // sp_size, H]
    output = output.reshape(B, N, S // sp_size, H).contiguous()
    return (output, None, None, None, None, None, None, None)
torch.library.register_autograd(
    "ulysses::all_to_all_qkv", all_to_all_qkv_backward, setup_context=all_to_all_qkv_setup_context
)

@torch.library.custom_op("ulysses::all_to_all_out", mutates_args=())
def all_to_all_out(input_tensor: torch.Tensor, B: int, S: int, N: int, H: int, sp_size: int, group_id: int) -> torch.Tensor:
    # b, n, s, h
    rank = dist.get_rank()
    group_ = get_group(group_id)
    input_t = input_tensor.reshape([B, N // sp_size, sp_size, S // sp_size, H]).contiguous() 
    input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=group_)
    output = output.permute(1, 0, 2, 3, 4).contiguous() 
    output = output.reshape(B, N, S // sp_size, H).contiguous()  
    print(f'pre permute shape: {output.shape}')
    #output = output.permute(0, 2, 1, 3).contiguous()
    print(f'post permute shape: {output.shape}')
    #output = torch.load(f'attn_eager_post_{rank}.pt')
    print(f'output: {output.sum()} rank: {rank}')
    #if rank == 0:
    #    torch.save(output, 'attn_post_.pt')
    return output
@torch.library.register_fake("ulysses::all_to_all_out")
def _(input_tensor, B, S, N, H, sp_size, group_id):
    #fake_tensor = input_tensor.new_empty((B, S // sp_size, N, H))
    fake_tensor = input_tensor.new_empty((B, N, S // sp_size, H))
    return fake_tensor
def all_to_all_out_setup_context(ctx, inputs, output) -> torch.Tensor:
    input_tensor, B, S, N, H, sp_size, group_id = inputs
    ctx.saved_data = (B, S, N, H, sp_size, group_id)
def all_to_all_out_backward(ctx, grad):
    B, S, N, H, sp_size, group_id = ctx.saved_data
    group_ = get_group(group_id)
    #grad = grad.permute(0, 2, 1, 3).contiguous()
    input_t = grad.reshape([B, sp_size, N // sp_size, S // sp_size, H]).contiguous()
    input_t = input_t.permute(1, 0, 2, 3, 4).contiguous() # [sp_size, B, N // sp_size, S // sp_size, H].
    output = torch.empty_like(input_t)
    all_to_all_inplace(output, input_t, group=group_)
    output = output.permute(1, 2, 0, 3, 4).contiguous() # [B, N // sp_size, sp_size, S // sp_size, H]. 
    output = output.reshape(B, N // sp_size, S, H).contiguous() 
    return (output, None, None, None, None, None, None)
torch.library.register_autograd(
    "ulysses::all_to_all_out", all_to_all_out_backward, setup_context=all_to_all_out_setup_context
)
