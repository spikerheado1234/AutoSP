"""AutoSP: Automatic Sequence Parallel (Ulysses) pass for graph modules.

Ulysses Transformation:
    Input:  [B, N, S/P, H]  (all heads, partitioned sequence)
    After A2A on QKV: [B, N/P, S, H]  (partitioned heads, full sequence)
    After SDPA: [B, N/P, S, H]
    After A2A on O: [B, N, S/P, H]  (all heads, partitioned sequence)

Where:
    B = batch size, N = num heads, S = full sequence length, H = head dim, P = world size
"""

import os
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.fx import GraphModule, Node

# Debug flag - set to True to dump tensors
DEBUG_DUMP_TENSORS = os.environ.get("AUTOSP_DEBUG", "0") == "1"
DEBUG_DUMP_DIR = os.environ.get("AUTOSP_DEBUG_DIR", "autosp_debug")


# ============================================================================
# Custom Op: Unified all-to-all with scatter/gather indices
# ============================================================================

@torch.library.custom_op("autosp::all_to_all", mutates_args=())
def all_to_all(
    input: torch.Tensor,
    scatter_idx: int,
    gather_idx: int,
    world_size: int,
    name: str,
) -> torch.Tensor:
    """
    All-to-all collective for SDPA tensors [B, N, S, H].
    
    For QKV (scatter_idx=1, gather_idx=2):
        [B, N, S/P, H] -> [B, N/P, S, H]
    For O (scatter_idx=2, gather_idx=1):
        [B, N/P, S, H] -> [B, N, S/P, H]
    """
    if DEBUG_DUMP_TENSORS:
        rank = dist.get_rank()
        os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
        torch.save(input, f"{DEBUG_DUMP_DIR}/{name}_input_rank{rank}.pt")
    
    B, dim1, dim2, H = input.shape
    
    if scatter_idx == 1:  # QKV: scatter heads, gather sequence
        # [B, N, S/P, H] -> [B, N/P, S, H]
        N, local_S = dim1, dim2
        
        # Split N into [P, N/P]: [B, P, N/P, S/P, H]
        input_t = input.reshape(B, world_size, N // world_size, local_S, H)
        # Permute P to front: [P, B, N/P, S/P, H]
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=dist.group.WORLD)
        
        # Permute: [P, B, N/P, S/P, H] -> [B, N/P, P, S/P, H]
        output = output.permute(1, 2, 0, 3, 4).contiguous()
        # Merge P into S: [B, N/P, S, H]
        output = output.reshape(B, N // world_size, world_size * local_S, H)
    
    else:  # scatter_idx == 2, O: scatter sequence, gather heads
        # [B, N/P, S, H] -> [B, N, S/P, H]
        local_N, S = dim1, dim2
        
        # Split S into [P, S/P]: [B, N/P, P, S/P, H]
        input_t = input.reshape(B, local_N, world_size, S // world_size, H)
        # Permute P to front: [P, B, N/P, S/P, H]
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()
        
        output = torch.empty_like(input_t)
        dist.all_to_all_single(output, input_t, group=dist.group.WORLD)
        
        # Permute: [P, B, N/P, S/P, H] -> [B, P, N/P, S/P, H]
        output = output.permute(1, 0, 2, 3, 4).contiguous()
        # Merge P into N: [B, N, S/P, H]
        output = output.reshape(B, world_size * local_N, S // world_size, H)
    
    if DEBUG_DUMP_TENSORS:
        torch.save(output, f"{DEBUG_DUMP_DIR}/{name}_output_rank{rank}.pt")
    
    return output


@torch.library.register_fake("autosp::all_to_all")
def all_to_all_fake(input: torch.Tensor, scatter_idx: int, gather_idx: int, world_size: int, name: str):
    B, dim1, dim2, H = input.shape
    
    if scatter_idx == 1:  # QKV
        return input.new_empty(B, dim1 // world_size, dim2 * world_size, H)
    else:  # O
        return input.new_empty(B, dim1 * world_size, dim2 // world_size, H)


def all_to_all_backward_setup(ctx, inputs, output):
    _, scatter_idx, gather_idx, world_size, name = inputs
    # Backward swaps scatter and gather
    ctx.scatter_idx = gather_idx
    ctx.gather_idx = scatter_idx
    ctx.world_size = world_size
    ctx.name = name + "_grad"


def all_to_all_backward(ctx, grad):
    return (
        all_to_all(grad, ctx.scatter_idx, ctx.gather_idx, ctx.world_size, ctx.name),
        None, None, None, None,
    )


torch.library.register_autograd(
    "autosp::all_to_all", all_to_all_backward, setup_context=all_to_all_backward_setup
)


# ============================================================================
# Graph Transformation
# ============================================================================

def insert_all_to_all(gm: GraphModule, node: Node, scatter_idx: int, gather_idx: int, name: str) -> Node:
    """Insert an all-to-all node after the given node."""
    world_size = dist.get_world_size()
    
    with gm.graph.inserting_after(node):
        a2a_node = gm.graph.call_function(
            torch.ops.autosp.all_to_all.default,
            args=(node, scatter_idx, gather_idx, world_size, name),
        )
        a2a_node.name = f"a2a_{name}"
        node.replace_all_uses_with(a2a_node)
        a2a_node.update_arg(0, node)  # Restore original input
    
    return a2a_node


def apply_autosp(gm: GraphModule) -> GraphModule:
    """
    Apply Ulysses sequence parallel transformation to a graph module.
    
    For each SDPA node:
        - Insert all-to-all before Q, K, V: scatter=1 (heads), gather=2 (seq)
        - Insert all-to-all after O: scatter=2 (seq), gather=1 (heads)
    """
    attention_nodes = list(gm.graph.find_nodes(
        op="call_function",
        target=F.scaled_dot_product_attention,
    ))
    
    for idx, attn_node in enumerate(attention_nodes):
        q, k, v = attn_node.args[:3]
        suffix = f"_{idx}" if len(attention_nodes) > 1 else ""
        
        # Before QKV: [B, N, S/P, H] -> [B, N/P, S, H]
        # scatter on heads (dim=1), gather on sequence (dim=2)
        insert_all_to_all(gm, q, scatter_idx=1, gather_idx=2, name=f"q{suffix}")
        insert_all_to_all(gm, k, scatter_idx=1, gather_idx=2, name=f"k{suffix}")
        insert_all_to_all(gm, v, scatter_idx=1, gather_idx=2, name=f"v{suffix}")
        
        # After O: [B, N/P, S, H] -> [B, N, S/P, H]
        # scatter on sequence (dim=2), gather on heads (dim=1)
        insert_all_to_all(gm, attn_node, scatter_idx=2, gather_idx=1, name=f"o{suffix}")
    
    gm.recompile()
    return gm