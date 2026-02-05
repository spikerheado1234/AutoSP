"""AutoSP: Automatic Sequence Parallel (Ulysses) pass for graph modules.

Ulysses Transformation:
    Input:  [B, N, S/P, H]  (all heads, partitioned sequence)
    After A2A on QKV: [B, N/P, S, H]  (partitioned heads, full sequence)
    After SDPA: [B, N/P, S, H]
    After A2A on O: [B, N, S/P, H]  (all heads, partitioned sequence)

Where:
    B = batch size, N = num heads, S = full sequence length, H = head dim, P = world size
"""

import operator
from typing import Optional, List, Tuple, Callable

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._functorch.compile_utils import fx_graph_cse

from ..custom_ops import all_to_all
from ..fx import find_node_by_name, get_node_shape_meta, replace_node_users
from ..util import get_input_id_node, get_label_id_node, get_position_id_node, shard_tensor_node, get_sdpa_nodes, ShardingConfig

def pass_shard_seq_dim(gm: GraphModule, example_inputs):
    """
    Replace s0 with s0 // world_size for all users.
    
    This must run FIRST to avoid modifying sharding computation nodes.
    """
    input_id_node = get_input_id_node(gm)
    val = get_node_shape_meta(input_id_node)
    
    seq_symint = val.shape[1]
    assert isinstance(seq_symint, torch.SymInt), f"expected sequence dimension to be of type `torch.SymInt` but found `{type(seq_symint)}`"
    
    world_size = dist.get_world_size()
    s0_node = find_node_by_name(gm, str(seq_symint))
    
    if s0_node is None:
        print(f"WARNING: Could not find s0 node, skipping update_views_and_reshapes")
        return

    with gm.graph.inserting_after(s0_node):
        sharded_node = gm.graph.call_function(
            operator.floordiv, 
            args=(s0_node, world_size)
        )
    
    replace_node_users(s0_node, sharded_node, exclude=[sharded_node])


def pass_shard_input_ids(gm: GraphModule, example_inputs):
    """Shard input_ids tensor across ranks."""
    config = ShardingConfig.from_distributed()
    input_ids_node = get_input_id_node(gm)
    shard_tensor_node(gm, input_ids_node, config)


def pass_shard_label_ids(gm: GraphModule, example_inputs):
    """Shard label_ids tensor across ranks."""
    config = ShardingConfig.from_distributed()
    label_ids_node = get_label_id_node(gm)
    shard_tensor_node(gm, label_ids_node, config)

def pass_shard_position_ids(gm: GraphModule, example_inputs):
    """Shard label_ids tensor across ranks."""
    config = ShardingConfig.from_distributed()
    position_ids_node = get_position_id_node(gm)
    if position_ids_node is None:
        print("[WARNING] position id node not found. Skipping sharding of position ids.")
        return
    shard_tensor_node(gm, position_ids_node, config)


def pass_insert_attention_all_to_all(gm: GraphModule, real_inputs):
    """
    Insert all-to-all collectives around SDPA for Ulysses parallelism.
    
    For each SDPA:
        - Before Q, K, V: scatter heads (dim=1), gather sequence (dim=2)
        - After O: scatter sequence (dim=2), gather heads (dim=1)
    """
    world_size = dist.get_world_size()
    attention_nodes = get_sdpa_nodes(gm)
    
    def insert_a2a(node: Node, scatter_idx: int, gather_idx: int, name: str) -> Node:
        with gm.graph.inserting_after(node):
            a2a_node = gm.graph.call_function(
                torch.ops.autosp.all_to_all.default,
                args=(node, scatter_idx, gather_idx, world_size, name),
            )
            a2a_node.name = f"a2a_{name}"
            node.replace_all_uses_with(a2a_node)
            a2a_node.update_arg(0, node)
        return a2a_node
    
    for idx, attn_node in enumerate(attention_nodes):
        q, k, v = attn_node.args[:3]
        suffix = f"_{idx}" if len(attention_nodes) > 1 else ""
        
        # QKV: [B, N, S/P, H] -> [B, N/P, S, H]
        insert_a2a(q, scatter_idx=1, gather_idx=2, name=f"q{suffix}")
        insert_a2a(k, scatter_idx=1, gather_idx=2, name=f"k{suffix}")
        insert_a2a(v, scatter_idx=1, gather_idx=2, name=f"v{suffix}")
        
        # O: [B, N/P, S, H] -> [B, N, S/P, H]
        insert_a2a(attn_node, scatter_idx=2, gather_idx=1, name=f"o{suffix}")
    

def pass_canonicalize(gm: GraphModule, real_inputs):
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()

def pass_propagate_shapes(gm: GraphModule, real_inputs):
    FakeTensorProp(gm).propagate(*real_inputs)


def apply_autosp(
    gm: GraphModule, 
    real_inputs, 
    debug: bool = False,
    passes: Optional[List[Callable]] = None,
):
    """
    Apply AutoSP (Ulysses) transformation passes to the graph.
    
    Args:
        gm: GraphModule to transform
        real_inputs: Example inputs for shape propagation
        debug: If True, print graph before/after each pass
        passes: Optional custom list of passes (default: DEFAULT_PASSES)
    """

    AUTOSP_PASSES = [
        pass_shard_seq_dim,
        pass_shard_input_ids,
        pass_shard_label_ids,
        pass_shard_position_ids,
        pass_insert_attention_all_to_all,
        pass_propagate_shapes,
        pass_canonicalize,
    ]
    
    passes = passes or AUTOSP_PASSES
    rank = dist.get_rank()
    
    for p in passes:
        if debug and rank == 0:
            print(f"\n{'='*60}")
            print(f" BEFORE: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))
        
        p(gm, real_inputs)
        
        if debug and rank == 0:
            print(f"\n{'='*60}")
            print(f" AFTER: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))
    

    # fx_graph_cse(gm.graph)
