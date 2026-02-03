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
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.fx import GraphModule, Node
from torch.fx.passes.fake_tensor_prop import FakeTensorProp

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

def insert_attention_all_to_all(gm: GraphModule, real_inputs):
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
    
    gm.graph.eliminate_dead_code()
    gm.recompile()


def find_input_ids(gm: GraphModule) -> Node:
    """
    Find input_ids by tracing backwards from SDPA -> Q -> ... -> placeholder.
    Returns the first int64 2D placeholder found (which should be input_ids).
    """
    attention_nodes = list(gm.graph.find_nodes(
        op="call_function",
        target=F.scaled_dot_product_attention,
    ))
    assert len(attention_nodes) > 0, "No SDPA nodes found in graph"
    
    q_node = attention_nodes[0].args[0]
    q_val = q_node.meta.get("val") or q_node.meta.get("example_value")
    assert q_val is not None, "Q node has no shape metadata"
    B, S = q_val.shape[0], q_val.shape[2]
    
    visited = set()
    
    def dfs(node) -> Node | None:
        if node in visited or not isinstance(node, Node):
            return None
        
        visited.add(node)
        
        if node.op == "placeholder":
            val = node.meta.get("val") or node.meta.get("example_value")
            if val is not None and val.ndim == 2 and val.shape == (B, S):
                return node
        
        for arg in node.args:
            result = dfs(arg)
            if result is not None:
                return result
        
        return None
    
    result = dfs(q_node)
    
    if result is None:
        raise RuntimeError(f"Could not find input_ids: no 2D placeholder with shape ({B}, {S}) found in path from SDPA")
    
    return result

def find_label_ids(gm: GraphModule) -> Node:
    for node in gm.graph.nodes:
        if node.name == 'l_kwargs_labels_':
            return node
    raise RuntimeError("Node with name 'l_kwargs_labels_' not found in graph")

def find_symbolic_placeholders(gm: GraphModule, symint: torch.SymInt):
    """Find all placeholder nodes that represent the given SymInt."""
    s = str(symint)
    placeholders = []
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            if node.name == s:
                placeholders.append(node)
            else:
                val = node.meta.get("val")
                if isinstance(val, torch.SymInt) and str(val) == s:
                    placeholders.append(node)
    return placeholders

def shard_across_ranks(gm: GraphModule, example_inputs, input_ids_node):
    """
    Shard input_ids along sequence dimension based on rank.
    rank 0 -> input_ids[:, 0:S//P]
    rank 1 -> input_ids[:, S//P:2*S//P]
    """

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    val = input_ids_node.meta.get("val") or input_ids_node.meta.get("example_value")
    assert val is not None, "input_ids node has no shape metadata"
    
    seq_len = val.shape[1]
    
    if isinstance(seq_len, torch.SymInt):
        placeholders = find_symbolic_placeholders(gm, seq_len)
        if not placeholders:
            # Fallback to current node name if it matches
            if input_ids_node.name == str(seq_len):
                s0_node = input_ids_node
            else:
                raise RuntimeError(f"Could not find placeholder for symbolic sequence length {seq_len}")
        else:
            s0_node = placeholders[0]
        
        # Insert shared indices computation after the placeholder
        # We use explicit insertion points to avoid topological order issues
        with gm.graph.inserting_after(s0_node):
            chunk_size_node = gm.graph.call_function(operator.floordiv, args=(s0_node, world_size))
        with gm.graph.inserting_after(chunk_size_node):
            start_node = gm.graph.call_function(operator.mul, args=(rank, chunk_size_node))
        with gm.graph.inserting_after(start_node):
            end_node = gm.graph.call_function(operator.add, args=(start_node, chunk_size_node))
        with gm.graph.inserting_after(end_node):
            s1 = gm.graph.call_function(slice, args=(None, None, None))
        with gm.graph.inserting_after(s1):
            s2 = gm.graph.call_function(slice, args=(start_node, end_node, None))
            indices = (s1, s2)
    else:
        assert seq_len % world_size == 0, f"Sequence length {seq_len} not divisible by world_size {world_size}"
        chunk_size = seq_len // world_size
        start_idx = rank * chunk_size
        end_idx = start_idx + chunk_size
        indices = (slice(None), slice(start_idx, end_idx))
    
    # Insert getitem after the input node
    with gm.graph.inserting_after(input_ids_node):
        sliced_node = gm.graph.call_function(
            operator.getitem,
            args=(input_ids_node, indices),
        )
        
        # Replace all uses of the original input with the sliced one, 
        # except the sliced_node itself which obviously needs the original input.
        to_replace = [u for u in input_ids_node.users if u != sliced_node]
        for user in to_replace:
            user.replace_input_with(input_ids_node, sliced_node)

def update_views_and_reshapes(gm: GraphModule, example_inputs):
    """
    Replace all view/reshape operations that use s0 (full sequence length) 
    with operations using s0 // world_size (sharded sequence length).
    
    s0 is a SymInt placeholder, so we work with SymInt arithmetic directly.
    """
    input_id_node = find_input_ids(gm)     
    val = input_id_node.meta.get("val") or input_id_node.meta.get("example_value")
    
    # Get the actual SymInt value from the tensor shape
    seq_symint = val.shape[1]  # This is a SymInt like Sym(s0)
    print(f"DEBUG: seq_symint = {seq_symint}, type = {type(seq_symint)}")
    
    world_size = dist.get_world_size()
    
    placeholders_to_shard = find_symbolic_placeholders(gm, seq_symint)
    if not placeholders_to_shard:
        print(f"WARNING: Could not find any placeholder nodes for {seq_symint}, skipping update_views_and_reshapes")
        return

    for ph in placeholders_to_shard:
        with gm.graph.inserting_after(ph):
            sharded_node = gm.graph.call_function(operator.floordiv, args=(ph, world_size))
        
        # Replace all original uses of ph with ph // world_size.
        # We must avoid replacing ph in the newly created sharded_node.
        to_replace = [u for u in ph.users if u != sharded_node]
        for user in to_replace:
            user.replace_input_with(ph, sharded_node)
    
    print(f"DEBUG: Updated views and reshapes for {len(placeholders_to_shard)} symbolic placeholders.")
    
def shard_input_ids(gm: GraphModule, example_inputs):
    input_ids_node = find_input_ids(gm)
    shard_across_ranks(gm, example_inputs, input_ids_node)

def shard_label_ids(gm: GraphModule, example_inputs):
    label_ids_node = find_label_ids(gm)
    shard_across_ranks(gm, example_inputs, label_ids_node)

def replace_sp_mask(gm: GraphModule, real_inputs):
    pass

def apply_autosp(gm: GraphModule, real_inputs, debug: bool = False):
    """
    Apply AutoSP (Ulysses) passes to the graph.
    
    Passes:
    1. shard_sequence: Slice input_ids and fix hardcoded S constants
    2. insert_attention_all_to_all: Insert A2A around SDPA
    3. replace_sp_mask: Replace attention masks for SP context
    """
    gm.graph.eliminate_dead_code()
    passes = [
        update_views_and_reshapes,
        shard_input_ids,
        shard_label_ids,
        insert_attention_all_to_all,
    ]
    
    for p in passes:
        if debug and dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f" BEFORE: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))
        p(gm, real_inputs)
        if debug and dist.get_rank() == 0:
            print(f"\n{'='*60}")
            print(f" AFTER: {p.__name__}")
            print(f"{'='*60}\n")
            print(gm.print_readable(print_output=False))
    
    FakeTensorProp(gm).propagate(*real_inputs)
    gm.graph.lint()
    gm.recompile()
    print("LAST!!!!")
    print(gm.print_readable(print_output=False))