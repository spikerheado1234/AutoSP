import os
import torch
import torch.distributed as dist
from deepspeed.sequence.layer import DistributedAttention

# Debug flags - same as autosp.py for comparison
DEBUG_DUMP_TENSORS = os.environ.get("AUTOSP_DEBUG", "0") == "1"
DEBUG_DUMP_DIR = os.environ.get("AUTOSP_DEBUG_DIR", "autosp_debug")

# 1. Define the custom interface function
def ulysses_attention_forward(
    self,  # This will be the LlamaAttention instance
    query_states,
    key_states,
    value_states,
    attention_mask=None,
    scaling=None,
    dropout=0.0,
    is_causal=True,
    **kwargs,
):
    # Ulysses expects (batch, seq, heads, dim)
    # HF standard provides (batch, heads, seq, dim)
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()
    
    # Debug: save q, k, v in [B, N, S, H] format to match autosp.py
    if DEBUG_DUMP_TENSORS:
        rank = dist.get_rank()
        os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
        # q, k, v are [B, S, N, H], convert to [B, N, S, H] for comparison with autosp
        torch.save(query_states.contiguous(), f"{DEBUG_DUMP_DIR}/gt_q_input_rank{rank}.pt")
        torch.save(key_states.contiguous(), f"{DEBUG_DUMP_DIR}/gt_k_input_rank{rank}.pt")
        torch.save(value_states.contiguous(), f"{DEBUG_DUMP_DIR}/gt_v_input_rank{rank}.pt")

    # Initialize the Ulysses engine if it doesn't exist on this layer yet
    if not hasattr(self, "ulysses_engine"):
        self.ulysses_engine = DistributedAttention(
            sdpa_wrapper,
            dist.group.WORLD,
            scatter_idx=2, # Shard heads
            gather_idx=1   # Gather sequences
        )

    # The actual distributed call
    # b, s, n, h
    attn_output = self.ulysses_engine(
        q, k, v,
        batch_dim_idx=0,
        attn_mask=None,
        dropout_p=dropout,
        is_causal=is_causal,
        scale=scaling
    )
    
    # Debug: save attention output in [B, N, S, H] format to match autosp.py
    if DEBUG_DUMP_TENSORS:
        rank = dist.get_rank()
        # attn_output is [B, S, N, H], convert to [B, N, S, H] for comparison
        torch.save(attn_output.permute(0, 2, 1, 3).contiguous(), f"{DEBUG_DUMP_DIR}/gt_o_output_rank{rank}.pt")

    # Return to HF format: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
    # Note: Transformers usually expects (B, N, S, H) back, 
    # but Llama's forward handles the reshape if we are careful.
    return attn_output, None

def sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None):
    # Permute from [b, s, n, h] to [b, n, s, h] for SDPA
    q = query.permute(0, 2, 1, 3).contiguous()
    k = key.permute(0, 2, 1, 3).contiguous()
    v = value.permute(0, 2, 1, 3).contiguous()

    # Debug: save q, k, v AFTER all-to-all (inside SDPA wrapper) in [B, N, S, H] format
    if DEBUG_DUMP_TENSORS:
        rank = dist.get_rank()
        os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
        # These are already [B, N, S, H] format
        torch.save(q, f"{DEBUG_DUMP_DIR}/gt_q_after_a2a_rank{rank}.pt")
        torch.save(k, f"{DEBUG_DUMP_DIR}/gt_k_after_a2a_rank{rank}.pt")
        torch.save(v, f"{DEBUG_DUMP_DIR}/gt_v_after_a2a_rank{rank}.pt")
    
    # For distributed attention, we cannot use the original mask since it's not distributed.
    # Use is_causal instead.
    output = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=None,  # Ignore the mask for distributed case
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True
    )
    
    # Debug: save SDPA output (before output all-to-all) in [B, N, S, H] format
    if DEBUG_DUMP_TENSORS:
        rank = dist.get_rank()
        # output is [B, N, S, H]
        torch.save(output, f"{DEBUG_DUMP_DIR}/gt_o_before_a2a_rank{rank}.pt")
    
    # Permute back from [b, n, s, h] to [b, s, n, h]
    output = output.permute(0, 2, 1, 3).contiguous()
    return output
