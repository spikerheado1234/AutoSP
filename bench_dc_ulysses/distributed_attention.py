import torch
import torch.distributed as dist
from deepspeed.sequence.layer import DistributedAttention

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
    # Ulysses core expects (batch, seq, heads, dim)
    # HF standard provides (batch, heads, seq, dim)
    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    # Initialize the Ulysses engine if it doesn't exist on this layer yet
    if not hasattr(self, "ulysses_engine"):
        self.ulysses_engine = DistributedAttention(
            torch.nn.functional.scaled_dot_product_attention,
            dist.group.WORLD,
            scatter_idx=2, # Shard sequence
            gather_idx=1   # Gather heads
        )

    # The actual distributed call
    attn_output = self.ulysses_engine(
        q, k, v,
        batch_dim_idx=0,
        attn_mask=attention_mask,
        dropout_p=dropout,
        is_causal=is_causal,
        scale=scaling
    )

    # Return to HF format: (batch, seq, heads, dim) -> (batch, heads, seq, dim)
    # Note: Transformers usually expects (B, N, S, H) back, 
    # but Llama's forward handles the reshape if we are careful.
    return attn_output, None

