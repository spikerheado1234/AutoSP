import torch
import torch.distributed as dist
import os
from torch.distributed._functional_collectives import all_to_all_inplace

def post_all2all(scatter_idx, batch_dim_idx, seq_world_size, bs, seq_len, num_head, head_dim):
    def post_func(input):
        if scatter_idx < 2:
            output = input.permute(1, 2, 0, 3, 4).contiguous()
            output = output.reshape(bs, seq_len // seq_world_size, seq_world_size * num_head, head_dim).contiguous()
        else:
            output = input.permute(1, 0, 2, 3, 4).contiguous()
            output = output.reshape(bs, seq_world_size * seq_len, num_head // seq_world_size, head_dim).contiguous()
        return output
    return post_func


def single_all_to_all(input, scatter_idx, gather_idx, batch_dim_idx, group, async_op=False, handle=None, type=None):
    seq_world_size = dist.get_world_size(group)
    num_heads = input.shape[2]
    if scatter_idx < 2:
        bs, global_seq_len, num_local_head, head_dim = input.shape
        input_t = input.reshape([bs, seq_world_size, global_seq_len // seq_world_size, num_local_head, head_dim]).contiguous()
        input_t = input_t.permute(1, 0, 2, 3, 4).contiguous()
    else:
        bs, local_seq_len, num_total_head, head_dim = input.shape
        assert num_total_head % seq_world_size == 0
        input_t = input.reshape([bs, local_seq_len, seq_world_size, num_total_head // seq_world_size, head_dim]).contiguous()
        input_t = input_t.permute(2, 0, 1, 3, 4).contiguous()

    if scatter_idx < 2:
        post_all2all_fun = post_all2all(scatter_idx, batch_dim_idx, seq_world_size, bs, global_seq_len, num_local_head, head_dim)
    else:
        post_all2all_fun = post_all2all(scatter_idx, batch_dim_idx, seq_world_size, bs, local_seq_len, num_total_head, head_dim)

    output = torch.empty_like(input_t)
    work = dist.all_to_all_single(output, input_t, group=group, async_op=async_op)

    if async_op:
        if type in ('dq', 'dk'):
            handle[type + '_work'] = work
            handle[type + '_grad'] = output
            handle[type + '_post_all2all_func'] = post_all2all_fun
            return output

    res = post_all2all_fun(output)
    return res

def all_to_all_qkv(input_tensor) -> torch.Tensor:
    B, global_heads, local_seq_len, head_dim = input_tensor.shape
    
    world_size = 2
    local_heads = global_heads // world_size
    global_seq_len = local_seq_len * world_size

    input_t = input_tensor.reshape([B, world_size, local_heads, local_seq_len, head_dim]).contiguous() # b, w, n // w, s, h
    input_t = input_t.permute(1, 0, 2, 3, 4).contiguous() # w, b, n // w, s, h
    output = torch.empty_like(input_t) # w, b, n // w, s, h
    all_to_all_inplace(output, input_t) # w, b, n // w, s, h
    output = output.permute(1, 2, 0, 3, 4).contiguous() # b, n // w, w, s, h
    output = output.reshape(B, local_heads, global_seq_len, head_dim).contiguous() # b, n // w, w * s, h
    return output # b, n // w, w * s, h

def print_debug_0(msg):
    rank = dist.get_rank()
    if rank == 0:
        print(msg)

def main():
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    group = dist.group.WORLD

    torch.manual_seed(0)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    b = 1
    n = 4
    s = 6
    h = 1

    query = torch.arange(b * n * s * h, dtype=torch.float32).reshape(b, n, s, h)
    query = query[:, :, rank * s // world_size : (rank + 1) * s // world_size].to(device)

    print_debug_0(f"[Rank 0] query slice:\n{query}")

    print_debug_0("============== COMPILED ULYSSES ================")
    query_states_sbnh = query.permute(0, 2, 1, 3).contiguous().to(device)
    print_debug_0(f"[Rank 0] query_states (after permute to [b, s_local, n, h]):\n{query_states_sbnh.cpu()}")

    output = single_all_to_all(query_states_sbnh, scatter_idx=2, gather_idx=1, batch_dim_idx=0, group=group)
    output = output.permute(0, 2, 1, 3)
    print_debug_0(f"[Rank 0] all2all final result:\n{output.cpu()}")


    print_debug_0("============== DEEP COMPILED ULYSSES =============")
    output = all_to_all_qkv(query)
    print_debug_0(f"[Rank 0] all2all final result:\n{output.cpu()}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
