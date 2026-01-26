import torch
import torch.distributed as dist

GROUP_REGISTRY = {} # int -> dist.ProcessGroup

def register_groups(groups):
    """groups: List[List[int]], e.g. [[0,1],[2,3]]"""
    for gid, ranks in enumerate(groups):
        if gid not in GROUP_REGISTRY:
            GROUP_REGISTRY[gid] = dist.new_group(ranks)

def get_group(gid: int):
    return GROUP_REGISTRY[gid] if gid is not None else dist.group.WORLD

def get_registry():
    return GROUP_REGISTRY
