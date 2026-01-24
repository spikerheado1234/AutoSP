# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import List

import torch
from torch.fx import GraphModule

from ..util import get_deepcompile_handle

import torch.distributed as dist
from torch._dynamo.utils import to_fake_tensor
from ..util import get_param_nodes
from torch.fx import passes
from torch.fx.node import Node

NAME = "ulysses_compile"
seq_len = -1
seq_len_partition = -1

def add_ulysses_all2all():
    pass
