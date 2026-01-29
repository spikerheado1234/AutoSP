# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch

from deepspeed.accelerator import get_accelerator
from .backend import make_ulysses_backend, launch_compile_passes, init_schedule

def init_ulysses(engine, backend, compile_config, compile_kwargs, schedule=None):
    return make_ulysses_backend(backend, compile_kwargs=compile_kwargs, free_activation=False)
