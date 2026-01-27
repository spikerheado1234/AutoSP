# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch

from deepspeed.accelerator import get_accelerator
from .backend import make_ulysses_backend, launch_compile_passes, init_schedule

WARMUP = 5

def dummy():
    pass

def init_ulysses(engine, backend, compile_config, compile_kwargs, schedule=None):
    # if schedule is None:
    #     schedule = []
    #     schedule.append((0, [dummy]))
    # init_schedule(schedule)
    # engine.launch_compile_passes = launch_compile_passes
    return make_ulysses_backend(backend, compile_kwargs=compile_kwargs, free_activation=False)
