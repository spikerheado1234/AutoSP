# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import copy

import torch

from deepspeed.accelerator import get_accelerator
from .passes import ulysses_compile
from .backend import make_ulysses_backend, launch_compile_passes, init_schedule

WARMUP = 5

def init_ulysses(engine, backend, compile_config, compile_kwargs, schedule=None):
    # ############################### Use less ###################################
    if schedule is None:
        schedule = []
        schedule.append((0, [ulysses_compile.add_ulysses_all2all]))

    init_schedule(schedule)

    engine.launch_compile_passes = launch_compile_passes
    # ############################### Use less ###################################
    return make_ulysses_backend(backend, compile_kwargs=compile_kwargs, free_activation=False)
