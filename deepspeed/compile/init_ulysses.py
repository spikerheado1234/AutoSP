# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
from .backend import make_ulysses_backend

def init_ulysses(engine, backend, compile_config, compile_kwargs, schedule=None):
    return make_ulysses_backend(backend, compile_kwargs=compile_kwargs, free_activation=False)
