"""Distributed: compute on main process + pickle sync (same idea as original GraphText)."""
from __future__ import annotations

import os
from functools import wraps

import torch.distributed as dist

from .runtime import logger, pickle_load


def get_rank() -> int:
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_world_size() -> int:
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize() -> None:
    if get_world_size() > 1:
        dist.barrier()


def process_on_master_and_sync_by_pickle(cache_arg=None, cache_kwarg=None, log_func=logger.info):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if cache_kwarg is not None:
                filename = kwargs[cache_kwarg]
            elif cache_arg is not None:
                filename = args[cache_arg]
            else:
                log_func("No cache file specified")
            skip_cache = kwargs.pop("skip_cache", False)
            if not os.path.exists(filename) or skip_cache:
                if get_rank() == 0:
                    func(*args, **kwargs)
            else:
                if get_rank() == 0:
                    log_func("Loaded cache %s, skipped %s", filename, func.__name__)
                return pickle_load(filename)
            synchronize()
            assert os.path.exists(filename), f"The {filename} must be saved in the {func.__name__}"
            return pickle_load(filename)

        return wrapper

    return decorator


def master_process_only(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        synchronize()
        return None

    return wrapper
