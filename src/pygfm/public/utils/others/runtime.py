"""Graph-text utils: paths, pickle, RNG, timing logs (from GraphText; baseline-agnostic)."""
from __future__ import annotations

import logging
import os
import pickle
import time
from contextlib import ContextDecorator
from functools import wraps

import numpy as np

logger = logging.getLogger(__name__)


def get_dir_of_file(f_name: str) -> str:
    return os.path.dirname(os.path.abspath(f_name))


def mkdir_p(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def init_path(dir_or_file: str) -> str:
    if dir_or_file.startswith("~"):
        dir_or_file = os.path.expanduser(dir_or_file)
    path = get_dir_of_file(dir_or_file)
    if path and not os.path.exists(path):
        mkdir_p(path)
    return dir_or_file.replace("//", "/")


def pickle_save(var, f_name: str) -> None:
    init_path(f_name)
    with open(f_name, "wb") as f:
        pickle.dump(var, f)
    logger.info("Saved %s", f_name)


def pickle_load(f_name: str):
    with open(f_name, "rb") as f:
        return pickle.load(f)


def init_random_state(seed: int = 0) -> None:
    """Delegate to :func:`pygfm.public.utils.set_seed` to avoid duplicate RNG logic."""
    from .core import set_seed

    set_seed(seed)


class time_logger(ContextDecorator):
    def __init__(self, name=None, log_func=None):
        self.name = name
        self.log_func = log_func or logger.info

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *exc):
        dt = time.time() - self.start_time
        self.log_func("%s finished in %.2fs", self.name or "block", dt)
        return False

    def __call__(self, func):
        self.name = self.name or func.__name__

        @wraps(func)
        def decorator(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return decorator
