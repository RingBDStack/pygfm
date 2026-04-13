"""Shim: attack scripts that use `from lib.paths import paths` with attack_data_gen on sys.path."""

from pygfm.baseline_models.sa2gfm.paths import Paths, paths  # noqa: F401

__all__ = ["Paths", "paths"]
