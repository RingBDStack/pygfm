"""
pygfm: An AI copilot for graph data and models.

This package provides tools and utilities for working with Graph Foundation Models.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("python-gfm")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__author__ = "BUAA SKLCCSE"


def print_gfm_style(spacing: int = 0) -> None:
    """Print the GFM ASCII banner; ``spacing`` is the gap width between G, F, and M columns."""
    g_lines = [
        " ____ ",
        "/ ___|",
        "| |  _",
        "| |_| |",
        " \\____|",
    ]
    f_lines = [
        "  _____ ",
        " |  ___|",
        " | |_   ",
        "|  _|  ",
        "|_|    ",
    ]
    m_lines = [
        " __  __ ",
        "|  \\/  |",
        "| |\\/| |",
        "| |  | |",
        "|_|  |_|",
    ]

    gap = " " * spacing
    print("Welcome to")
    for i in range(5):
        print(f"{g_lines[i]}{gap}{f_lines[i]}{gap}{m_lines[i]}")


__all__ = ["__version__", "print_gfm_style"]

# NOTE:
# Keep top-level import lightweight. Some baselines (e.g. graphgpt) have optional
# assets/deps and should not be imported as a side-effect of `import pygfm`.

print_gfm_style(1)  # Balanced spacing on package import

