"""Support ``python -m pygfm -c config.yaml`` and ``python -m pygfm download ...``."""

from __future__ import annotations

import sys


def _entry() -> None:
    rest = sys.argv[1:]
    if rest and rest[0] == "download":
        from pygfm.tool_download import main as download_main

        download_main(rest[1:])
        return
    from pygfm.cli.run_yaml import main as run_yaml_main

    run_yaml_main(rest)


if __name__ == "__main__":
    _entry()
