"""Support ``python -m pygfm -c config.yaml`` (same as the ``pygfm`` / ``gfm`` entry points)."""

from pygfm.cli.run_yaml import main

if __name__ == "__main__":
    main()
