"""
Export argparse namespaces to YAML; works together with ``-c`` / ``--config``.

If ``PATH`` has no directory part, it is written next to ``script_file`` (typically ``scripts/<baseline>/``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import AbstractSet, Any

EXPORT_ARG_DESTS = frozenset({"export_default_yaml", "export_run_yaml"})


def resolve_export_path(maybe_path: str | Path, script_file: Path | None) -> Path:
    """
    Bare filename (e.g. ``pretrain_smoke.yaml``) → ``<script_dir>/pretrain_smoke.yaml``;
    otherwise resolve relative to the current working directory.
    """
    raw = os.path.normpath(str(maybe_path).strip())
    parent, name = os.path.split(raw)
    if script_file is not None and (not parent or parent in (".", "")):
        return script_file.resolve().parent / name
    return Path(raw).expanduser()


def add_export_yaml_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--export-default-yaml",
        default=None,
        metavar="PATH",
        help="Write defaults + CLI-overridden args to YAML and exit",
    )
    parser.add_argument(
        "--export-run-yaml",
        default=None,
        metavar="PATH",
        help="Write fully resolved args (after -c and CLI) to YAML and exit",
    )


def namespace_from_parser_defaults(parser: argparse.ArgumentParser) -> argparse.Namespace:
    d: dict[str, Any] = {}
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if dest in (None, "help"):
            continue
        dv = getattr(action, "default", None)
        if dv is argparse.SUPPRESS:
            continue
        if (dv is not None) or (not getattr(action, "required", False)):
            d[dest] = dv
        else:
            d[dest] = None
    return argparse.Namespace(**d)


def _strip_config_and_export_flags(argv: list[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-c", "--config"):
            i += 2
            continue
        if a in ("--export-default-yaml", "--export-run-yaml"):
            i += 2
            continue
        if a.startswith("--export-default-yaml=") or a.startswith("--export-run-yaml="):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _argv_mentions_option(tokens: list[str], option_string: str) -> bool:
    if not option_string.startswith("--"):
        return option_string in tokens
    eq = option_string + "="
    for t in tokens:
        if t == option_string or t.startswith(eq):
            return True
    return False


def _explicit_option_dests_from_argv(parser: argparse.ArgumentParser, argv: list[str]) -> set[str]:
    tokens = _strip_config_and_export_flags(argv)
    explicit: set[str] = set()
    for action in parser._actions:
        dest = getattr(action, "dest", None)
        if dest in (None, "help"):
            continue
        for opt in getattr(action, "option_strings", ()) or ():
            if opt and _argv_mentions_option(tokens, opt):
                explicit.add(dest)
                break
    return explicit


def namespace_for_default_yaml_export(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]
    base = namespace_from_parser_defaults(parser)
    explicit = _explicit_option_dests_from_argv(parser, argv)
    merged = argparse.Namespace(**vars(base))
    for dest in explicit:
        if dest in EXPORT_ARG_DESTS:
            continue
        if hasattr(args, dest):
            setattr(merged, dest, getattr(args, dest))
    return merged


def namespace_to_plain_dict(
    ns: argparse.Namespace,
    exclude: AbstractSet[str] | None = None,
) -> dict[str, Any]:
    ex = EXPORT_ARG_DESTS | (exclude or frozenset())
    return {k: v for k, v in vars(ns).items() if k not in ex}


def dump_namespace_to_yaml(
    ns: argparse.Namespace,
    path: str | Path,
    *,
    exclude: AbstractSet[str] | None = None,
    script_file: Path | None = None,
) -> None:
    try:
        import yaml
    except ImportError as e:
        raise SystemExit("YAML export requires PyYAML: pip install pyyaml") from e
    path = resolve_export_path(path, script_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = namespace_to_plain_dict(ns, exclude=exclude)
    path.write_text(
        yaml.safe_dump(
            data,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )


def handle_export_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    argv: list[str] | None = None,
    *,
    script_file: Path | None = None,
) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if args.export_default_yaml:
        dump_namespace_to_yaml(
            namespace_for_default_yaml_export(parser, args, argv),
            args.export_default_yaml,
            script_file=script_file,
        )
        sys.exit(0)
    if args.export_run_yaml:
        dump_namespace_to_yaml(
            args,
            args.export_run_yaml,
            exclude=EXPORT_ARG_DESTS | {"config"},
            script_file=script_file,
        )
        sys.exit(0)
