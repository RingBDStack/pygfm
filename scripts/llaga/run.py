#!/usr/bin/env python3
"""
LLaGA unified experiment entry (Python, few files). From repo root::

    python scripts/llaga/run.py train --model opt_2.7b --task nc --dataset cora --bs 4 --emb simteg --max_steps 3
    python scripts/llaga/run.py smoke --max-steps 3
    python scripts/llaga/run.py deepspeed --model vicuna --task nc --dataset cora --bs 8 --emb simteg
    python scripts/llaga/run.py yaml -c configs/llaga/smoke.yaml
    python scripts/llaga/run.py yaml-eval -c configs/llaga/02_eval_opt27b_cora.yaml   # run_yaml stage=eval
    python scripts/llaga/run.py download opt
    python scripts/llaga/run.py eval --model-path /path/to/projector --answers-file /path/to/out.jsonl --dataset arxiv --task nc

Env vars: see README for ``LLAGA_DATA_ROOT``, ``LLAGA_HF_CACHE``, ``LLAGA_CKPT_ROOT``,
``LLAGA_MODEL_BASE_OVERRIDE``, ``LLAGA_REPORT_TO``, etc.

``smoke``: default graph is ``<repo>/datasets/llaga/Cora.pt`` (see ``resolve_cora_data_path``),
with a stub OPT under ``fixtures/smoke_opt_llm``; **no** HuggingFace weight download.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from pygfm.public.repo_paths import driver_script_repo_root

_ROOT = driver_script_repo_root(__file__)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_TRAIN_MOD = "pygfm.baseline_models.llaga.train.train_mem"
_TRAIN_ENTRY = _ROOT / "pygfm/baseline_models/llaga/train/train_mem.py"


def _chdir_repo() -> None:
    os.chdir(_ROOT)
    os.environ.setdefault("LLAGA_DATA_ROOT", str(_ROOT / "datasets/llaga"))
    os.environ.setdefault("LLAGA_HF_CACHE", str(_ROOT / "ckpts/llaga/hf_cache"))


def _ckpt_base() -> Path:
    p = os.environ.get("LLAGA_CKPT_ROOT")
    return Path(p) if p else (_ROOT / "ckpts/llaga/checkpoints")


def _resolve_report_to() -> str:
    backend = os.environ.get("LLAGA_REPORT_TO", "wandb")
    if backend == "swanlab":
        os.environ["LLAGA_REPORT_TO"] = "swanlab"
        return "none"
    if backend == "none":
        return "none"
    return backend


def _wandb_offline_if_needed(report_to_arg: str) -> None:
    if report_to_arg == "wandb":
        subprocess.run(["wandb", "offline"], cwd=_ROOT, capture_output=True)


def _preset(model: str) -> dict:
    """Matches legacy train.sh branches."""
    sample_size = 10
    m = model
    if m == "vicuna":
        return dict(
            use_hop=2, template="ND", projector_type="linear",
            prefix=f"llaga-vicuna-7b-{{emb}}-2-{sample_size}-linear-projector",
            model_base="lmsys/vicuna-7b-v1.5-16k", mode="v1", max_len=4096,
        )
    if m == "vicuna_2layer":
        return dict(
            use_hop=2, template="ND", projector_type="2-layer-mlp",
            prefix=f"llaga-vicuna-7b-{{emb}}-2-{sample_size}-2-layer-mlp-projector",
            model_base="lmsys/vicuna-7b-v1.5-16k", mode="v1", max_len=4096,
        )
    if m == "vicuna_4hop":
        return dict(
            use_hop=4, template="HO", projector_type="linear",
            prefix="llaga-vicuna-7b-{emb}-4-hop-token-linear-projector",
            model_base="lmsys/vicuna-7b-v1.5-16k", mode="v1", max_len=4096,
        )
    if m == "vicuna_4hop_2layer":
        return dict(
            use_hop=4, template="HO", projector_type="2-layer-mlp",
            prefix="llaga-vicuna-7b-{emb}-4-hop-token-2-layer-mlp-projector",
            model_base="lmsys/vicuna-7b-v1.5-16k", mode="v1", max_len=4096,
        )
    if m == "llama":
        return dict(
            use_hop=2, template="ND", projector_type="linear",
            prefix=f"llaga-llama-2-7b-hf-{{emb}}-2-{sample_size}-linear-projector",
            model_base="meta-llama/Llama-2-7b-hf", mode="llaga_llama_2", max_len=4096,
        )
    if m == "llama_4hop":
        return dict(
            use_hop=4, template="HO", projector_type="linear",
            prefix="llaga-llama-2-7b-hf-{emb}-4-hop-token-linear-projector",
            model_base="meta-llama/Llama-2-7b-hf", mode="llaga_llama_2", max_len=4096,
        )
    if m == "opt_2.7b":
        return dict(
            use_hop=2, template="ND", projector_type="linear",
            prefix=f"llaga-opt-2.7b-{{emb}}-2-{sample_size}-linear-projector",
            model_base="facebook/opt-2.7b", mode="v1", max_len=1536,
        )
    if m == "opt_2.7b_4hop":
        return dict(
            use_hop=4, template="HO", projector_type="linear",
            prefix="llaga-opt-2.7b-{emb}-4-hop-token-linear-only-train-pretrain",
            model_base="facebook/opt-2.7b", mode="v1", max_len=1536,
        )
    raise SystemExit(f"Unknown --model: {model!r}; see run.py --help")


def _build_train_argv(
    model: str,
    task: str,
    dataset: str,
    bs: int,
    emb: str,
    extra: list[str],
) -> list[str]:
    pr = _preset(model)
    prefix = pr["prefix"].format(emb=emb)
    model_base = os.environ.get("LLAGA_MODEL_BASE_OVERRIDE") or pr["model_base"]
    out_dir = _ckpt_base() / dataset / f"{prefix}_{task}"
    report_arg = _resolve_report_to()
    _wandb_offline_if_needed(report_arg)
    sample_size = 10
    argv = [
        "--model_name_or_path", model_base,
        "--version", pr["mode"],
        "--cache_dir", os.environ["LLAGA_HF_CACHE"],
        "--pretrained_embedding_type", emb,
        "--tune_mm_mlp_adapter", "True",
        "--mm_use_graph_start_end", "False",
        "--mm_use_graph_patch_token", "False",
        "--bf16", "True",
        "--output_dir", str(out_dir),
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(bs),
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--eval_strategy", "no",
        "--save_strategy", "epoch",
        "--learning_rate", "2e-3",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", str(pr["max_len"]),
        "--gradient_checkpointing", "True",
        "--lazy_preprocess", "True",
        "--report_to", report_arg,
        "--use_hop", str(pr["use_hop"]),
        "--sample_neighbor_size", str(sample_size),
        "--mm_projector_type", pr["projector_type"],
        "--use_task", task,
        "--use_dataset", dataset,
        "--template", pr["template"],
    ]
    argv.extend(extra)
    return argv


def _run_train(argv: list[str]) -> int:
    cmd = [sys.executable, "-m", _TRAIN_MOD, *argv]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=_ROOT)


def cmd_train(args: argparse.Namespace, unknown: list[str]) -> int:
    print(f"PREFIX: {_preset(args.model)['prefix'].format(emb=args.emb)}")
    if os.environ.get("LLAGA_OFFLINE_SMOKE") == "1":
        from pygfm.baseline_models.llaga.fixtures.smoke_bundle import smoke_opt_llm_dir

        print("model: offline smoke (fake OPT) ->", smoke_opt_llm_dir())
    else:
        print("model:", os.environ.get("LLAGA_MODEL_BASE_OVERRIDE") or _preset(args.model)["model_base"])
    print("out:", _ckpt_base() / args.dataset / f"{_preset(args.model)['prefix'].format(emb=args.emb)}_{args.task}")
    argv = _build_train_argv(args.model, args.task, args.dataset, args.bs, args.emb, unknown)
    return _run_train(argv)


def cmd_smoke(args: argparse.Namespace, unknown: list[str]) -> int:
    os.environ["LLAGA_OFFLINE_SMOKE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ["LLAGA_REPORT_TO"] = "none"
    from pygfm.baseline_models.llaga.fixtures.smoke_bundle import ensure_smoke_llm_fixture
    from pygfm.baseline_models.llaga.paths import resolve_cora_data_path

    fix = ensure_smoke_llm_fixture()
    print("Smoke offline fake LLM:", fix)
    print("Smoke Cora graph (default):", resolve_cora_data_path(None))
    extra = ["--offline_smoke", "True", "--max_steps", str(args.max_steps), *unknown]
    ns = argparse.Namespace(model="opt_2.7b", task="nc", dataset="cora", bs=4, emb="node_x")
    return cmd_train(ns, extra + ["--node_feature_dim", "1433"])


def cmd_deepspeed(args: argparse.Namespace, unknown: list[str]) -> int:
    pr = _preset(args.model)
    prefix = pr["prefix"].format(emb=args.emb)
    model_base = os.environ.get("LLAGA_MODEL_BASE_OVERRIDE") or pr["model_base"]
    out_dir = _ckpt_base() / args.dataset / f"{prefix}_{args.task}"
    ds_cfg = Path(args.deepspeed_config).resolve()
    report_arg = "wandb"
    _wandb_offline_if_needed(report_arg)
    sample_size = 10
    train_args = [
        "--deepspeed", str(ds_cfg),
        "--model_name_or_path", model_base,
        "--version", pr["mode"],
        "--cache_dir", os.environ["LLAGA_HF_CACHE"],
        "--pretrained_embedding_type", args.emb,
        "--tune_mm_mlp_adapter", "True",
        "--mm_use_graph_start_end", "False",
        "--mm_use_graph_patch_token", "False",
        "--bf16", "True",
        "--output_dir", str(out_dir),
        "--num_train_epochs", "1",
        "--per_device_train_batch_size", str(args.bs),
        "--per_device_eval_batch_size", "4",
        "--gradient_accumulation_steps", "1",
        "--eval_strategy", "no",
        "--save_strategy", "epoch",
        "--learning_rate", "2e-3",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--tf32", "True",
        "--model_max_length", str(pr["max_len"]),
        "--gradient_checkpointing", "True",
        "--lazy_preprocess", "True",
        "--report_to", report_arg,
        "--use_hop", str(pr["use_hop"]),
        "--sample_neighbor_size", str(sample_size),
        "--mm_projector_type", pr["projector_type"],
        "--use_task", args.task,
        "--use_dataset", args.dataset,
        "--template", pr["template"],
        *unknown,
    ]
    cmd = [
        "deepspeed",
        "--include", args.include,
        "--master_port", str(args.master_port),
        str(_TRAIN_ENTRY),
        *train_args,
    ]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=_ROOT)


def cmd_yaml_eval(args: argparse.Namespace) -> int:
    """``run_yaml`` / flat temp YAML → :func:`cmd_eval` (same flags as ``run.py eval``)."""
    from pathlib import Path

    from pygfm.public.cli.yaml_config import load_yaml

    flat = load_yaml(args.config)
    if not isinstance(flat, dict):
        raise TypeError("eval YAML must load to a mapping")

    dr = flat.get("data_root")
    if dr is not None and str(dr).strip():
        p = Path(str(dr)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        os.environ["LLAGA_DATA_ROOT"] = str(p)

    cd = flat.get("cache_dir")
    if cd is not None and str(cd).strip():
        p = Path(str(cd)).expanduser()
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        else:
            p = p.resolve()
        os.environ["LLAGA_HF_CACHE"] = str(p)

    mp = flat.get("model_path")
    af = flat.get("answers_file")
    if not mp:
        raise ValueError("eval YAML must set eval.model_path (or top-level model_path)")
    if not af:
        raise ValueError("eval YAML must set eval.answers_file (or top-level answers_file)")

    ns = argparse.Namespace(
        model_path=str(mp),
        model_base=str(flat.get("model_base") or "lmsys/vicuna-7b-v1.5-16k"),
        conv_mode=str(flat.get("conv_mode") or "v1"),
        dataset=str(flat.get("dataset") or "arxiv"),
        task=str(flat.get("task") or "nc"),
        emb=str(flat.get("emb") or flat.get("pretrained_embedding_type") or "simteg"),
        use_hop=int(flat.get("use_hop", 4)),
        sample_size=int(flat.get("sample_size", 10)),
        template=str(flat.get("template") or "HO"),
        answers_file=str(af),
    )
    return cmd_eval(ns)


def cmd_eval(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-m", "pygfm.baseline_models.llaga.eval.eval_pretrain",
        "--model_path", args.model_path,
        "--model_base", args.model_base,
        "--conv_mode", args.conv_mode,
        "--dataset", args.dataset,
        "--pretrained_embedding_type", args.emb,
        "--use_hop", str(args.use_hop),
        "--sample_neighbor_size", str(args.sample_size),
        "--answers_file", args.answers_file,
        "--task", args.task,
        "--cache_dir", os.environ["LLAGA_HF_CACHE"],
        "--template", args.template,
    ]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=_ROOT)


def cmd_download(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Install first: pip install -U huggingface_hub", file=sys.stderr)
        return 1
    out_root = Path(os.environ.get("LLAGA_PRETRAINED_ROOT", str(_ROOT / "ckpts/llaga/pretrained_models")))
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    if args.kind == "vicuna":
        dest = out_root / "vicuna-7b-v1.5-16k"
        print("Downloading", "lmsys/vicuna-7b-v1.5-16k", "->", dest)
        snapshot_download("lmsys/vicuna-7b-v1.5-16k", local_dir=str(dest))
        print('Then: export LLAGA_MODEL_BASE_OVERRIDE="%s"' % dest)
    else:
        dest = out_root / "opt-2.7b"
        print("Downloading", "facebook/opt-2.7b", "->", dest)
        snapshot_download("facebook/opt-2.7b", local_dir=str(dest))
        print('Then: export LLAGA_MODEL_BASE_OVERRIDE="%s"' % dest)
        print("Then run: python scripts/llaga/run.py smoke")
    return 0


def cmd_yaml(args: argparse.Namespace) -> int:
    import yaml

    from pygfm.public.cli.export_yaml import resolve_export_path
    from pygfm.public.cli.yaml_config import load_yaml
    from pygfm.public.cli.yaml_flat_to_argv import yaml_flat_to_argv

    script_file = Path(__file__)
    if getattr(args, "export_default_yaml", None):
        tpl = {
            "_comment": "Keys match train_mem --help",
            "output_dir": "ckpts/llaga/checkpoints/yaml_smoke",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "report_to": "none",
        }
        out = resolve_export_path(args.export_default_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(tpl, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote template -> {out}")
        return 0
    data = load_yaml(args.config)
    if getattr(args, "export_run_yaml", None):
        out = resolve_export_path(args.export_run_yaml, script_file)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
        print(f"Wrote -> {out}")
        return 0
    extra = yaml_flat_to_argv(args.config)
    cmd = [sys.executable, "-m", _TRAIN_MOD, *extra]
    print(">>", " ".join(cmd))
    return subprocess.call(cmd, cwd=_ROOT)


def _add_yaml_export_flags(p: argparse.ArgumentParser) -> None:
    from pygfm.public.cli.export_yaml import add_export_yaml_arguments

    add_export_yaml_arguments(p)


def main(argv: list[str] | None = None) -> int:
    _chdir_repo()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__ or "", file=sys.stderr if not argv else sys.stdout)
        return 2 if not argv else 0
    cmd = argv[0]
    tail = argv[1:]

    if cmd == "train":
        p = argparse.ArgumentParser(prog="llaga-run train")
        p.add_argument("--model", default="vicuna", help="vicuna|vicuna_2layer|vicuna_4hop|vicuna_4hop_2layer|llama|llama_4hop|opt_2.7b|opt_2.7b_4hop")
        p.add_argument("--task", default="nc")
        p.add_argument("--dataset", default="arxiv-products")
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--emb", default="simteg")
        args, unknown = p.parse_known_args(tail)
        return cmd_train(args, unknown)

    if cmd == "smoke":
        p = argparse.ArgumentParser(prog="llaga-run smoke")
        p.add_argument("--max-steps", type=int, default=3)
        args, unknown = p.parse_known_args(tail)
        return cmd_smoke(args, unknown)

    if cmd == "deepspeed":
        p = argparse.ArgumentParser(prog="llaga-run deepspeed")
        p.add_argument("--model", default="vicuna")
        p.add_argument("--task", default="nc")
        p.add_argument("--dataset", default="arxiv-products")
        p.add_argument("--bs", type=int, default=16)
        p.add_argument("--emb", default="simteg")
        p.add_argument("--deepspeed-config", default=str(_SCRIPT_DIR / "zero2.json"))
        p.add_argument("--include", default="localhost:0,1,2,3")
        p.add_argument("--master-port", default="61000")
        args, unknown = p.parse_known_args(tail)
        return cmd_deepspeed(args, unknown)

    if cmd == "eval":
        p = argparse.ArgumentParser(prog="llaga-run eval")
        p.add_argument("--model-path", required=True)
        p.add_argument("--model-base", default="lmsys/vicuna-7b-v1.5-16k")
        p.add_argument("--conv-mode", default="v1")
        p.add_argument("--dataset", default="arxiv")
        p.add_argument("--task", default="nc")
        p.add_argument("--emb", default="simteg")
        p.add_argument("--use-hop", type=int, default=4)
        p.add_argument("--sample-size", type=int, default=10)
        p.add_argument("--template", default="HO")
        p.add_argument("--answers-file", required=True)
        args = p.parse_args(tail)
        return cmd_eval(args)

    if cmd == "download":
        p = argparse.ArgumentParser(prog="llaga-run download")
        p.add_argument("kind", choices=("vicuna", "opt"), nargs="?", default="opt")
        args = p.parse_args(tail)
        return cmd_download(args)

    if cmd == "yaml":
        p = argparse.ArgumentParser(prog="llaga-run yaml")
        p.add_argument("-c", "--config", type=str, required=True, metavar="PATH")
        _add_yaml_export_flags(p)
        args = p.parse_args(tail)
        return cmd_yaml(args)

    if cmd == "yaml-eval":
        p = argparse.ArgumentParser(prog="llaga-run yaml-eval")
        p.add_argument("-c", "--config", type=str, required=True, metavar="PATH")
        _add_yaml_export_flags(p)
        args = p.parse_args(tail)
        return cmd_yaml_eval(args)

    print(
        f"Unknown subcommand: {cmd!r}; use train|smoke|deepspeed|eval|download|yaml|yaml-eval",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
