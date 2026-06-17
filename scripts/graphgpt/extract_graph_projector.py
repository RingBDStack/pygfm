import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from pygfm.public.cli.yaml_config import parse_args_with_optional_yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MMProjector weights")
    parser.add_argument("--model_name_or_path", type=str, help="model folder")
    parser.add_argument("--output", type=str, help="output file")
    return parse_args_with_optional_yaml(parser)


def _torch_load_bin(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _fill_ckpt_to_key_from_graph_projector_bin(
    model_dir: Path, keys_to_match: list[str], ckpt_to_key: defaultdict[str, list[str]]
) -> bool:
    """Load ``graph_projector.bin`` if present; return True if any matching keys were found."""
    gp = model_dir / "graph_projector.bin"
    if not gp.is_file():
        return False
    tensors = _torch_load_bin(gp)
    if not isinstance(tensors, dict):
        raise TypeError(f"Expected state dict in {gp}, got {type(tensors)}")
    name = "graph_projector.bin"
    for k in tensors:
        if any(m in k for m in keys_to_match):
            ckpt_to_key[name].append(k)
    return bool(ckpt_to_key[name])


def main() -> None:
    args = parse_args()

    keys_to_match = ["graph_projector", "embed_tokens", "transformer.wte"]
    ckpt_to_key: defaultdict[str, list[str]] = defaultdict(list)
    model_dir = Path(args.model_name_or_path).expanduser()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {model_dir.resolve()}")

    try:
        with open(model_dir / "pytorch_model.bin.index.json", encoding="utf-8") as f:
            model_indices = json.load(f)
        for k, v in model_indices["weight_map"].items():
            if any(key_match in k for key_match in keys_to_match):
                ckpt_to_key[v].append(k)
    except FileNotFoundError:
        if (model_dir / "pytorch_model.bin").is_file():
            v = "pytorch_model.bin"
            for k in _torch_load_bin(model_dir / v).keys():
                if any(key_match in k for key_match in keys_to_match):
                    ckpt_to_key[v].append(k)
        elif (model_dir / "model.safetensors").is_file():
            from safetensors.torch import load_file

            v = "model.safetensors"
            tensors = load_file(str(model_dir / v), device="cpu")
            for k in tensors.keys():
                if any(key_match in k for key_match in keys_to_match):
                    ckpt_to_key[v].append(k)
        elif _fill_ckpt_to_key_from_graph_projector_bin(model_dir, keys_to_match, ckpt_to_key):
            pass
        else:
            listing = ""
            try:
                names = sorted(p.name for p in model_dir.iterdir())
                listing = f" Found files: {names[:40]!r}."
            except OSError:
                pass
            raise FileNotFoundError(
                "No checkpoint weights under "
                f"{model_dir.resolve()!s}. Expected one of: "
                "pytorch_model.bin.index.json, pytorch_model.bin, model.safetensors, graph_projector.bin."
                f"{listing}"
                " Run stage-1 pretrain first (e.g. configs/graphgpt/00_smoke_pretrain_cora.yaml) "
                "with a build that saves weights to this folder."
            ) from None

    if not ckpt_to_key:
        if _fill_ckpt_to_key_from_graph_projector_bin(model_dir, keys_to_match, ckpt_to_key):
            pass
        else:
            raise FileNotFoundError(
                f"No parameter keys matching {keys_to_match} under {model_dir.resolve()} "
                f"(index.json present but no matching weight_map entries)."
            )

    loaded_weights = {}

    for ckpt_name, weight_keys in ckpt_to_key.items():
        if ckpt_name.endswith(".safetensors"):
            from safetensors.torch import load_file

            ckpt = load_file(str(model_dir / ckpt_name), device="cpu")
        else:
            ckpt = _torch_load_bin(model_dir / ckpt_name)
        for k in weight_keys:
            loaded_weights[k] = ckpt[k]

    print(loaded_weights.keys())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(loaded_weights, args.output)


if __name__ == "__main__":
    main()
