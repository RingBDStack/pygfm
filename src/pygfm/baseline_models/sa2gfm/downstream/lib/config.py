"""Path-aware argparse for downstream MoE finetuning (replaces hard-coded home paths)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from pygfm.baseline_models.sa2gfm.paths import paths
from pygfm.public.utils.legacy_pickles import register_pygfm_unpickling_aliases
from pygfm.public.cli.export_yaml import add_export_yaml_arguments, handle_export_args
from pygfm.public.cli.yaml_config import load_yaml, merge_yaml_defaults


# Paths materialized by ``--export-run-yaml`` may reappear in a follow-up ``-c`` file; ignore them when merging.
_YAML_IGNORE_DERIVED = frozenset(
    {
        "data_path",
        "pre_train_model_dir_single",
        "pre_train_model_dir_many",
        "communitys_dir",
        "community_file",
        "down_data_dir",
        "txt_features",
        "num_nodes",
        "num_classes",
    }
)

# Feature / label key priority when loading saved graphs (GFM exports, OGB-style aliases).
GRAPH_FEATURE_KEYS = (
    "enhanced_x_64",
    "enhanced_x",
    "x",
    "feat",
    "features",
    "node_feat",
    "node_features",
    "attr",
    "attrs",
)
LABEL_KEYS = ("y", "label", "labels")
EDGE_KEYS = ("edge_index", "edges", "edge_index_")


def unwrap_saved_graph(obj, depth: int = 0):
    """
    Unwrap common containers (``[obj]``, ``{"data": ...}``, etc.) to reach ``x`` / ``edge_index``.
    """
    if depth > 10 or obj is None:
        return obj
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        return unwrap_saved_graph(obj[0], depth + 1)
    if isinstance(obj, dict):
        for k in ("data", "graph", "g", "raw", "sample", "object"):
            if k in obj and obj[k] is not None:
                return unwrap_saved_graph(obj[k], depth + 1)
    return obj


def normalize_sa2gfm_loaded_object(raw):
    """
    After ``torch.load``: unwrap; normalize PyG InMemory tuples like ``(dict, slices, cls)`` to a single ``Data``.
    """
    raw = unwrap_saved_graph(raw)
    if isinstance(raw, tuple):
        try:
            from pygfm.public.utils.runtime import _torch_load_to_single_data

            raw = _torch_load_to_single_data(raw)
        except Exception:
            pass
    return raw


def graph_attr(obj, *names: str, required: bool = True):
    """
    Read the first non-empty attribute from a PyG ``Data`` or ``dict`` (common ``torch.save`` layout).
    Dict keys are matched case-insensitively (e.g. ``X`` / ``x``).
    """
    if isinstance(obj, dict):
        key_cf = {str(k).casefold(): k for k in obj.keys()}
        for name in names:
            if name in obj and obj[name] is not None:
                return obj[name]
            lk = name.casefold()
            if lk in key_cf:
                k = key_cf[lk]
                if obj[k] is not None:
                    return obj[k]
    else:
        for k in names:
            if hasattr(obj, k):
                v = getattr(obj, k)
                if v is not None:
                    return v
    if required:
        need = " / ".join(names)
        raise ValueError(f"Graph object has none of: {need} (type={type(obj).__name__})")
    return None


def graph_feature_first(obj):
    return graph_attr(obj, *GRAPH_FEATURE_KEYS, required=False)


def graph_label_first(obj):
    return graph_attr(obj, *LABEL_KEYS, required=True)


def graph_edge_index_first(obj):
    return graph_attr(obj, *EDGE_KEYS, required=True)


def get_pretrain_datasets(dataset: str) -> list:
    data_name = (dataset or "cora").strip().lower()
    datasets = []
    if data_name == "cora":
        datasets = ["citeseer", "pubmed", "P-home", "wikics"]
    elif data_name == "citeseer":
        datasets = ["cora", "pubmed", "P-home", "wikics"]
    elif data_name == "pubmed":
        datasets = ["cora", "citeseer", "P-home", "wikics"]
    elif data_name == "P-tech":
        datasets = ["cora", "citeseer", "pubmed", "P-home", "wikics"]
    elif data_name == "P-home":
        datasets = ["cora", "citeseer", "pubmed", "wikics"]
    elif data_name == "wikics":
        datasets = ["cora", "citeseer", "pubmed", "P-home"]
    elif data_name == "arxiv":
        datasets = ["P-home", "P-tech", "wikics"]
    return datasets


from pygfm.public.utils import set_seed as _set_seed


def _torch_load(path: str):
    register_pygfm_unpickling_aliases()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _preview_keys_for_error(data) -> str:
    if isinstance(data, dict):
        keys = list(data.keys())
        return f"dict keys (first 40): {keys[:40]!r}"
    return f"type={type(data).__name__}"


def _num_nodes_and_classes(data_path: str):
    data = normalize_sa2gfm_loaded_object(_torch_load(data_path))
    feat = graph_feature_first(data)
    if feat is None:
        raise ValueError(
            f"No usable node features in {data_path} "
            f"(tried {GRAPH_FEATURE_KEYS}). {_preview_keys_for_error(data)}"
        )
    n = int(np.asarray(feat).shape[0])
    y = graph_label_first(data)
    y_t = torch.as_tensor(y)
    if y_t.dim() > 1 and y_t.size(-1) > 1:
        y_t = y_t.argmax(dim=-1)
    else:
        y_t = y_t.view(-1)
    c = int(y_t.max().item()) + 1
    return n, c


def get_args():
    parser = argparse.ArgumentParser("SA2GFM_downstream")
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--seed", type=int, default=39)
    parser.add_argument("--shot_num", type=int, default=1)
    parser.add_argument(
        "--unify_dim",
        type=int,
        default=64,
        help="Must match node feature dim; raw x (e.g. 1433) is auto-adjusted at runtime when needed",
    )
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.0)
    parser.add_argument("--lambda_var", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gcn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--out_channels", type=int, default=64)
    parser.add_argument("--hid_units", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Train-only optimization steps per split; test accuracy computed once after all epochs",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=-1,
        help="If >= 0, only run this few-shot split index; -1 = all splits up to --num-splits",
    )
    parser.add_argument("--moe_weight", type=float, default=0.1)
    parser.add_argument("--structure_weight", type=float, default=0.1)
    parser.add_argument("--bucket_boundaries", type=int, nargs="+", default=[30, 100])
    parser.add_argument("--inter_cluster_optimizer", default=True)
    parser.add_argument("--appnp_alpha", type=float, default=0.1)
    parser.add_argument("--appnp_k", type=int, default=10)
    parser.add_argument("--inter_cluster_threshold", type=float, default=0.5)
    parser.add_argument("--inter_cluster_temperature", type=float, default=10.0)
    parser.add_argument("--moe_embedding_weight", type=float, default=0.1)
    parser.add_argument("--multi_embedding_weight", type=float, default=0.9)
    parser.add_argument(
        "--attack_type",
        default="none",
        choices=["none", "random", "targeted_poisoning", "targeted_evasion"],
    )
    parser.add_argument("--attack_ratio", type=float, default=0.1)
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--random_attack_type", default="feature", choices=["feature", "structure"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--num-splits",
        type=int,
        default=-1,
        help="Number of few-shot splits to run (-1: 20 for all datasets)",
    )
    parser.add_argument(
        "--split-pool-size",
        type=int,
        default=-1,
        help="If >0, run splits in [split-pool-start, split-pool-start + split-pool-size)",
    )
    parser.add_argument(
        "--split-pool-start",
        type=int,
        default=0,
        help="Start index when --split-pool-size is set (default 0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="If >0, after the run print summary for the top-k splits by test accuracy",
    )
    parser.add_argument(
        "--results-json",
        type=str,
        default="",
        help="Optional path to save per-split results and top-k summary as JSON",
    )
    parser.add_argument(
        "--split-ids-json",
        type=str,
        default="",
        help='JSON file with a "split_ids" list; run only those few-shot splits',
    )
    parser.add_argument("--no-swanlab", action="store_true")

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", type=str, default=None)
    pre_args, rest = pre.parse_known_args(sys.argv[1:])
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        dest="config",
        help="YAML config path (CLI overrides YAML)",
    )
    add_export_yaml_arguments(parser)
    if pre_args.config:
        merge_yaml_defaults(
            parser,
            load_yaml(pre_args.config),
            ignore_keys=_YAML_IGNORE_DERIVED,
        )
    args = parser.parse_args(rest)
    if getattr(args, "config", None) is None and pre_args.config is not None:
        args.config = pre_args.config

    dr = paths.data_root
    ds = args.dataset.strip()
    ds_key = ds.lower()

    if args.attack_type == "none":
        args.data_path = str(paths.resolve_ori_graph_pt(ds))
    elif args.attack_type == "targeted_poisoning":
        sub = paths.attack_post_dir / f"{ds_key}_p{args.p}_final"
        args.data_path = str(sub / f"{ds_key}_poisoning_final.pt")
    elif args.attack_type == "targeted_evasion":
        sub = paths.attack_post_dir / f"{ds_key}_p{args.p}_final"
        args.data_path = str(sub / f"{ds_key}_evasion_final.pt")
    elif args.attack_type == "random":
        if args.random_attack_type == "feature":
            fname = f"{ds_key}_feature_p{args.attack_ratio}.pt"
        else:
            fname = f"{ds_key}_structure_p{args.attack_ratio}.pt"
        primary = paths.attack_random_dir / fname
        fallback = paths.data_root / "attacked_data_random" / fname
        if primary.is_file():
            args.data_path = str(primary)
        elif fallback.is_file():
            args.data_path = str(fallback)
        else:
            args.data_path = str(primary)
    else:
        raise ValueError(args.attack_type)

    args.pre_train_model_dir_single = str(paths.save_model_dir)
    args.pre_train_model_dir_many = str(paths.save_model_many_dir)
    args.communitys_dir = str(paths.communities_dir)
    args.community_file = str(paths.communities_dir / f"{ds_key}_communities.pt")
    args.down_data_dir = str(paths.few_shot_dir / ds_key / f"{args.shot_num}shot")

    emb_path = paths.reduced_embeddings_dir / f"{ds_key}_reduced_embeddings.pt"
    if emb_path.is_file():
        args.txt_features = _torch_load(str(emb_path))["embeddings"]
    else:
        args.txt_features = None

    _set_seed(args.seed)
    args.num_nodes, args.num_classes = _num_nodes_and_classes(args.data_path)

    if args.num_splits < 0:
        args.num_splits = 20
    _ts_script = Path(__file__).resolve().parents[1] / "pipeline" / "train_downstream.py"
    handle_export_args(parser, args, rest, script_file=_ts_script)
    return args
