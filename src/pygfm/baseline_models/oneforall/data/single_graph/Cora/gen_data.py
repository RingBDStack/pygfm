from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric as pyg
from torch_geometric.data import Data

from pygfm.baseline_models.oneforall.data.torch_io import torch_load_compat
from pygfm.baseline_models.oneforall.paths import get_data_root

# Planetoid Cora 7 classes (fallback without categories.csv)
_CORA_TOPIC_NAMES = (
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
)


def _cora_pt_candidates() -> list[Path]:
    root = get_data_root()
    return [
        root / "Cora.pt",
        root / "cora.pt",
        root / "single_graph" / "Cora" / "cora.pt",
    ]


def _load_pyg_data(path: Path) -> Data:
    obj = torch_load_compat(str(path))
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, tuple) and len(obj) >= 1:
        first = obj[0]
        if isinstance(first, Data):
            return first
        if isinstance(first, dict):
            return Data.from_dict(first)
    if isinstance(obj, dict):
        return Data.from_dict(obj)
    raise TypeError(f"Unsupported Cora graph file {path}: type {type(obj)}")


def get_logic_label(ordered_txt):
    or_labeled_text = []
    not_and_labeled_text = []
    for i in range(len(ordered_txt)):
        for j in range(len(ordered_txt)):
            c1 = ordered_txt[i]
            c2 = ordered_txt[j]
            d1, d2 = str(np.asarray(c1[1]).reshape(-1)[0]), str(np.asarray(c2[1]).reshape(-1)[0])
            txt = "prompt node. literature category and description: not " + c1[0] + ". " + d1 + " and not " + c2[
                0
            ] + ". " + d2
            not_and_labeled_text.append(txt)
            txt = "prompt node. literature category and description: either " + c1[0] + ". " + d1 + " or " + c2[
                0
            ] + ". " + d2
            or_labeled_text.append(txt)
    return or_labeled_text + not_and_labeled_text


def _ordered_desc_from_csv(base: Path) -> list:
    df = pd.read_csv(base / "categories.csv", sep=",")
    ordered_desc = []
    for _, row in df.iterrows():
        ordered_desc.append((str(row.iloc[0]), np.array([str(row.iloc[1])], dtype=object)))
    return ordered_desc


def _ordered_desc_synthetic(num_classes: int) -> list:
    ordered_desc = []
    for c in range(num_classes):
        name = _CORA_TOPIC_NAMES[c] if c < len(_CORA_TOPIC_NAMES) else f"class_{c}"
        desc = f"Scientific publication category {name}."
        ordered_desc.append((name, np.array([desc], dtype=object)))
    return ordered_desc


def _ensure_cite_split_masks(data: Data) -> None:
    """Planetoid often uses ``train_mask`` / ``val_mask`` / ``test_mask``; OFA ``CiteSplitter`` expects ``*_masks`` lists."""
    if getattr(data, "train_mask", None) is not None and getattr(data, "train_masks", None) is None:
        data.train_masks = [data.train_mask.view(-1).bool()]
    if getattr(data, "val_mask", None) is not None and getattr(data, "val_masks", None) is None:
        data.val_masks = [data.val_mask.view(-1).bool()]
    if getattr(data, "test_mask", None) is not None and getattr(data, "test_masks", None) is None:
        data.test_masks = [data.test_mask.view(-1).bool()]


def _synthetic_raw_texts(data: Data) -> list[str]:
    n = int(data.num_nodes)
    fdim = int(data.x.size(-1)) if data.x is not None else 0
    # Body only; outer code adds feature-node prefix
    return [
        f"document {i} with {fdim}-dimensional bag-of-words features." for i in range(n)
    ]


def get_data(dset):
    candidates = _cora_pt_candidates()
    base = None
    path = None
    for p in candidates:
        if p.is_file():
            path = p
            base = p.parent
            break
    if path is None:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            "Cora graph file not found. Place ``Cora.pt`` (PyG ``Data`` or ``Data.from_dict``) under "
            f"``{get_data_root()}`` or legacy ``.../single_graph/Cora/cora.pt``. Tried: {tried}"
        )

    data = _load_pyg_data(path)
    nx_g = pyg.utils.to_networkx(data, to_undirected=True)
    edge_index = torch.tensor(list(nx_g.edges())).T
    print(edge_index.size())
    data_dict = data.to_dict()
    data_dict["edge_index"] = edge_index
    new_data = Data(**data_dict)
    _ensure_cite_split_masks(new_data)

    use_csv = (base / "categories.csv").is_file()

    if hasattr(data, "raw_texts") and data.raw_texts is not None:
        text = data.raw_texts
        if isinstance(text, torch.Tensor):
            text = text.tolist()
    else:
        text = _synthetic_raw_texts(new_data)

    if use_csv:
        ordered_desc = _ordered_desc_from_csv(base)
    else:
        y = new_data.y.view(-1).long()
        num_classes = int(y.max().item()) + 1
        ordered_desc = _ordered_desc_synthetic(num_classes)

    clean_text = ["feature node. paper title and abstract: " + t for t in text]
    label_text = [
        "prompt node. literature category and description: "
        + desc[0]
        + "."
        + str(np.asarray(desc[1]).reshape(-1)[0])
        for desc in ordered_desc
    ]
    edge_label_text = [
        "prompt node. two papers do not have co-citation",
        "prompt node. two papers have co-citation",
    ]
    logic_label_text = get_logic_label(ordered_desc)
    edge_text = [
        "feature edge. connected papers are cited together by other papers.",
    ]
    noi_node_edge_text = [
        "prompt node. link prediction on the papers that are cited together",
    ]
    noi_node_text = [
        "prompt node. node classification on the paper's category",
    ]
    prompt_edge_text = [
        "prompt edge",
        "prompt edge. edge for query graph that is our target",
        "prompt edge. edge for support graph that is an example",
    ]
    return (
        [new_data],
        [
            clean_text,
            edge_text,
            noi_node_text + noi_node_edge_text,
            label_text + edge_label_text + logic_label_text,
            prompt_edge_text,
        ],
        {
            "e2e_node": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text), dtype=torch.long)],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
            },
            "e2e_link": {
                "noi_node_text_feat": ["noi_node_text_feat", [1]],
                "class_node_text_feat": [
                    "class_node_text_feat",
                    torch.arange(len(label_text), len(label_text) + len(edge_label_text), dtype=torch.long),
                ],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
            },
            "lr_node": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": ["class_node_text_feat", torch.arange(len(label_text), dtype=torch.long)],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0, 1, 2]],
            },
            "logic_e2e": {
                "noi_node_text_feat": ["noi_node_text_feat", [0]],
                "class_node_text_feat": [
                    "class_node_text_feat",
                    torch.arange(
                        len(label_text) + len(edge_label_text),
                        len(label_text) + len(edge_label_text) + len(logic_label_text),
                        dtype=torch.long,
                    ),
                ],
                "prompt_edge_text_feat": ["prompt_edge_text_feat", [0]],
            },
        },
    )
