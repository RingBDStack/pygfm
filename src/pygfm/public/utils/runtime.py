from __future__ import annotations

import torch
import random
import numpy as np
import os
from pathlib import Path
import scipy.sparse as sp
import scipy.sparse.linalg as sp_linalg
from typing import Any, Tuple
from torch_geometric.data import Data
from torch_geometric.datasets import Amazon, Planetoid, TUDataset
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_edge

def save_subdir_for_ckpt(
    datasets: str | None,
    *,
    target: str | None = None,
    default: str = "default",
) -> str:
    """
    Folder name under ckpts/<baseline>/ from --datasets (sanitized, lowercased),
    else target.lower(), else default.
    """
    if datasets is not None and str(datasets).strip():
        s = (
            str(datasets)
            .strip()
            .replace(os.sep, "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        return s.lower() if s else default
    if target is not None and str(target).strip():
        return str(target).strip().lower()
    return default


def resolve_preprompt_ckpt(
    baseline: str,
    datasets_subdir: str,
    finetune_dataset: str | None = None,
) -> str:
    """
    Resolve ckpt path: prefer ckpts/<baseline>/<subdir>/preprompt_<dataset>.pth,
    then preprompt.pth. Paths relative to cwd unless absolute.
    """
    from pathlib import Path

    sub = save_subdir_for_ckpt(datasets_subdir, target=None, default="default")
    d = Path.cwd() / "ckpts" / baseline / sub
    if finetune_dataset and str(finetune_dataset).strip():
        p1 = d / f"preprompt_{str(finetune_dataset).strip().lower()}.pth"
        if p1.is_file():
            return str(p1)
    p2 = d / "preprompt.pth"
    if p2.is_file():
        return str(p2)
    if finetune_dataset and str(finetune_dataset).strip():
        return str(d / f"preprompt_{str(finetune_dataset).strip().lower()}.pth")
    return str(p2)


def set_seed(seed=42):
    """Strictly maintain seed configuration."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def fast_aug(x, edge_index, drop_feat=0.2, drop_edge=0.2):
    """Original feature masking and edge dropping logic."""
    x_aug = x.clone()
    mask = torch.rand(x.size(1), device=x.device) < drop_feat
    x_aug[:, mask] = 0
    edge_index_aug, _ = dropout_edge(edge_index, p=drop_edge, training=True)
    return x_aug, edge_index_aug

def _safe_torch_load_pt(path: str) -> Any:
    """Load ``.pt`` with PyTorch 2.4+ ``weights_only=False`` when available."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _find_data_pt_in_subdir(data_path: str) -> str | None:
    """Prefer ``<subdir>/data.pt``, then PyG-style ``<subdir>/processed/data.pt``."""
    candidates = (
        os.path.join(data_path, "data.pt"),
        os.path.join(data_path, "processed", "data.pt"),
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _torch_load_to_single_data(raw: Any) -> Data:
    """
    Normalize ``torch.load`` output to a single :class:`Data` (first graph if batched).
    Supports: ``Data``; ``(Data, None)``; ``(Data, slice_dict)``; PyG 3-tuple ``(dict, slices, cls)``.
    """
    if isinstance(raw, Data):
        return raw

    if isinstance(raw, tuple):
        if len(raw) == 3:
            d_obj, slices, data_cls = raw
            if isinstance(d_obj, dict):
                data = data_cls.from_dict(d_obj)
            else:
                data = d_obj
            if slices is None:
                return data
            from torch_geometric.data.separate import separate

            return separate(
                cls=data.__class__,
                batch=data,
                idx=0,
                slice_dict=slices,
                inc_dict=None,
                decrement=False,
            )

        if len(raw) == 2:
            d, slices = raw
            if isinstance(d, Data):
                if slices is None:
                    return d
                from torch_geometric.data.separate import separate

                return separate(
                    cls=Data,
                    batch=d,
                    idx=0,
                    slice_dict=slices,
                    inc_dict=None,
                    decrement=False,
                )
            if isinstance(d, dict) and "x" in d and "edge_index" in d:
                data = Data.from_dict(d)
                if slices is None:
                    return data
                from torch_geometric.data.separate import separate

                return separate(
                    cls=Data,
                    batch=data,
                    idx=0,
                    slice_dict=slices,
                    inc_dict=None,
                    decrement=False,
                )

    raise TypeError(
        f"Unsupported .pt content for single-graph load: {type(raw)}; "
        "expected Data or PyG InMemoryDataset tuple."
    )


def _ensure_node_labels(data: Data) -> Data:
    """Pretrain paths expect ``y``; fill zeros if missing."""
    if getattr(data, "y", None) is None:
        n = int(data.num_nodes) if data.num_nodes is not None else data.x.size(0)
        data.y = torch.zeros(n, dtype=torch.long)
    return data


def _align_edge_type_with_edges(data: Data) -> Data:
    """
    ``ToUndirected`` / ``AddSelfLoops`` may add edges without updating ``edge_type``,
    so ``edge_type.numel() != edge_index.size(1)`` (e.g. self-loops missing types).
    Truncate or pad zeros so each edge has a matching type entry.
    """
    et = getattr(data, "edge_type", None)
    if et is None:
        return data
    et = et.view(-1)
    n_e = data.edge_index.size(1)
    if et.numel() == n_e:
        data.edge_type = et
        return data
    if et.numel() < n_e:
        pad = torch.zeros(n_e - et.numel(), dtype=et.dtype, device=et.device)
        data.edge_type = torch.cat([et, pad], dim=0)
        return data
    data.edge_type = et[:n_e]
    return data


def get_few_shot_split(labels, n_shot=5):
    """Original stratified few-shot split logic."""
    train_indices, test_indices = [], []
    labels_np = labels.cpu().numpy()
    num_classes = int(labels_np.max()) + 1
    for c in range(num_classes):
        idx = np.where(labels_np == c)[0]
        np.random.shuffle(idx)
        if len(idx) >= n_shot:
            train_indices.extend(idx[:n_shot])
            test_indices.extend(idx[n_shot:])
        else:
            train_indices.extend(idx)
    return torch.tensor(train_indices, dtype=torch.long), torch.tensor(test_indices, dtype=torch.long)

def _match_dataset_mapping_key(folder_name: str, dataset_mapping: dict) -> str | None:
    """Case-insensitive match for ``dataset_mapping`` keys (e.g. cora vs Cora on Linux)."""
    if folder_name in dataset_mapping:
        return folder_name
    lower = folder_name.casefold()
    for k in dataset_mapping:
        if k.casefold() == lower:
            return k
    return None


def load_all_datasets(data_root: str, *, allow_pyg_download: bool = False) -> list:
    """
    Scan ``data_root`` for graphs; returns ``list[dict]`` compatible with ``{d['name']: d for d in all_raw}``.

    **Flat ``*.pt``**: single-file graphs like ``data_root/Cora.pt`` loaded to one ``Data``;
    names are matched **case-insensitively** to ``dataset_mapping`` (e.g. ``cora.pt`` -> ``Cora``).

    **Subdirs**: ``<name>/data.pt`` or ``<name>/processed/data.pt``.

    **Else**:
    - If ``allow_pyg_download=False`` (default), do **not** fall back to PyG datasets
      (Planetoid/Amazon/TU). This avoids implicit downloads and enforces a uniform local
      ``*.pt``/``data.pt`` format for both pretrain and finetune.
    - If ``allow_pyg_download=True``, fall back to ``dataset_mapping`` (Planetoid / Amazon / TU / HGPROMPT).
    Use ``--datasets`` / ``--target`` names consistent with ``d['name']`` (e.g. ``Cora``).
    """
    datasets = []
    # Undirected + self-loops for robustness
    transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

    # Optional PyG loader mapping (used only when allow_pyg_download=True)
    dataset_mapping = {
        "ACM": "HETEROGENEOUS",
        "DBLP": "HETEROGENEOUS",
        "Freebase": "HETEROGENEOUS",
        "ENZYMES": TUDataset,
        "PROTEINS": TUDataset,
        "BZR": TUDataset,
        "COX2": TUDataset,
        "Cora": Planetoid,
        "Citeseer": Planetoid,
        "Pubmed": Planetoid,
        "Photo": Amazon,
        "Computers": Amazon,
    }

    data_root = os.path.abspath(data_root)
    if not os.path.exists(data_root):
        print(f"!! Error: data root does not exist: {data_root}")
        return []

    # Canonical names already loaded (avoid Cora.pt + Cora/ duplicate)
    loaded_canonical: set[str] = set()

    # --- 0) Flat *.pt under data_root (e.g. exported Cora.pt) ---
    try:
        root_entries = os.listdir(data_root)
    except OSError as e:
        print(f"!! Error: cannot list {data_root}: {e}")
        return []

    for entry in sorted(root_entries):
        if entry.startswith((".", "_")):
            continue
        if Path(entry).suffix.lower() != ".pt":
            continue
        pt_path = os.path.join(data_root, entry)
        if not os.path.isfile(pt_path):
            continue
        stem = Path(entry).stem
        if not stem:
            continue
        map_key = _match_dataset_mapping_key(stem, dataset_mapping)
        canonical_name = map_key if map_key is not None else stem
        if canonical_name in loaded_canonical:
            continue
        try:
            print(f">> Loading flat .pt: {canonical_name} ({pt_path})")
            raw = _safe_torch_load_pt(pt_path)
            data = _align_edge_type_with_edges(
                transform(_ensure_node_labels(_torch_load_to_single_data(raw)))
            )
            datasets.append({"name": canonical_name, "ds": [data]})
            loaded_canonical.add(canonical_name)
            nn = data.num_nodes if data.num_nodes is not None else data.x.size(0)
            print(f"Successfully initialized: {canonical_name} (Nodes: {nn})")
        except Exception as e:
            print(f"!! Failed to load flat file {entry}: {e}")

    found_dirs = [
        d
        for d in root_entries
        if os.path.isdir(os.path.join(data_root, d)) and not d.startswith((".", "_"))
    ]

    for name in found_dirs:
        data_path = os.path.join(data_root, name)
        map_key = _match_dataset_mapping_key(name, dataset_mapping)
        canonical_name = map_key if map_key is not None else name
        if canonical_name in loaded_canonical:
            continue

        pt_path = _find_data_pt_in_subdir(data_path)

        if pt_path is not None:
            try:
                print(f">> Loading data.pt: {canonical_name} ({pt_path})")
                raw = _safe_torch_load_pt(pt_path)
                data = _align_edge_type_with_edges(
                    transform(_ensure_node_labels(_torch_load_to_single_data(raw)))
                )
                datasets.append({"name": canonical_name, "ds": [data]})
                loaded_canonical.add(canonical_name)
                nn = data.num_nodes if data.num_nodes is not None else data.x.size(0)
                print(f"Successfully initialized: {canonical_name} (Nodes: {nn})")
            except Exception as e:
                print(f"!! Failed to load {name} from {pt_path}: {e}")
            continue

        if map_key is None:
            print(
                f"!! Skip: subfolder {name!r} has no data.pt and is not in dataset_mapping."
                f" Add data.pt / processed/data.pt or register a loader."
            )
            continue

        dtype = dataset_mapping[map_key]

        try:
            if dtype == "HETEROGENEOUS":
                if os.path.exists(os.path.join(data_path, "node.dat")):
                    print(f">> Loading HGPROMPT layout: {canonical_name}")

                    features = []
                    max_dim = 0
                    with open(os.path.join(data_path, "node.dat"), "r") as f:
                        for line in f:
                            parts = line.strip().split("\t")
                            if len(parts) > 3:
                                feat = [float(x) for x in parts[3].split(",")]
                                if max_dim == 0:
                                    max_dim = len(feat)

                                if len(feat) == max_dim:
                                    features.append(feat)
                                else:
                                    feat = feat[:max_dim] + [0.0] * (max_dim - len(feat))
                                    features.append(feat)

                    x = torch.tensor(features, dtype=torch.float)
                    num_nodes = x.size(0)

                    edge_index, edge_type = [], []
                    if os.path.exists(os.path.join(data_path, "link.dat")):
                        with open(os.path.join(data_path, "link.dat"), "r") as f:
                            for line in f:
                                parts = line.strip().split("\t")
                                if len(parts) < 3:
                                    continue
                                u, v, t = int(parts[0]), int(parts[1]), int(parts[2])
                                if u < num_nodes and v < num_nodes:
                                    edge_index.append([u, v])
                                    edge_type.append(t)

                    if len(edge_index) > 0:
                        ei = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                        et = torch.tensor(edge_type, dtype=torch.long)
                        current_data = Data(x=x, edge_index=ei)
                        current_data.edge_type = et
                        datasets.append({"name": canonical_name, "ds": [current_data]})
                        loaded_canonical.add(canonical_name)
                        print(
                            f"Successfully initialized: {canonical_name} (Nodes: {num_nodes}, Edges: {ei.size(1)})"
                        )
                    else:
                        print(f"!! Warning: {canonical_name} has no valid edges after filter; skipped.")

            elif dtype in (Planetoid, Amazon, TUDataset):
                if not allow_pyg_download:
                    print(
                        f"!! Skip PyG fallback for {canonical_name} (dir: {name}). "
                        f"Please provide a local export: "
                        f"{os.path.join(data_root, canonical_name + '.pt')} "
                        f"or {os.path.join(data_path, 'data.pt')}."
                    )
                    continue

                if dtype == Planetoid:
                    # Use on-disk folder ``name`` (Cora vs cora path case on Linux)
                    print(f">> Loading Planetoid: {canonical_name} (dir: {name})")
                    ds = Planetoid(root=data_root, name=name, transform=transform)
                    datasets.append({"name": canonical_name, "ds": ds})
                    loaded_canonical.add(canonical_name)
                    print(f"Successfully initialized: {canonical_name} (Nodes: {ds[0].num_nodes})")
                elif dtype == Amazon:
                    print(f">> Loading Amazon: {canonical_name} (dir: {name})")
                    ds = Amazon(root=data_root, name=name, transform=transform)
                    datasets.append({"name": canonical_name, "ds": ds})
                    loaded_canonical.add(canonical_name)
                    print(f"Successfully initialized: {canonical_name} (Nodes: {ds[0].num_nodes})")
                else:
                    print(f">> Loading TUDataset: {canonical_name} (dir: {name})")
                    ds = TUDataset(root=data_root, name=name, transform=transform)
                    datasets.append({"name": canonical_name, "ds": ds})
                    loaded_canonical.add(canonical_name)
                    print(f"Successfully initialized: {canonical_name} (Graphs: {len(ds)})")

        except Exception as e:
            print(f"!! Failed to load {canonical_name} ({name}) due to error: {e}")

    if not datasets:
        try:
            subs = sorted(os.listdir(data_root))
        except OSError:
            subs = []
        print(
            f"!! Warning: no datasets loaded. data_root={data_root} | entries: {subs}."
            f" Provide flat Cora.pt; or Cora/data.pt, Cora/processed/data.pt."
            + ("" if not allow_pyg_download else " Or enable PyG fallback with allow_pyg_download=True.")
        )

    return datasets


def load_single_graph_dataset(
    data_root: str, dataset: str, *, allow_pyg_download: bool = False
) -> tuple[Data, int]:
    """
    Load **one** node-level graph under a baseline data root (same rules as :func:`load_all_datasets`):

    - Flat ``datasets/<baseline>/Cora.pt``, ``Pubmed.pt``, ...;
    - Or ``Cora/data.pt``, ``Cora/processed/data.pt``;
    - Or Planetoid / Amazon / TU / HGPROMPT layouts from the mapping table.

    :param data_root: Baseline root, e.g. ``datasets/mdgpt``, ``datasets/graver``.
    :param dataset: Name matching ``d['name']`` from ``load_all_datasets`` (case-insensitive, e.g. ``Cora``).
    :return: ``(data, num_classes)`` with single :class:`~torch_geometric.data.Data`.
    :raises ValueError: Not found, or multi-graph TU entry (use :func:`load_multi_graph_pyg_dataset` for graph classification).
    """
    items = load_all_datasets(data_root, allow_pyg_download=allow_pyg_download)
    key = dataset.strip()
    found_names = [d["name"] for d in items]
    for d in items:
        if d["name"].casefold() != key.casefold():
            continue
        raw_ds = d["ds"]
        if isinstance(raw_ds, list):
            data = raw_ds[0]
            nc = _infer_num_classes_from_data(data)
            return data, nc
        n = len(raw_ds)
        if n > 1:
            raise ValueError(
                f"Dataset {dataset!r} under {data_root!r} is multi-graph ({n} graphs)."
                f"Use a single-graph export (e.g. Cora.pt) for node tasks; use load_multi_graph_pyg_dataset for graph classification."
            )
        data = raw_ds[0]
        nc = getattr(raw_ds, "num_classes", None)
        if nc is not None:
            return data, int(nc)
        return data, _infer_num_classes_from_data(data)

    raise ValueError(
        f"Dataset {dataset!r} not found under {data_root!r}."
        f" Loaded names: {found_names}."
        f" Place Cora.pt etc. here, or sync from the mdgpt layout."
    )


def _infer_num_classes_from_data(data: Data) -> int:
    y = getattr(data, "y", None)
    if y is None:
        return 1
    valid = y[y >= 0]
    if valid.numel() == 0:
        return 1
    return int(valid.max().item()) + 1


def load_multi_graph_pyg_dataset(data_root: str, dataset: str):
    """
    Load **multi-graph** classification data (e.g. TU) using the same discovery rules as :func:`load_all_datasets`.

    :return: ``(dataset_obj, num_classes)`` with PyG ``InMemoryDataset`` (e.g. :class:`~torch_geometric.datasets.TUDataset`).
    """
    items = load_all_datasets(data_root)
    key = dataset.strip()
    found_names = [d["name"] for d in items]
    for d in items:
        if d["name"].casefold() != key.casefold():
            continue
        raw_ds = d["ds"]
        if isinstance(raw_ds, list):
            raise ValueError(
                f"Dataset {dataset!r} is a single-graph .pt export; not for multi-graph classification. Use TU layout or a subfolder."
            )
        if len(raw_ds) <= 1:
            raise ValueError(
                f"Dataset {dataset!r} has only {len(raw_ds)} graph(s); not a multi-graph classification set."
            )
        nc = int(raw_ds.num_classes)
        return raw_ds, nc

    raise ValueError(
        f"Multi-graph dataset {dataset!r} not found under {data_root!r}. Loaded: {found_names}."
    )


def load_single_graph_dataset_or_reddit(data_root: str, dataset: str) -> tuple[Data, int]:
    """
    Same as :func:`load_single_graph_dataset`.

    Note:
    This helper used to fall back to PyG ``Reddit(root=data_root)`` if no local ``.pt`` export exists.
    For a uniform, offline-friendly workflow, require local exports here as well.
    """
    return load_single_graph_dataset(data_root, dataset, allow_pyg_download=False)


def early_stopping(
    loss: float,
    best: float,
    cnt_wait: int,
    patience: int,
) -> Tuple[bool, float, int]:
    """
    Early stopping helper. Returns (should_stop, new_best, new_cnt_wait).
    """
    if loss < best:
        return False, loss, 0
    cnt_wait += 1
    return cnt_wait >= patience, best, cnt_wait


def compute_prototypes(embeddings: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class mean embeddings (prototypes). [N,D] + [N] -> [num_classes, D]."""
    device = embeddings.device
    prototypes = torch.zeros(num_classes, embeddings.size(1), device=device, dtype=embeddings.dtype)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = embeddings[mask].mean(dim=0)
    return prototypes


def compute_spectral_components(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device | None = None,
    k: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Top-k eigenpairs of the symmetric normalized Laplacian (smallest eigenvalues).
    Used by BRIDGE downstream spectral regularization.
    """
    dev = device or edge_index.device
    edge_index_np = edge_index.detach().cpu().numpy()
    adj = sp.coo_matrix(
        (np.ones(edge_index_np.shape[1]), (edge_index_np[0], edge_index_np[1])),
        shape=(num_nodes, num_nodes),
    )
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(num_nodes)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    laplacian = sp.eye(num_nodes) - normalized_adj
    k_eff = min(k, num_nodes - 1, max(1, num_nodes - 2))
    if k_eff < 1:
        k_eff = 1
    evals, evecs = sp_linalg.eigsh(laplacian.astype(np.float64), k=k_eff, which="SM")
    return torch.tensor(evecs, dtype=torch.float32, device=dev), torch.tensor(
        evals, dtype=torch.float32, device=dev
    )


def bridge_preprompt_negative_tuples(adj, n_neg: int = 50) -> np.ndarray:
    """
    BRIDGE PrePrompt: one positive (1-hop neighbor) + n_neg non-neighbor negatives per row.
    adj may be coo / csr; returns [N, 1+n_neg] int64.
    """
    if not isinstance(adj, sp.csr_matrix):
        adj = adj.tocsr()
    nodenum = adj.shape[0]
    indices = adj.indices
    indptr = adj.indptr
    res = np.zeros((nodenum, 1 + n_neg), dtype=np.int64)
    whole = np.arange(nodenum)
    for i in range(nodenum):
        nz = indices[indptr[i] : indptr[i + 1]]
        zero_idx = np.setdiff1d(whole, nz, assume_unique=False)
        np.random.shuffle(nz)
        np.random.shuffle(zero_idx)
        if nz.size == 0:
            res[i, 0] = i
        else:
            res[i, 0] = nz[0]
        take = min(n_neg, len(zero_idx))
        res[i, 1 : 1 + take] = zero_idx[:take]
        if take < n_neg:
            res[i, 1 + take :] = np.random.randint(0, nodenum, size=n_neg - take)
    return res