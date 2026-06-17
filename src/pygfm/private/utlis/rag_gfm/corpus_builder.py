"""
RAG-GFM corpus build: extract node/class text from graph datasets, embed, write nano-vectordb.

Requires: nano-vectordb, sentence-transformers (or transformers).
Optional install: pip install -e ".[rag_gfm]"
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading (GFM-Toolbox layout + single-file .pt)
# ---------------------------------------------------------------------------

def _safe_torch_load(path: str):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)


def _parse_pt_to_data(obj) -> Data:
    if isinstance(obj, Data):
        return obj
    if isinstance(obj, tuple) and len(obj) >= 1:
        d = obj[0]
        if isinstance(d, Data):
            return d
        if isinstance(d, dict) and "x" in d and "edge_index" in d:
            return Data(**{k: v for k, v in d.items() if k in ("x", "edge_index", "y", "raw_texts", "label_name", "label_names", "label_text")})
    raise TypeError(f"Cannot parse .pt: expected Data or (dict, slices), got {type(obj)}")


def resolve_rag_gfm_graph_pt_path(data_root: str, dataset_name: str) -> Optional[str]:
    """
    Resolve a graph ``.pt`` under ``data_root`` for RAG-GFM loaders.

    Tries, in order:

    1. ``{data_root}/{Dataset}/processed/data.pt`` (exact name, lower, capitalized, upper)
    2. Same path with a **subdir** whose name matches ``dataset_name`` case-insensitively
    3. Flat ``{data_root}/*.pt`` whose stem matches ``dataset_name`` case-insensitively
       (Linux often has ``Cora.pt`` while callers pass ``Cora`` — older code only looked for ``cora.pt``).
    """
    root = os.path.abspath(os.path.expanduser(data_root))
    if not os.path.isdir(root):
        return None
    raw = (dataset_name or "").strip()
    if not raw:
        return None
    target_cf = raw.casefold()

    def p_join(*parts: str) -> str:
        return os.path.join(root, *parts)

    for sub in (raw, raw.lower(), raw.capitalize(), raw.upper()):
        candidate = p_join(sub, "processed", "data.pt")
        if os.path.isfile(candidate):
            return candidate
    try:
        for entry in os.listdir(root):
            subdir = p_join(entry)
            if not os.path.isdir(subdir):
                continue
            if entry.casefold() != target_cf:
                continue
            candidate = os.path.join(subdir, "processed", "data.pt")
            if os.path.isfile(candidate):
                return candidate
    except OSError:
        pass

    for stem in (raw, raw.lower(), raw.capitalize(), raw.upper()):
        candidate = os.path.join(root, f"{stem}.pt")
        if os.path.isfile(candidate):
            return candidate
    try:
        for fn in os.listdir(root):
            if not fn.lower().endswith(".pt"):
                continue
            stem, _ = os.path.splitext(fn)
            if stem.casefold() == target_cf:
                return os.path.join(root, fn)
    except OSError:
        pass
    return None


def load_node_data_for_rag(data_root: str, dataset_name: str) -> Optional[Data]:
    """
    Load one graph dataset for corpus build.
    Prefer data_root/{dataset_name}/processed/data.pt, else data_root/{name}.pt.
    Data must have raw_texts; label_name/label_text default from y if missing.
    """
    pt_path = resolve_rag_gfm_graph_pt_path(data_root, dataset_name)
    if pt_path is None:
        return None
    data = _parse_pt_to_data(_safe_torch_load(pt_path))

    if not hasattr(data, "raw_texts") or data.raw_texts is None or len(data.raw_texts) == 0:
        raise ValueError(
            f"Dataset {dataset_name} has no raw_texts; RAG-GFM corpus needs node text. "
            f"Provide a .pt with raw_texts or processed/data.pt."
        )
    if not hasattr(data, "y") or data.y is None:
        data.y = torch.zeros(data.num_nodes, dtype=torch.long)
    num_classes = int(data.y.max().item()) + 1
    if not hasattr(data, "label_name") and not hasattr(data, "label_names"):
        data.label_name = [f"Class_{i}" for i in range(num_classes)]
    elif hasattr(data, "label_names") and not hasattr(data, "label_name"):
        data.label_name = data.label_names
    if not hasattr(data, "label_text"):
        data.label_text = getattr(data, "label_name", [f"Class_{i}" for i in range(num_classes)])
    return data


# ---------------------------------------------------------------------------
# Text encoders (thin wrapper; default sentence-transformers)
# ---------------------------------------------------------------------------

def _get_text_encoder(encoder_name: str, device: torch.device):
    """Returns (encode_fn, embedding_dim). encode_fn(texts: List[str]) -> numpy [N, D]."""
    ename = (encoder_name or "SentenceBert").strip()
    if ename in ("SentenceBert", "sentence-transformers/multi-qa-distilbert-cos-v1"):
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device=str(device))
            dim = model.get_sentence_embedding_dimension()
            def encode(texts):
                return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return encode, dim
        except ImportError:
            raise ImportError("RAG-GFM corpus defaults to sentence-transformers: pip install sentence-transformers")
    if ename in ("bert", "Bert"):
        try:
            from transformers import AutoTokenizer, AutoModel
            model_path = "bert-base-uncased"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path).to(device)
            model.eval()
            def encode(texts):
                out = []
                bs = 32
                for i in range(0, len(texts), bs):
                    batch = texts[i : i + bs]
                    inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True).to(device)
                    with torch.no_grad():
                        h = model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()
                    out.append(h)
                return np.concatenate(out, axis=0)
            dim = 768
            return encode, dim
        except ImportError:
            raise ImportError("BERT encoder requires: pip install transformers")
    raise ValueError(f"Unsupported text_encoder: {encoder_name}; use SentenceBert or bert")


# ---------------------------------------------------------------------------
# Corpus config and main build
# ---------------------------------------------------------------------------

@dataclass
class RAGCorpusBuilderConfig:
    """RAG-GFM corpus build config"""

    data_root: str = "datasets/rag_gfm"
    """Graph data root (same layout as GFM-Toolbox baselines)."""
    dataset_names: List[str] = field(default_factory=lambda: ["Cora", "Citeseer", "Pubmed", "Photo", "Computers"])
    """Dataset names to include in the corpus."""
    corpus_output_path: str = "downstream_data/rag_gfm/corpus/unified_database.json"
    """nano-vectordb output path"""
    text_encoder: str = "SentenceBert"
    """Text encoder: SentenceBert | bert"""
    device: str = "cuda"
    """Device string."""


def build_rag_corpus(config: RAGCorpusBuilderConfig) -> str:
    """
    Build RAG-GFM corpus: for each config.dataset_names, load node/class text,
    embed, write nano-vectordb to config.corpus_output_path.
    Returns absolute path written.
    """
    try:
        from nano_vectordb import NanoVectorDB
    except ImportError:
        raise ImportError("RAG-GFM corpus needs nano-vectordb: pip install nano-vectordb")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    encode_fn, embedding_dim = _get_text_encoder(config.text_encoder, device)

    documents = []
    doc_id = 0

    for dataset_name in config.dataset_names:
        try:
            data = load_node_data_for_rag(config.data_root, dataset_name)
        except Exception as e:
            import warnings
            warnings.warn(f"Skipping dataset {dataset_name}: {e}", UserWarning)
            continue
        if data is None:
            import warnings
            warnings.warn(f"Dataset not found: {dataset_name} (check {config.data_root})", UserWarning)
            continue

        raw_texts = data.raw_texts
        y = data.y
        label_text = getattr(data, "label_text", [])
        label_name = getattr(data, "label_name", [f"Class_{i}" for i in range(int(y.max().item()) + 1)])
        if len(label_text) < len(label_name):
            label_text = list(label_name)
        node_labels = y.numpy() if hasattr(y, "numpy") else y.cpu().numpy()
        class_id_to_desc = {i: label_text[i] if i < len(label_text) else f"Class_{i}" for i in range(len(label_name))}
        class_id_to_name = {i: label_name[i] if i < len(label_name) else f"Class_{i}" for i in range(len(label_name))}

        current_texts = []
        current_meta = []
        for i, text in enumerate(raw_texts):
            cl = int(node_labels[i]) if i < len(node_labels) else 0
            cname = class_id_to_name.get(cl, f"Class_{cl}")
            cdesc = class_id_to_desc.get(cl, "unknown_class")
            structured = f"Dataset: {dataset_name}, Node ID: {i}, Class Label: {cl}, Class Description: {cdesc}, Node Text: {text}"
            current_texts.append(structured)
            current_meta.append({
                "type": "node",
                "id_in_dataset": i,
                "dataset": dataset_name,
                "class_label": int(cl),
                "class_name": cname,
                "original_text": text,
            })
        for i, text in enumerate(label_text):
            current_texts.append(text)
            current_meta.append({
                "type": "class",
                "class_name": label_name[i] if i < len(label_name) else f"Class_{i}",
                "dataset": dataset_name,
            })

        for text, meta in tqdm(
            list(zip(current_texts, current_meta)),
            desc=f"Encode {dataset_name}",
            leave=False,
        ):
            emb = encode_fn([text])[0]
            vec = np.asarray(emb, dtype=np.float32).reshape(-1)
            documents.append({
                "__id__": str(doc_id),
                "__vector__": vec,
                "text": text,
                "metadata": meta,
            })
            doc_id += 1

    if not documents:
        raise RuntimeError("No datasets loaded; corpus is empty. Check data_root and dataset_names.")

    out_path = os.path.abspath(config.corpus_output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    db = NanoVectorDB(embedding_dim, storage_file=out_path)
    db.upsert(documents)
    db.save()
    return out_path
