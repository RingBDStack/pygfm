# RAG-GFM

RAG-GFM adds **retrieval-augmented** training/finetuning and optional **motif** structure retrieval. Corpus and motif libraries are baseline-specific preparation steps.

## Install

```bash
cd /path/to/repo
pip install -e .
# Extra retrieval deps (example): pip install nano-vectordb sentence-transformers
```

Run scripts from the **repository root** (or set `PYTHONPATH` to the repo). There is no separate `gfm-rag-gfm-*` console entry yet; use `python scripts/rag_gfm/...`.

---

## Step 1: Build the corpus (RAG text retrieval)

Before pretrain or finetune, build the nano-vectordb corpus used by RAG.

### Data layout

- Put graph data with **raw_texts** (node text) under `data_root`.
- Two layouts are supported:
  - `data_root/{Dataset}/processed/data.pt`
  - `data_root/{name}.pt`
- `Data` must expose `raw_texts`; if `label_name` / `label_text` are missing they are inferred from `y`.

### Run

```bash
python scripts/rag_gfm/build_corpus.py

python scripts/rag_gfm/build_corpus.py \
  --data_root datasets/rag_gfm \
  --corpus_output downstream_data/rag_gfm/corpus/unified_database.json \
  --datasets Cora,Citeseer,Pubmed
```

Default output: `downstream_data/rag_gfm/corpus/unified_database.json`.

### Python API

```python
from pygfm.private.utlis.rag_gfm import RAGCorpusBuilderConfig, build_rag_corpus

config = RAGCorpusBuilderConfig(
    data_root="datasets/rag_gfm",
    dataset_names=["Cora", "Citeseer", "Pubmed"],
    corpus_output_path="downstream_data/rag_gfm/corpus/unified_database.json",
    text_encoder="SentenceBert",
)
out_path = build_rag_corpus(config)
```

---

## Step 2 (optional): Build the motif library

Train a subgraph encoder per dataset and build motif nano-vectordbs for **motif retrieval** in downstream training. If you skip this, finetuning uses RAG text augmentation only.

### Data layout

- Graphs under `data_root` as above, with **x** and **edge_index**.
- Paths: `data_root/{Dataset}/processed/data.pt` or `data_root/{name}.pt`.

### Run

```bash
python scripts/rag_gfm/build_motif_lib.py

python scripts/rag_gfm/build_motif_lib.py \
  --data_root datasets/rag_gfm \
  --motif_lib_path downstream_data/rag_gfm/motif_lib \
  --datasets Cora,Citeseer,Pubmed \
  --epochs 200
```

Each dataset writes under `motif_lib_path/{dataset}/`:

- `encoder.pth` — subgraph encoder weights  
- `config.pth` — encoder config (`struct_input_dim`, `hidden_dim`, `output_dim`)  
- `motif_vectordb.json` — subgraph vector store  

### Python API

```python
from pygfm.data import MotifLibBuilderConfig, build_motif_lib

config = MotifLibBuilderConfig(
    data_root="datasets/rag_gfm",
    dataset_names=["Cora", "Citeseer", "Pubmed"],
    motif_lib_path="downstream_data/rag_gfm/motif_lib",
    epochs=200,
)
paths = build_motif_lib(config)
```

---

---

## Step 3: Pretraining (RAG-GFM script)

Pretraining is **per baseline**: MDGPT uses `scripts/mdgpt/pretrain.py`, RAG-GFM uses `scripts/rag_gfm/pretrain.py`. Both reuse shared encoders inside the toolbox (e.g. GCN, NodeLevelPrompt, NodeNodeContrastiveLoss), but models and driver scripts stay separate by baseline name.

### Run

```bash
python scripts/rag_gfm/pretrain.py --target_dataset Cora
python scripts/rag_gfm/pretrain.py --datasets Cora,Citeseer,Pubmed --save_dir ckpts/rag_gfm
```

Default save: `ckpts/rag_gfm/preprompt.pth`; with `--target_dataset XXX`, `ckpts/rag_gfm/xxx/preprompt_xxx.pth`. Optionally save `aligners.pkl` for downstream feature alignment.

### Notes

- Data: loaded from `--data_root` (default `datasets/rag_gfm`) using the same paths as corpus/motif (`processed/data.pt` or `{name}.pt`), with `x` and `edge_index`.
- Model: `pygfm.baseline_models.rag_gfm.PrePromptModel` (per-domain NodeLevelPrompt + shared GCN + NodeNodeContrastiveLoss); GCN / NodeLevelPrompt / loss come from `pygfm.private.utlis` / `pygfm.public.utils`.

---

## Step 4: Node classification finetuning

Few-shot node classification with RAG-GFM checkpoints and the **shared** DownPrompt (`pygfm.baseline_models.mdgpt.DownPromptModel`).

### Data

- Generate few-shot splits to `downstream_data/rag_gfm/{dataset}/{k}shot/splits.pt` (same format as MDGPT). Example:

  ```bash
  python scripts/mdgpt/generate_downstream.py few_shot --dataset Cora --k_shot 1 \
    --downstream_root downstream_data/rag_gfm --data_root datasets/rag_gfm
  ```

### Run

```bash
python scripts/rag_gfm/finetune.py --dataset Cora --k_shot 1 --ckpt ckpts/rag_gfm/cora/preprompt_cora.pth
python scripts/rag_gfm/finetune.py --dataset Cora --k_shot 5 --ckpt ckpts/rag_gfm/preprompt.pth --task_num 10
```

---

## Step 5: Graph classification finetuning

Few-shot graph classification with RAG-GFM checkpoints and the **shared** DownPromptGraph (`pygfm.baseline_models.mdgpt.DownPromptGraphModel`) plus `build_test_subgraphs`.

### Data

- Generate graph-level few-shot splits to `downstream_data/rag_gfm/{dataset}/{k}shot_graph_batch/splits.pt`:

  ```bash
  python scripts/mdgpt/generate_downstream.py graph_batch --dataset Cora --k_shot 1 \
    --downstream_root downstream_data/rag_gfm --data_root datasets/rag_gfm
  ```

### Run

```bash
python scripts/rag_gfm/finetune_graph.py --dataset Cora --k_shot 1 --ckpt ckpts/rag_gfm/cora/preprompt_cora.pth
python scripts/rag_gfm/finetune_graph.py --dataset Cora --k_shot 5 --ckpt ckpts/rag_gfm/preprompt.pth --task_num 10
```

---

## Summary

- **Pretrain**: RAG-GFM only (`models.rag_gfm.PrePromptModel` + `scripts/rag_gfm/pretrain.py`).
- **Finetune**: node and graph tasks use the **shared** DownPrompt / DownPromptGraph (`models.mdgpt`), loading RAG-GFM checkpoints and paths (`downstream_data/rag_gfm`, `datasets/rag_gfm`). You can add RAG/motif feature augmentation on top of this flow later.
