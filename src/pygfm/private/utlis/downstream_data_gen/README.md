# Downstream data generation (`pygfm.private.utlis.downstream_data_gen`)

Pre-generates few-shot splits and subgraph batches for downstream scripts such as MDGPT.

> **Relationship to repo layout**: the generator can still write splits for **any supported dataset**; **split data shipped under `downstream_data/`** is maintained for **Cora only** (see repo root `downstream_data/README.md`). For other datasets, generate locally or point to your own paths.

## Where files are written (important)

- **Upstream / read**: load PyG data or `processed/data.pt` from `data_root` (default `datasets/mdgpt`).
- **Downstream / write**: few-shot and subgraph splits are written **only under `downstream_root`** (default **`downstream_data/mdgpt`**), **not** under `few_shot` inside `data_root`.

Example layout (relative to repo root):

```text
downstream_data/mdgpt/Cora/1shot/splits.pt
downstream_data/mdgpt/Cora/1shot_graph_batch/splits.pt
```

Override with `DownstreamGeneratorConfig.downstream_root` or `downstream_root=` on `generate_*` (e.g. `downstream_data/rag_gfm`).

## Python API

```python
from pygfm.private.utlis.downstream_data_gen import (
    generate_few_shot_splits,
    generate_graph_batch_splits,
    DownstreamGeneratorConfig,
)

generate_few_shot_splits(
    dataset="Cora",
    data_root="datasets/mdgpt",
    downstream_root="downstream_data/mdgpt",
    k_shot=1,
    n_splits=100,
)
```

## CLI

From repo root:

```bash
python scripts/mdgpt/generate_downstream.py few_shot --dataset Cora --k_shot 5
python scripts/mdgpt/generate_downstream.py graph_batch --dataset Cora --k_shot 1
```

Default `--downstream_root downstream_data/mdgpt` (see `scripts/mdgpt/generate_downstream.py`).
