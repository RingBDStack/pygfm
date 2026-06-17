<div align="center">
<img src="https://raw.githubusercontent.com/RingBDStack/pygfm/main/assets/LOGO.png" style="width:30%; display:block; margin:0 auto;" alt="LOGO">

[![PyPI version](https://img.shields.io/pypi/v/python-gfm?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/python-gfm/)
[![Python](https://img.shields.io/pypi/pyversions/python-gfm?logo=python&logoColor=white)](https://pypi.org/project/python-gfm/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![PyPI Downloads](https://img.shields.io/pypi/dm/python-gfm?color=orange)](https://pypi.org/project/python-gfm/)

[Installation](#installation) · [Quick Start](#quick-start) · [Repository layout](#repository-layout) · [Supported Baselines](#supported-baselines) · [Documentation](#baseline-documentation)

</div>

---

`pygfm` is a unified Python toolkit for **Graph Foundation Model (GFM)** research. It integrates **17 state-of-the-art baseline methods** under a single, pip-installable package with shared utilities, standardized interfaces, and fully reproducible experiment pipelines.

Developed by **Beihang University · School of Computer Science and Engineering · ACT Lab · MAGIC GROUP**.

## Framework Overview

<div align="center">
  <img src="https://raw.githubusercontent.com/RingBDStack/pygfm/main/assets/framework.png" alt="PyGFM Framework Overview" width="90%">
</div>

PyGFM is organized into four stacked layers — **Graph Data Abstraction → Alignment & Fusion Bridge → Representation Backbones → Task Heads & Orchestration** — with a unified CLI, model recipes, and an auto-experiment tracker sitting on top.

## Highlights

- **One package, 17 baselines** — prompt-based GFMs, structure-aware models, LLM-integrated approaches, and retrieval-augmented methods all available via a single `pip install`.
- **Reproducible pipelines** — every baseline ships with YAML-driven experiment configs, training scripts, and evaluation helpers.
- **Shared backbone library** — common GNN encoders, loss functions, and data utilities are factored out and reused across all baselines, reducing code duplication.
- **CLI-first design** — launch pre-training, fine-tuning, and evaluation jobs directly from the command line without writing any boilerplate.
- **LLM-ready** — first-class support for LLM-integrated GFMs (GraphGPT, GraphText, LLaGA, OneForAll) with HuggingFace-compatible YAML configs.

## Installation

### CUDA (recommended)

**Default (fresh env): `torch` + `light` together** — PyTorch wheel index + PyPI + PyG find-links:

```bash
pip install "python-gfm[torch,light]" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
```

**If CUDA PyTorch / PyG is already in the env** — install **`[light]`** from PyPI only:

```bash
pip install "python-gfm[light]"
```

**LLM-integrated GFMs** — after **`[torch]`** and **`[light]`** are in place:

```bash
pip install "python-gfm[llm]"
```

> **CPU:** `--index-url https://download.pytorch.org/whl/cpu` and `-f https://data.pyg.org/whl/torch-2.8.0+cpu.html`.

### Extras overview

| Extra | Contents (short) |
|-------|------------------|
| **`torch`** | PyTorch Geometric stack, graph libs, sklearn helpers |
| **`light`** | NumPy/Pandas stack, Transformers, Hydra, APIs, Gradio, W&B, SwanLab |
| **`llm`** | PEFT, bitsandbytes, datasets, fschat, Ray, Vertex, DeepSpeed |

### Optional `dev` extra

`pip install "python-gfm[dev]"` adds `pytest` and `ruff` for testing and linting.

## Quick Start

This repository is the **PyPI package source**. For installation, quick start, YAML configs, Hugging Face asset download, CLI commands, and per-baseline experiment steps, see the **[PyPI project page](https://pypi.org/project/python-gfm/)**.

## Repository layout

```
.
├── assets/                     # Logo and framework figures
├── ckpts/                      # Example / default checkpoint outputs
├── dist/                       # Built wheel artifacts (release)
├── scripts/                    # Per-baseline experiment scripts and configs
│   ├── bridge/
│   ├── gcot/
│   ├── graphgpt/
│   ├── graphkeeper/
│   ├── graphmore/
│   ├── graphprompt/
│   ├── graphtext/
│   ├── graver/
│   ├── hgprompt/
│   ├── llaga/
│   ├── mdgfm/
│   ├── mdgpt/
│   ├── multigprompt/
│   ├── oneforall/
│   ├── rag_gfm/
│   ├── sa2gfm/
│   └── samgpt/
├── src/pygfm/                  # Installable package source
│   ├── baseline_models/        # GFM baseline implementations (17 methods)
│   ├── cli/                    # Console entry points and YAML stage runners
│   ├── download/               # `python -m pygfm.download` module shim
│   ├── private/                # Core encoders and internal helpers
│   │   ├── core/
│   │   └── utlis/              # Domain alignment, RAG builders, data gen, etc.
│   ├── public/                 # Shared utilities exposed across baselines
│   │   ├── backbone_models/    # Reusable GNN encoders
│   │   ├── cli/                # YAML / config helpers
│   │   └── utils/              # Data, loss, LLM, and misc helpers
│   ├── tools/                  # In-package maintenance utilities
│   ├── tool_download.py        # Hugging Face dataset downloader
│   ├── _scripts_bundle.zip     # Packaged copy of `scripts/` (shipped in wheel)
│   └── __main__.py             # `python -m pygfm` entry (run YAML / download)
├── tools/                      # Repo-level build helpers (e.g. wheel bundling)
├── pyproject.toml              # Package metadata and dependency pins
└── README.md
```

After `pip install`, the installed package exposes the same `pygfm/` tree (with `scripts/` bundled as `_scripts_bundle.zip` inside the wheel).

## Supported Baselines

| Category | Methods |
|---|---|
| **Prompt-based GFM** | MDGPT, SAMGPT, MDGFM, GraphPrompt, HGPrompt, MultiGPrompt, GCoT |
| **Structure-aware GFM** | SA2GFM, Bridge, GraphKeeper, GraphMore, Graver |
| **LLM-integrated GFM** | GraphGPT, GraphText, LLaGA, OneForAll |
| **Retrieval-augmented GFM** | RAG-GFM |

## Baseline Documentation

Per-method setup, data layout, and evaluation notes live under `scripts/<baseline>/`. Index:

| Baseline | Docs |
|---|---|
| MDGPT | [scripts/mdgpt/README.md](scripts/mdgpt/README.md) |
| SA2GFM | [scripts/sa2gfm/README.md](scripts/sa2gfm/README.md) |
| SAMGPT | [scripts/samgpt/README.md](scripts/samgpt/README.md) |
| MDGFM | [scripts/mdgfm/README.md](scripts/mdgfm/README.md) |
| GraphPrompt | [scripts/graphprompt/README.md](scripts/graphprompt/README.md) |
| HGPrompt | [scripts/hgprompt/README.md](scripts/hgprompt/README.md) |
| MultiGPrompt | [scripts/multigprompt/README.md](scripts/multigprompt/README.md) |
| GCoT | [scripts/gcot/README.md](scripts/gcot/README.md) |
| Graver | [scripts/graver/README.md](scripts/graver/README.md) |
| GraphMore | [scripts/graphmore/README.md](scripts/graphmore/README.md) |
| Bridge | [scripts/bridge/README.md](scripts/bridge/README.md) |
| GraphKeeper | [scripts/graphkeeper/README.md](scripts/graphkeeper/README.md) |
| GraphGPT | [scripts/graphgpt/README.md](scripts/graphgpt/README.md) |
| GraphText | [scripts/graphtext/README.md](scripts/graphtext/README.md) |
| LLaGA | [scripts/llaga/README.md](scripts/llaga/README.md) |
| OneForAll | [scripts/oneforall/README.md](scripts/oneforall/README.md) |
| RAG-GFM | [scripts/rag_gfm/README.md](scripts/rag_gfm/README.md) |

Several baselines also ship an `Experiment-Manual.md` under `scripts/<baseline>/` with step-by-step reproduction notes.

## Requirements

Versions below match [`pyproject.toml`](pyproject.toml).

### Base (always installed)

| Dependency | Version |
|---|---|
| Python | ≥ 3.12 |
| NumPy | ≥ 1.20 |
| PyYAML | ≥ 6 |

### `[torch]` extra

| Dependency | Version |
|---|---|
| PyTorch | 2.8.0 |
| PyTorch Geometric | 2.7.0 |
| torch-scatter | 2.1.2 |
| torch-sparse | 0.6.18 |
| torch-cluster | 1.6.3 |
| torch-spline-conv | 1.2.2 |
| Lightning | ≥ 2.2.0, < 3 |
| torchmetrics | ≥ 1.3.0 |
| Accelerate | ≥ 0.26.0 |
| OGB | 1.3.6 |
| geoopt | 0.5.1 |
| scikit-learn | ≥ 1.3.0 |

### `[light]` extra

| Dependency | Version |
|---|---|
| Transformers | ≥ 4.36.0 |
| Hugging Face Hub | 1.10.2 |
| Hydra Core | 1.3.2 |
| OmegaConf | 2.3.0 |
| NumPy | 2.3.2 |
| Pandas | 3.0.2 |
| scikit-learn | 1.8.0 |
| Gradio | ≥ 5.0.0 |
| W&B | ≥ 0.19.0 |
| SwanLab | ≥ 0.7.11, < 0.8 |

### `[llm]` extra

| Dependency | Version |
|---|---|
| Transformers | 5.5.4 |
| Accelerate | 1.13.0 |
| PEFT | 0.19.0 |
| bitsandbytes | 0.49.2 |
| DeepSpeed | 0.18.9 |
| datasets | 4.8.4 |
| sentence-transformers | 5.4.1 |

See [`pyproject.toml`](pyproject.toml) for the full pinned dependency list.

## License

This project is licensed under the **[Apache License 2.0](LICENSE)**.

## Team

**MAGIC GROUP** — Beihang University, School of Computer Science and Engineering, ACT Lab.

---

<div align="center">
<sub>If you find this toolkit useful in your research, please consider <a href="https://github.com/RingBDStack/pygfm/">starring the repository</a> ⭐</sub>
</div>
