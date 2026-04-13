# OneForAll (GFM-Toolbox)

## Install

At repo root:

```bash
pip install -e ".[oneforall]"
```

For DGL-related tasks: `pip install -e ".[oneforall-dgl]"` (pick a DGL wheel that matches your CUDA).

## Run (this folder is YAML-only entry)

From **any working directory** (child training processes `cwd` to repo root). Paths after `-c` resolve relative to CWD first, then repo root:

```bash
python scripts/oneforall/run.py -c configs/oneforall/smoke.yaml
# Skip pulling HuggingFace sentence-encoder weights (random placeholder encodings, smoke test only):
python scripts/oneforall/run.py -c configs/oneforall/smoke_stub.yaml
python scripts/oneforall/run.py -c pygfm/baseline_models/oneforall/e2e_all_config.yaml
```

Append `key value` overrides like `run_cdm`:

```bash
python scripts/oneforall/run.py -c configs/oneforall/smoke.yaml num_epochs 2 logger_backend none
```

Export merged YAML (no training):

```bash
python scripts/oneforall/run.py --export-run-yaml /tmp/merged.yaml -c configs/oneforall/smoke.yaml
```

**Without YAML**, from repo root:

```bash
python -m pygfm.baseline_models.oneforall.run_cdm task_names cora_node num_epochs 1
```

(In-package `run_cdm.py` sets `HF_ENDPOINT=https://hf-mirror.com` before other imports.)

## Environment variables

| Variable | Meaning |
|----------|---------|
| `ONEFORALL_DATA_ROOT` | **Your** PyG graph assets root (default `<repo>/datasets/oneforall`). Smoke Cora: place `Cora.pt` here for the `cora_node` task. |
| `ONEFORALL_CACHE_ROOT` | OFA preprocessing / text encoding cache (default `<ONEFORALL_DATA_ROOT>/cache_data`) |
| `ONEFORALL_EXP_ROOT` | Experiment outputs (default `<repo>/ckpts/oneforall/runs`) |
| `HF_ENDPOINT` | Hugging Face endpoint (if unset, package `run_cdm` defaults to the mirror) |

See `docs/oneforall/ONEFORALL_INTEGRATION_PLAN.md`.

## SwanLab / WandB

Set in **YAML** or **`run.py` trailing opts**, e.g.:

`logger_backend swanlab`, `offline_log False`, `swanlab_mode cloud` (cloud needs `swanlab login` first).

Defaults: `pygfm/baseline_models/oneforall/configs/default_config.yaml`.
