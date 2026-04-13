# MultiGPrompt

- **Library code**: `pygfm/baseline_models/multigprompt/`
- **Experiment driver**: **`execute.py`** in this folder

## Install

```bash
cd /path/to/repo
pip install -e .
```

Run from **repository root**:

```bash
python scripts/multigprompt/execute.py --dataset cora --gpu 0
```

### Config files (YAML + Pydantic validation, recommended)

Long option lists can go in YAML and be passed with `--config` / `-c`; explicit CLI flags **override** the file.

```bash
pip install pydantic pyyaml

python scripts/multigprompt/execute.py --config configs/multigprompt/example_cora.yaml
python scripts/multigprompt/execute.py -c configs/multigprompt/example_cora.yaml --epochs 500 --gpu 1
```

- Config schema: `pygfm/public/cli/multigprompt_config.py` (`MultiGPromptRunConfig`)
- Example: `configs/multigprompt/example_cora.yaml`

Common flags: `--epochs`, `--tasks`, `--inner_steps`, `--save_name` (default `ckpts/multigprompt/checkpoints/<dataset>.pkl`).

## Data layout (aligned with MDGPT: upstream / downstream split)

| Kind | Default path | Notes |
|------|--------------|--------|
| **Upstream** (Planetoid) | `datasets/multigprompt/<dataset>/` | Per-dataset subdirs, e.g. `.../cora/ind.cora.*` |
| **Downstream** (few-shot + CSV) | `downstream_data/multigprompt/<dataset>/` | e.g. `.../cora/fewshot_cora/`, `cora_fewshot.csv`; repo ships Cora only |

See also:

- `datasets/multigprompt/README.md`
- `downstream_data/multigprompt/README.md`

### Migrating from a single flat directory

If everything used to live in one folder:

- Put `ind.cora.*` under **`datasets/multigprompt/cora/`** (same idea for `citeseer`)
- Put Cora few-shot assets under **`downstream_data/multigprompt/cora/`**
- `MULTIGPROMPT_*_DATA_DIR` env vars point at a **parent** directory whose children are `cora/`, `citeseer/`, etc.—not a single dataset folder

Or keep **`MULTIGPROMPT_DATA_DIR`** pointing at the old combined directory for legacy behavior.

Optional extras: `pip install -e ".[multigprompt]"` (see root `pyproject.toml`).
