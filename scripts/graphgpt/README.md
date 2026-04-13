# GraphGPT

Training uses HuggingFace-style arguments flattened to YAML (see `pygfm/public/cli/yaml_flat_to_argv.py`).

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments

| Stage | Typical entry | YAML template |
|-------|----------------|----------------|
| Stage-1 alignment (train_mem) | `python scripts/graphgpt/run_with_config.py -c configs/graphgpt/train_mem_template.yaml` | `configs/graphgpt/train_mem_template.yaml` |
| Smoke | `configs/graphgpt/train_mem_smoke.yaml` | 1-step sanity check |

Shell equivalents (multi-GPU): `scripts/graphgpt/tune_script/graphgpt_stage1_1gpu.sh`, `graphgpt_stage2_1gpu.sh`.

Data prep helpers: `prepare_data.py`, `ensure_stage1_files.py`, `prepare_models.py`.

## Export YAML

```bash
python scripts/graphgpt/run_with_config.py --export-default-yaml configs/graphgpt/_defaults.yaml
python scripts/graphgpt/run_with_config.py --export-run-yaml configs/graphgpt/_run.yaml -c configs/graphgpt/train_mem_smoke.yaml
```

## Environment

Set `GRAPHGPT_NO_FLASH_ATTN=1` if FlashAttention is unavailable.
