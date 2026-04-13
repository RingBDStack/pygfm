# GraphText (Hydra)

GraphText uses **Hydra** configs under `scripts/graphtext/config/`. Entry scripts live in `scripts/graphtext/` and `pygfm/baseline_models/graphtext/`.

## Install

```bash
cd /path/to/repo
pip install -e .
```

## Experiments

| Mode | Entry | YAML template |
|------|--------|----------------|
| SFT | `python scripts/graphtext/run_with_config.py -c configs/graphtext/sft.yaml` | `configs/graphtext/sft.yaml` |
| ICL | `python scripts/graphtext/run_with_config.py -c configs/graphtext/icl.yaml` | `configs/graphtext/icl.yaml` |
| Smoke SFT | `configs/graphtext/sft_smoke.yaml` | tiny / offline-friendly overrides |
| Smoke ICL | `configs/graphtext/icl_smoke.yaml` | |

Hydra overrides are listed under `hydra_overrides` in the wrapper YAML (see `run_with_config.py`).

Direct Hydra (advanced):

```bash
python scripts/graphtext/run_sft.py exp=sft data=cora
python scripts/graphtext/run_icl.py exp=icl data=cora
```

## API keys

Copy `scripts/graphtext/config/user/env.yaml.example` → `env.yaml` and set `openai_api_key` / `deepseek_api_key`, or use environment variables. Do not commit real secrets.

## Export wrapper YAML

```bash
python scripts/graphtext/run_with_config.py --export-default-yaml configs/graphtext/_template.yaml
python scripts/graphtext/run_with_config.py --export-run-yaml configs/graphtext/_exported.yaml -c configs/graphtext/sft.yaml
```
