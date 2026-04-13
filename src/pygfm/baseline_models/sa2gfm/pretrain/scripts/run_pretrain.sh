#!/usr/bin/env bash
# Train ONE checkpoint: ori/{DS}.pt -> save_model/{DS}.pt (MoE expert for graph name DS).
# For a downstream target T, the usual experts are *other* graphs; use run_experts_for_downstream.sh T.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/pretrain/pipeline:${ROOT}/attack_data_gen:${PYTHONPATH:-}"
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  export SA2GFM_DATA_ROOT="${ROOT}/data"
fi
DS="${1:?usage: $0 <dataset> [extra args to train_single.py...]}"
shift || true
python "$ROOT/pretrain/pipeline/train_single.py" --dataset "$DS" "$@"
