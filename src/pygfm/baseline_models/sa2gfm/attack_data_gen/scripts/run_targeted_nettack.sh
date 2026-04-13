#!/usr/bin/env bash
# Packaged pipeline; unified CLI also: scripts/sa2gfm/attack_targeted_nettack.py
set -euo pipefail
AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AG_ROOT/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SA2GFM_DATA_ROOT="${SA2GFM_DATA_ROOT:-${REPO_ROOT}/datasets/sa2gfm/data}"
cd "$AG_ROOT"

DATASET="${1:?usage: $0 <dataset> [p]}"
P="${2:-1}"

echo "SA2GFM_DATA_ROOT=$SA2GFM_DATA_ROOT"
echo "dataset=$DATASET p=$P"

if [[ "${SKIP_SURROGATE_TRAIN:-0}" != "1" ]]; then
  python pipeline/01_train_gcn_surrogate.py --datasets "$DATASET" --device "${ATTACK_DEVICE:-cuda}"
else
  echo "SKIP_SURROGATE_TRAIN=1 -> reuse checkpoints/gcn_${DATASET}.pth"
fi
python pipeline/02_nettack_reports.py --dataset "$DATASET" --p "$P"
python pipeline/03_assemble_final.py --dataset "$DATASET" --p-values "$P"

echo "Final graphs under: $AG_ROOT/outputs/attack_post/${DATASET}_p${P}_final/"
