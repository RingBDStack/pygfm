#!/usr/bin/env bash
# Packaged pipeline; unified CLI also: scripts/sa2gfm/attack_random.py
set -euo pipefail
AG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$AG_ROOT/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SA2GFM_DATA_ROOT="${SA2GFM_DATA_ROOT:-${REPO_ROOT}/datasets/sa2gfm/data}"
cd "$AG_ROOT"
DATASET="${1:?usage: $0 <dataset>}"
python pipeline/04_random_perturb.py --dataset "$DATASET" --type both
echo "Saved under $AG_ROOT/outputs/attacked_data_random/"
