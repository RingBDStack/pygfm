#!/usr/bin/env bash
# Generate few_shot/{dataset}/{k}shot/split_*.pt under SA2GFM_DATA_ROOT.
# Usage:
#   bash scripts/run_generate.sh <dataset> <1|5> [--n-splits N] [--n-way N] [--test-reserve N] [--write-example]
# Example:
#   bash scripts/run_generate.sh cora 5 --n-splits 20
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/attack_data_gen:${PYTHONPATH:-}"
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  export SA2GFM_DATA_ROOT="${ROOT}/data"
fi
if [[ $# -lt 2 ]]; then
  echo "usage: $0 <dataset> <k_shot:1|5> [extra args for 01_generate_splits.py, e.g. --n-splits 20]"
  exit 1
fi
DS="$1"
K="$2"
shift 2
exec python "$ROOT/few_shot_gen/pipeline/01_generate_splits.py" --dataset "$DS" --k-shot "$K" "$@"
