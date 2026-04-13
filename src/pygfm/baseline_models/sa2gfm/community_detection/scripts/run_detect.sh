#!/usr/bin/env bash
# Generate communities for one dataset (default: Louvain)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/attack_data_gen:${PYTHONPATH:-}"
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  # ROOT = SA2GFM repo root (…/community_detection/scripts/../..)
  export SA2GFM_DATA_ROOT="${ROOT}/data"
fi
DS="${1:?usage: $0 <dataset> [method]}"
METHOD="${2:-louvain}"
python "$ROOT/community_detection/pipeline/01_detect_communities.py" --dataset "$DS" --method "$METHOD"
echo "Written to: ${SA2GFM_DATA_ROOT}/communities/${DS}_communities.pt"
