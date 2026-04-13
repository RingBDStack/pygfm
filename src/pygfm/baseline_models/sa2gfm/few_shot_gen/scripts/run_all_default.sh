#!/usr/bin/env bash
# All default datasets × (1-shot, 5-shot). n_splits=20 (same as downstream default --num-splits).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export SA2GFM_DATA_ROOT="${SA2GFM_DATA_ROOT:-$(cd "$ROOT/../.." && pwd)/data}"
DATASETS=(cora citeseer pubmed wikics arxiv P-tech P-home)
for ds in "${DATASETS[@]}"; do
  for k in 1 5; do
    echo "=== $ds ${k}-shot (20 splits) ==="
    bash "$ROOT/few_shot_gen/scripts/run_generate.sh" "$ds" "$k" --n-splits 20
  done
done
