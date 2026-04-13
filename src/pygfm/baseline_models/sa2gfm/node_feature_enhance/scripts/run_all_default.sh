#!/usr/bin/env bash
# Run feature enhancement for the default seven datasets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTRA=("$@")
for ds in cora citeseer pubmed wikics arxiv P-home P-tech; do
  echo "=== $ds ==="
  bash "$ROOT/node_feature_enhance/scripts/run_build.sh" "$ds" "${EXTRA[@]}"
done
