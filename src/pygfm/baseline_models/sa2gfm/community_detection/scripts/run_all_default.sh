#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
METHOD="${1:-louvain}"
for ds in cora citeseer pubmed wikics arxiv P-home P-tech; do
  echo "=== $ds ==="
  bash "$ROOT/community_detection/scripts/run_detect.sh" "$ds" "$METHOD"
done
