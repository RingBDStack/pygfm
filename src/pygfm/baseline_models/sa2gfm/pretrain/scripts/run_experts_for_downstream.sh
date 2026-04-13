#!/usr/bin/env bash
# For downstream target T: pretrain only the *source* graphs in get_pretrain_datasets(T), not T itself.
# Matches down_all_sparse_multi/config.py get_pretrain_datasets.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:?usage: $0 <downstream_dataset> [extra args...]}"
shift || true
case "$TARGET" in
  cora)           EXPERTS=(citeseer pubmed P-home wikics) ;;
  citeseer)       EXPERTS=(cora pubmed P-home wikics) ;;
  pubmed)         EXPERTS=(cora citeseer P-home wikics) ;;
  P-tech)         EXPERTS=(cora citeseer pubmed P-home wikics) ;;
  P-home)         EXPERTS=(cora citeseer pubmed wikics) ;;
  wikics)         EXPERTS=(cora citeseer pubmed P-home) ;;
  arxiv)          EXPERTS=(P-home P-tech wikics) ;;
  *) echo "Unknown target: $TARGET"; exit 1 ;;
esac
for e in "${EXPERTS[@]}"; do
  echo "=== expert dataset: $e (for downstream $TARGET) ==="
  bash "$ROOT/pretrain/scripts/run_pretrain.sh" "$e" "$@"
done
