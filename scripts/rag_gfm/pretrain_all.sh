#!/usr/bin/env bash
# RAG-GFM: leave-one-out pretraining for every downstream target dataset.
# Each target saves to ckpts/rag_gfm/{target}/preprompt_{target}.pth
#
# Usage:
#   bash scripts/rag_gfm/pretrain_all.sh
#   bash scripts/rag_gfm/pretrain_all.sh --no_swanlab
#   bash scripts/rag_gfm/pretrain_all.sh --max_epochs 100

set -e
cd "$(dirname "$0")/../.."
ROOT="$(pwd)"

# Downstream dataset list (matches directories under downstream_data/rag_gfm; excludes corpus / motif_lib; no Reddit / Photo)
TARGETS=(Cora Citeseer Pubmed Computers Home Tech Wikics)

# Optional extra args for pretrain.py, e.g. --no_swanlab --max_epochs 100
EXTRA_ARGS=("$@")

echo ">> RAG-GFM full pretrain run; targets = ${TARGETS[*]}"
echo ">> Working directory: $ROOT"
echo ">> Extra args: ${EXTRA_ARGS[*]}"
echo ""

for target in "${TARGETS[@]}"; do
  echo "========== target: $target =========="
  python scripts/rag_gfm/pretrain.py --target_dataset "$target" "${EXTRA_ARGS[@]}"
  echo ""
done

echo ">> All done. Checkpoints under ckpts/rag_gfm/{target}/."
