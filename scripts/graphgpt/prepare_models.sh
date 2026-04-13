#!/bin/bash
# Download GraphGPT weights: Vicuna-7B-v1.5 + graph encoder Arxiv-PubMed-GraphCLIP-GT
# Usage: bash scripts/prepare_models.sh  or  ./scripts/prepare_models.sh

set -e
cd "$(dirname "$0")/.."

echo "=== GraphGPT model download ==="
echo "Downloads go under ckpts/graphgpt/checkpoints/:"
echo "  - vicuna-7b-v1.5   (~14GB)"
echo "  - clip_gt_arxiv    (graph encoder)"
echo ""

pip install -q huggingface_hub 2>/dev/null || true
python scripts/prepare_models.py
