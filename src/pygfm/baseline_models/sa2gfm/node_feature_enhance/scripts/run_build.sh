#!/usr/bin/env bash
# Build enhanced_x_64 for one dataset (requires ori/*.pt + communities/*.pt).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${ROOT}/attack_data_gen:${PYTHONPATH:-}"
if [[ -z "${SA2GFM_DATA_ROOT:-}" ]]; then
  export SA2GFM_DATA_ROOT="$(cd "$ROOT/../.." && pwd)/data"
fi
DS="${1:?usage: $0 <dataset> [extra args passed to python...]}"
shift || true
python "$ROOT/node_feature_enhance/pipeline/01_build_enhanced_x.py" --dataset "$DS" "$@"
echo "Done (see --output or default: .../ori/${DS}_enhanced_x64.pt)"
