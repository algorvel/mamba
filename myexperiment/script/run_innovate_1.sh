#!/usr/bin/env bash
# Uses the existing normalized dataset; runs in the foreground.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$HERE/../.." && pwd)"
EXP="$(cd -- "$HERE/.." && pwd)"
PYTHON="${DEHAZEMAMBA_PYTHON:-python}"
exec "$PYTHON" "$PROJECT/train_dehaze.py" \
  --experiment innovate-1 \
  --data-root "${INNOVATE_DATA_ROOT:-$EXP/data/satehaze1k_mrshaze}" \
  --output-dir "${INNOVATE_OUTPUT_DIR:-$EXP/results/innovate_1}" \
  --model-size "${MODEL_SIZE:-T}" \
  --epochs "${EPOCHS:-150}" \
  --batch-size "${BATCH_SIZE:-6}" \
  --workers "${WORKERS:-6}" \
  --seed "${SEED:-2026}" "$@"
