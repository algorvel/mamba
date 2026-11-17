#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd -- "$HERE/../.." && pwd)"
EXP="$(cd -- "$HERE/.." && pwd)"
DATASET="${SATEHAZE_ROOT:-/home/johnny/code/dataset/SateHaze1k}"
PYTHON="${DEHAZEMAMBA_PYTHON:-/home/johnny/miniconda3/envs/mamba/bin/python}"
ADAPTED="$EXP/data/satehaze1k_mrshaze"
OUT="$EXP/results/satehaze1k_all_levels"
LOG="$EXP/logs/train.log"
PIDFILE="$EXP/train.pid"

mkdir -p "$EXP/logs" "$EXP/results"
if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null && [[ "$(ps -o stat= -p "$(cat "$PIDFILE")" 2>/dev/null | cut -c1)" != Z ]]; then
  echo "Training already running (PID $(cat "$PIDFILE")); log: $LOG"
  exit 0
fi
[[ -x "$PYTHON" ]] || { echo "Python environment not found: $PYTHON" >&2; exit 1; }
"$PYTHON" "$HERE/prepare_satehaze.py" --dataset-root "$DATASET" --output-root "$ADAPTED"

cat > "$EXP/run_config.txt" <<EOF
start_time=$(date --iso-8601=seconds)
dataset=$DATASET
adapted_dataset=$ADAPTED
output=$OUT
model_size=${MODEL_SIZE:-T}
epochs=${EPOCHS:-150}
batch_size=${BATCH_SIZE:-6}
workers=${WORKERS:-6}
seed=${SEED:-2026}
EOF
cd "$PROJECT"
setsid "$PYTHON" "$PROJECT/train_dehaze.py" \
  --data-root "$ADAPTED" \
  --output-dir "$OUT" \
  --model-size "${MODEL_SIZE:-T}" \
  --epochs "${EPOCHS:-150}" \
  --batch-size "${BATCH_SIZE:-6}" \
  --workers "${WORKERS:-6}" \
  --seed "${SEED:-2026}" \
  > "$LOG" 2>&1 < /dev/null &
echo $! > "$PIDFILE"
sleep 8
if ! kill -0 "$(cat "$PIDFILE")" 2>/dev/null || [[ "$(ps -o stat= -p "$(cat "$PIDFILE")" 2>/dev/null | cut -c1)" == Z ]]; then
  echo "Training exited during startup; inspect $LOG" >&2
  exit 1
fi
echo "Training started detached: PID $(cat "$PIDFILE")"
echo "Log: $LOG"
echo "Checkpoints: $OUT"
