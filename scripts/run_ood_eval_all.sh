#!/bin/bash
# Run OOD evaluation for all trained models on a user-provided OOD dataset directory
# Usage: 
#   PEPSPECBENCH_OOD_DIR=/path/to/ood_reference bash scripts/run_ood_eval_all.sh
#   bash scripts/run_ood_eval_all.sh /path/to/ood_reference
#   bash scripts/run_ood_eval_all.sh /path/to/ood_reference parallel
#
# Note: Serial mode is default. Parallel mode can cause GPU OOM when 5+ models run at once.

set -e
cd "$(dirname "$0")/.."
RUN_DIR="output/latest_run_massive_kb_mini"
OOD_DIR="${1:-${PEPSPECBENCH_OOD_DIR:-data/ood_reference}}"
PARALLEL="${2:-serial}"
LOG_DIR="output/logs"
mkdir -p "$LOG_DIR"
PY="${PYTHON:-python}"

MODELS="prosit prosit_transformer predfull_torch unispec fastspel alphapeptdeep"
for m in $MODELS; do
  if [ -d "$RUN_DIR/$m" ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running OOD eval for $m on $OOD_DIR..."
    if [ "$PARALLEL" = "parallel" ]; then
      $PY scripts/eval_ood.py --model_name "$m" --ckpt_path "$RUN_DIR/$m" --ood_dir "$OOD_DIR" --batch_size 512 \
        > "$LOG_DIR/ood_eval_${m}.log" 2>&1 &
    else
      $PY scripts/eval_ood.py --model_name "$m" --ckpt_path "$RUN_DIR/$m" --ood_dir "$OOD_DIR" --batch_size 512 \
        2>&1 | tee "$LOG_DIR/ood_eval_${m}.log"
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished $m"
  else
    echo "Skip $m (no checkpoint)"
  fi
done
[ "$PARALLEL" = "parallel" ] && wait
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Check $LOG_DIR/ood_eval_*.log"
