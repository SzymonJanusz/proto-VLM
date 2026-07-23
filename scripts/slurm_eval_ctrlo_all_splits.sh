#!/bin/bash
# Run CTRL-O on all 7 splits PNP itself is evaluated on (mirrors
# proto-non-param/scripts/slurm_eval_vg_contrastive_M1.sh's split structure):
#   refcocog(google): val
#   refcoco(unc):      val testA testB
#   refcoco+(unc):     val testA testB
#
# Submits 3 jobs (slurm_eval_ctrlo.sh evaluates all requested splits for one
# dataset within a single model load), not 7 -- one job per dataset.
#
# Required env vars:
#   CKPT, CONFIG  — path to CTRL-O .ckpt / config.yaml
#
# Usage:
#   CKPT=$SCRATCH/ctrl-o/pretrained_models/ctrlo/pretrained_model.ckpt \
#   CONFIG=$SCRATCH/ctrl-o/pretrained_models/ctrlo/config.yaml \
#       bash scripts/slurm_eval_ctrlo_all_splits.sh

set -euo pipefail

: "${CKPT:?'CKPT env var required (path to .ckpt file)'}"
: "${CONFIG:?'CONFIG env var required (path to config.yaml)'}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found: $CKPT"
    exit 1
fi
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: config not found: $CONFIG"
    exit 1
fi

echo "=== CTRL-O — all splits ==="
echo "  Ckpt   : $CKPT"
echo "  Config : $CONFIG"
echo ""

submit() {
    local dataset="$1" splitby="$2" splits="$3"
    JOB=$(DATASET="$dataset" SPLITBY="$splitby" SPLITS="$splits" \
          CKPT="$CKPT" CONFIG="$CONFIG" \
          sbatch --parsable --job-name="ctrlo-eval-${dataset}" \
          "${SCRIPT_DIR}/slurm_eval_ctrlo.sh")
    echo "  ${JOB}  ${dataset} (${splitby})  splits: ${splits}"
}

submit refcocog  google "val"
submit refcoco   unc    "val testA testB"
submit "refcoco+" unc   "val testA testB"

echo ""
echo "3 jobs submitted (7 splits total). Monitor with: squeue -u \$USER"
echo "Results land in eval_results/ctrlo/{refcocog,refcoco,refcoco+}_metrics.json"
echo "(and are auto-copied to \$SCRATCH/eval_results/ctrlo/ by slurm_eval_ctrlo.sh)"
