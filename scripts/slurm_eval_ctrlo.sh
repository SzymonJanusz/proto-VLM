#!/bin/bash
#SBATCH --job-name=ctrlo-eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/ctrlo_eval_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/ctrlo_eval_%j.err

# Evaluate CTRL-O on RefCOCO / RefCOCO+ / RefCOCOg using inference_refer.py.
# Loads model once and evaluates all requested splits in a single job.
#
# Required env vars:
#   DATASET  — refcoco | refcoco+ | refcocog
#   CKPT     — path to CTRL-O .ckpt checkpoint
#   CONFIG   — path to CTRL-O config.yaml
#
# Optional env vars:
#   SPLITS   — space-separated split names (default: "val testA testB")
#              (RefCOCOg only has "val"; set SPLITS="val" for that dataset)
#   SPLITBY  — unc | google (default: unc; use "google" for refcocog)
#
# Examples:
#   # RefCOCO (val + testA + testB)
#   DATASET=refcoco \
#       CKPT=$SCRATCH/ctrl-o/pretrained_models/ctrlo/pretrained_model.ckpt \
#       CONFIG=$SCRATCH/ctrl-o/pretrained_models/ctrlo/config.yaml \
#       sbatch scripts/slurm_eval_ctrlo.sh
#
#   # RefCOCOg (val only, google split)
#   DATASET=refcocog SPLITS="val" SPLITBY=google \
#       CKPT=$SCRATCH/ctrl-o/pretrained_models/ctrlo/pretrained_model.ckpt \
#       CONFIG=$SCRATCH/ctrl-o/pretrained_models/ctrlo/config.yaml \
#       sbatch scripts/slurm_eval_ctrlo.sh

set -euo pipefail

SCRATCH=/net/tscratch/people/plgabedychaj
REPO=~/proto-VLM
WORKDIR=$SCRATCH/ctrl-o

: "${DATASET:?'DATASET env var required (refcoco|refcoco+|refcocog)'}"
: "${CKPT:?'CKPT env var required (path to .ckpt file)'}"
: "${CONFIG:?'CONFIG env var required (path to config.yaml)'}"

SPLITS="${SPLITS:-val testA testB}"
SPLITBY="${SPLITBY:-unc}"

OUT_DIR="eval_results/ctrlo"

mkdir -p "$SCRATCH/logs" "$OUT_DIR"

echo "==> Job started on $(hostname) at $(date)"
echo "==> Evaluating CTRL-O: dataset=$DATASET  splits=($SPLITS)  splitBy=$SPLITBY"
echo "==> Checkpoint: $CKPT"

module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

source "$SCRATCH/venv/bin/activate"

export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/hub"
export WORKDIR="$WORKDIR"

cd "$REPO"

python inference_refer.py \
    --data_root "$SCRATCH/ctrl-o/data" \
    --dataset "$DATASET" \
    --splitBy "$SPLITBY" \
    --splits $SPLITS \
    --checkpoint "$CKPT" \
    --config "$CONFIG" \
    --image_root "$SCRATCH/ctrl-o/data/images/mscoco/images/train2014" \
    --output_dir "$OUT_DIR"

# Copy results to scratch for easy access
RESULT_JSON="${OUT_DIR}/${DATASET}_metrics.json"
if [ -f "$RESULT_JSON" ]; then
    mkdir -p "$SCRATCH/eval_results/ctrlo"
    cp "$RESULT_JSON" "$SCRATCH/eval_results/ctrlo/"
    echo "==> Results copied to $SCRATCH/eval_results/ctrlo/"
fi

echo "==> Eval done at $(date)"
