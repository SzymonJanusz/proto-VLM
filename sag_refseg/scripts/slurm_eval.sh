#!/bin/bash
#SBATCH --job-name=sag-eval
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/sag_eval_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/sag_eval_%j.err

# Evaluate sag_refseg on a single dataset + split.
# Required env vars (set via sbatch --export or wrapper script):
#   DATASET  — Gref | unc | unc+
#   SPLIT    — val | testA | testB
#   CKPT     — path to best.pth
#
# Examples:
#   DATASET=Gref SPLIT=val CKPT=$SCRATCH/checkpoints/sag_refseg_Gref/Gref/best.pth \
#       sbatch sag_refseg/scripts/slurm_eval.sh
#
#   # Evaluate all unc splits after training:
#   for split in val testA testB; do
#       DATASET=unc SPLIT=$split CKPT=$SCRATCH/checkpoints/sag_refseg_unc/unc/best.pth \
#           sbatch sag_refseg/scripts/slurm_eval.sh
#   done

set -euo pipefail

SCRATCH=/net/tscratch/people/plgabedychaj
REPO=~/proto-VLM

: "${DATASET:?'DATASET env var required (Gref|unc|unc+)'}"
: "${SPLIT:?'SPLIT env var required (val|testA|testB)'}"
: "${CKPT:?'CKPT env var required (path to .pth file)'}"

mkdir -p "$SCRATCH/logs"
mkdir -p "$SCRATCH/eval_results/sag_refseg"

echo "==> Job started on $(hostname) at $(date)"
echo "==> Evaluating: dataset=$DATASET  split=$SPLIT"
echo "==> Checkpoint: $CKPT"

module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

source "$SCRATCH/venv/bin/activate"

export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/hub"

cd "$REPO"

# Map dataset name to config file
case "$DATASET" in
    Gref)  CONFIG=sag_refseg/configs/gref.json ;;
    unc)   CONFIG=sag_refseg/configs/unc.json ;;
    "unc+") CONFIG="sag_refseg/configs/unc+.json" ;;
    *) echo "Unknown dataset: $DATASET" && exit 1 ;;
esac

python sag_refseg/evaluate.py \
    --config "$CONFIG" \
    --data_root "$SCRATCH/data/refcoco" \
    --data_split "$SPLIT" \
    --ckpt "$CKPT"

# Copy results to scratch for easy access
RESULT_JSON="eval_results/sag_refseg/${DATASET}_${SPLIT}.json"
if [ -f "$RESULT_JSON" ]; then
    cp "$RESULT_JSON" "$SCRATCH/eval_results/sag_refseg/"
    echo "==> Results copied to $SCRATCH/eval_results/sag_refseg/"
fi

echo "==> Eval done at $(date)"
