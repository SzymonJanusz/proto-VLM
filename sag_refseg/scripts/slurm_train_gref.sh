#!/bin/bash
#SBATCH --job-name=sag-train-gref
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/sag_train_gref_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/sag_train_gref_%j.err

# Train sag_refseg on RefCOCOg (Gref / Google split).
# Submit from repo root: sbatch sag_refseg/scripts/slurm_train_gref.sh
# Resume:                sbatch sag_refseg/scripts/slurm_train_gref.sh --ckpt <path>

set -euo pipefail

SCRATCH=/net/tscratch/people/plgabedychaj
REPO=~/proto-VLM

mkdir -p "$SCRATCH/logs"

echo "==> Job started on $(hostname) at $(date)"
echo "==> GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"

module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

source "$SCRATCH/venv/bin/activate"

export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/hub"
export WANDB_DIR="$SCRATCH/logs"

cd "$REPO"

python sag_refseg/train.py \
    --config sag_refseg/configs/gref.json \
    --data_root "$SCRATCH/data/refcoco" \
    --out_dir   "$SCRATCH/checkpoints/sag_refseg_Gref" \
    --workers 8 \
    "$@"

echo ""
echo "==> Training done at $(date)"
echo "==> Checkpoints: $SCRATCH/checkpoints/sag_refseg_Gref/Gref/"
