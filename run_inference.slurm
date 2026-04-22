#!/bin/bash
#SBATCH --job-name=ctrlo-refer
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -e

SCRATCH=/net/tscratch/people/plgabedychaj
WORKDIR=$SCRATCH/ctrl-o
export WORKDIR

echo "==> Job started on $(hostname) at $(date)"
echo "==> WORKDIR: $WORKDIR"

# Load modules
module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

export PIP_CACHE_DIR=$SCRATCH/.cache/pip
export HF_HUB_CACHE=$SCRATCH/.cache/huggingface/hub

source "$WORKDIR/venv/bin/activate"

CHECKPOINT=$WORKDIR/pretrained_models/ctrlo/pretrained_model.ckpt
CONFIG=$WORKDIR/pretrained_models/ctrlo/config.yaml
IMAGE_ROOT=$WORKDIR/data/images/mscoco/images/train2014
DATA_ROOT=$WORKDIR/data
OUTPUT_DIR=$WORKDIR/results_refer

mkdir -p "$OUTPUT_DIR"

# Use inference script from the directory where sbatch was submitted
SCRIPT=$SLURM_SUBMIT_DIR/inference_refer.py

# --- RefCOCO ---
echo ""
echo "==> Running RefCOCO inference..."
python "$SCRIPT" \
    --data_root "$DATA_ROOT" \
    --dataset refcoco \
    --splitBy unc \
    --splits val testA testB \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

# --- RefCOCO+ ---
echo ""
echo "==> Running RefCOCO+ inference..."
python "$SCRIPT" \
    --data_root "$DATA_ROOT" \
    --dataset "refcoco+" \
    --splitBy unc \
    --splits val testA testB \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --image_root "$IMAGE_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --device cuda

echo ""
echo "==> All done at $(date)"
echo "==> Results in $OUTPUT_DIR"
