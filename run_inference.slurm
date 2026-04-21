#!/bin/bash
#SBATCH --job-name=ctrlo-refer
#SBATCH --partition=plgrid-gpu-a100   # check available partitions with: sinfo
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

source "$WORKDIR/venv/bin/activate"

CHECKPOINT=$WORKDIR/pretrained_models/ctrlo/pretrained_model.ckpt
CONFIG=$WORKDIR/pretrained_models/ctrlo/config.yaml
IMAGE_ROOT=$WORKDIR/data/images/mscoco/images/train2014
DATA_ROOT=$WORKDIR/data
OUTPUT_DIR=$WORKDIR/results_refer

mkdir -p "$OUTPUT_DIR"

# Copy inference script to workdir (if running from a different source dir)
cp "$WORKDIR/../inference_refer.py" "$WORKDIR/inference_refer.py" 2>/dev/null || true

# --- RefCOCO ---
echo ""
echo "==> Running RefCOCO inference..."
python "$WORKDIR/inference_refer.py" \
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
python "$WORKDIR/inference_refer.py" \
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
