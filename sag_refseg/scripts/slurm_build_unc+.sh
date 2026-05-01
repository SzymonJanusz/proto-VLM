#!/bin/bash
#SBATCH --job-name=sag-build-unc+
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/build_unc+_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/build_unc+_%j.err

# Build the 4 unc+ .npz batch files.
# Submit from repo root: sbatch sag_refseg/scripts/slurm_build_unc+.sh

set -euo pipefail

SCRATCH=/net/tscratch/people/plgabedychaj
REPO=~/proto-VLM

mkdir -p "$SCRATCH/logs"

module load Python/3.10.4
module load CUDA/12.4.0
module load cuDNN/9.2.1.18-CUDA-12.4.0

source "$SCRATCH/venv/bin/activate"

export HF_HUB_CACHE="$SCRATCH/.cache/huggingface/hub"
export TRANSFORMERS_CACHE="$SCRATCH/.cache/huggingface/hub"
export PYTHONPATH="$SCRATCH/refer:$PYTHONPATH"

COCO_DIR="$SCRATCH/ctrl-o/data/images/mscoco/images"
REFER_DIR="$SCRATCH/ctrl-o"
OUTPUT_DIR="$SCRATCH/data/refcoco"
VOCAB_FILE="$REPO/data/vocabulary_Gref.txt"

echo "==> Job started on $(hostname) at $(date)"

cd "$REPO"

for split in train val testA testB; do
    echo ""
    echo "--- Building unc+ / $split ---"
    python sag_refseg/scripts/build_batches.py \
        -d "unc+" -t "$split" \
        --coco-dir "$COCO_DIR" \
        --refer-data "$REFER_DIR/data" \
        --vocab-file "$VOCAB_FILE" \
        --output-dir "$OUTPUT_DIR"
done

echo ""
echo "==> Done at $(date)"
find "$OUTPUT_DIR/unc+" -name '*.npz' | wc -l
