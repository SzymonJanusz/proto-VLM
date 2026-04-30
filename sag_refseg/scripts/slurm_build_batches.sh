#!/bin/bash
#SBATCH --job-name=sag-build-batches
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/sag_build_batches_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/sag_build_batches_%j.err

# Build all 10 .npz batch files required for sag_refseg training/evaluation.
# Submit from repo root: sbatch sag_refseg/scripts/slurm_build_batches.sh

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

REFER_DIR="$SCRATCH/refer"
COCO_DIR="$SCRATCH/ctrl-o/data/images/mscoco/images"
OUTPUT_DIR="$SCRATCH/data/refcoco"
VOCAB_FILE="$REPO/data/vocabulary_Gref.txt"

# refer library must be on PYTHONPATH (no C extension needed, pure Python)
export PYTHONPATH="$REFER_DIR:$PYTHONPATH"

echo "==> Job started on $(hostname) at $(date)"
echo "==> COCO images: $COCO_DIR"
echo "==> Refer data:  $REFER_DIR/data"
echo "==> Output:      $OUTPUT_DIR"

cd "$REPO"

build() {
    DATASET=$1
    SPLIT=$2
    echo ""
    echo "--- Building $DATASET / $SPLIT ---"
    python sag_refseg/scripts/build_batches.py \
        -d "$DATASET" -t "$SPLIT" \
        --coco-dir "$COCO_DIR" \
        --refer-data "$REFER_DIR/data" \
        --vocab-file "$VOCAB_FILE" \
        --output-dir "$OUTPUT_DIR"
}

build Gref  train
build Gref  val
build unc   train
build unc   val
build unc   testA
build unc   testB
build unc+  train
build unc+  val
build unc+  testA
build unc+  testB

echo ""
echo "==> All batches built at $(date)"
echo "==> Output layout:"
find "$OUTPUT_DIR" -name '*.npz' | awk -F/ '{print $(NF-1)}' | sort | uniq -c
