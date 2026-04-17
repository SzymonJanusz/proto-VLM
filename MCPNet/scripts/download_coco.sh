#!/bin/bash
#SBATCH --job-name=coco_download
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-08:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/coco_download_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/coco_download_%j.err

# Downloads COCO 2017 (~19 GB) and converts to ImageFolder format.
# train2017: ~18 GB / ~118k images
# val2017:   ~1 GB  /  ~5k images
# Each image assigned to the dominant-area category (largest total bbox area).
# Output: /net/tscratch/people/plgabedychaj/coco_dataset/{train,val}/<class>/

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

SCRATCH=/net/tscratch/people/plgabedychaj
PYTHON="${SCRATCH}/conda_envs/MCPNet/bin/python"

cd ~/proto-VLM/MCPNet

"${PYTHON}" scripts/prepare_coco.py \
  --out_dir "${SCRATCH}/coco_dataset" \
  --raw_dir "${SCRATCH}/coco_dataset/raw"
