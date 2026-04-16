#!/bin/bash
# Setup MCPNet conda environment on Athena and download pretrained weights.
#
# Run once after cloning the repo:
#   bash ~/proto-VLM/MCPNet/scripts/setup_env.sh

set -e

SCRATCH=/net/tscratch/people/plgabedychaj
PRETRAINED_DIR="${SCRATCH}/pretrained"
REPO="${HOME}/proto-VLM"

echo "=== Creating MCPNet conda environment ==="
conda env create -f "${REPO}/MCPNet/environment.yml"

echo ""
echo "=== Activating environment ==="
# shellcheck disable=SC1090
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MCPNet

echo ""
echo "=== Downloading pretrained backbone weights ==="
mkdir -p "${PRETRAINED_DIR}"
cd "${REPO}/MCPNet"
python scripts/download_pretrained.py --output-dir "${PRETRAINED_DIR}"

echo ""
echo "=== Setup complete ==="
echo "Pretrained weights saved to: ${PRETRAINED_DIR}"
echo ""
echo "To submit training jobs:"
echo "  sbatch ${REPO}/MCPNet/scripts/train_awa2_resnet50.sh"
echo "  sbatch ${REPO}/MCPNet/scripts/train_awa2_inceptionv3.sh"
echo "  sbatch ${REPO}/MCPNet/scripts/train_awa2_convnext.sh"
echo "  sbatch ${REPO}/MCPNet/scripts/train_cub200_resnet50.sh"
echo "  sbatch ${REPO}/MCPNet/scripts/train_cub200_inceptionv3.sh"
echo "  sbatch ${REPO}/MCPNet/scripts/train_cub200_convnext.sh"
