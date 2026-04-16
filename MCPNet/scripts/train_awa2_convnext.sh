#!/bin/bash
#SBATCH --job-name=mcpnet_awa2_cnx
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/mcpnet_awa2_cnx_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/mcpnet_awa2_cnx_%j.err

# ConvNeXt-Small stage outputs (input 224x224):
#   l1: stage 0 → 96ch  → 96//16 = 6 concepts
#   l2: stage 1 → 192ch → 192//16 = 12 concepts
#   l3: stage 2 → 384ch → 384//16 = 24 concepts
#   l4: stage 3 → 768ch → 768//16 = 48 concepts

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

module load Miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /net/tscratch/people/plgabedychaj/conda_envs/MCPNet

cd ~/proto-VLM/MCPNet

SCRATCH=/net/tscratch/people/plgabedychaj

python -m torch.distributed.launch --nproc_per_node=1 --master_port 9575 train.py \
  --index AWA2_convnext_small \
  --model convnext \
  --basic_model convnext_small \
  --parameter_path "${SCRATCH}/pretrained/convnext_small.pth" \
  --devices 0 \
  --dataset_name AWA2 \
  --train_dataset_path "${SCRATCH}/awa2/train" \
  --val_dataset_path   "${SCRATCH}/awa2/val" \
  --epoch 100 \
  --optimizer adam \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --lr_scheduler 40 \
  --margin 0.05 \
  --CCD_weight 100 \
  --concept_cha 16 16 16 16 \
  --concept_per_layer 6 12 24 48 \
  --train_batch_size 64 \
  --val_batch_size 64 \
  --train_num_workers 8 \
  --val_num_workers 8 \
  --saved_dir "${SCRATCH}/mcpnet"
