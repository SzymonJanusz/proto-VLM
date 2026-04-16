#!/bin/bash
#SBATCH --job-name=mcpnet_awa2_r50
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgbcfg-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/mcpnet_awa2_r50_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/mcpnet_awa2_r50_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate MCPNet

cd ~/proto-VLM/MCPNet

SCRATCH=/net/tscratch/people/plgabedychaj

python -m torch.distributed.launch --nproc_per_node=1 --master_port 9573 train.py \
  --index AWA2_resnet50 \
  --model ResNet \
  --basic_model resnet50_relu \
  --parameter_path "${SCRATCH}/pretrained/resnet50.pth" \
  --devices 0 \
  --dataset_name AWA2 \
  --train_dataset_path "${SCRATCH}/awa2/train" \
  --val_dataset_path   "${SCRATCH}/awa2/val" \
  --epoch 100 \
  --optimizer adam \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --lr_scheduler 40 \
  --margin 0.01 \
  --CCD_weight 100 \
  --concept_cha 32 32 32 32 \
  --concept_per_layer 8 16 32 64 \
  --train_batch_size 64 \
  --val_batch_size 64 \
  --train_num_workers 8 \
  --val_num_workers 8 \
  --saved_dir "${SCRATCH}/mcpnet"
