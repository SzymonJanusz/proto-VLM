#!/bin/bash
#SBATCH --job-name=mcpnet_cub_r50
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/mcpnet_cub_r50_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/mcpnet_cub_r50_%j.err

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

SCRATCH=/net/tscratch/people/plgabedychaj
PYTHON="${SCRATCH}/conda_envs/MCPNet/bin/python"

cd ~/proto-VLM/MCPNet

"${PYTHON}" -m pip install wandb -q

"${PYTHON}" -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 9576 train.py \
  --index CUB200_resnet50 \
  --model ResNet \
  --basic_model resnet50_relu \
  --parameter_path "${SCRATCH}/pretrained/resnet50.pth" \
  --devices 0 \
  --dataset_name CUB_200_2011 \
  --train_dataset_path "${SCRATCH}/cub200/train" \
  --val_dataset_path   "${SCRATCH}/cub200/val" \
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
  --saved_dir "${SCRATCH}/mcpnet" \
  --wandb \
  --wandb_project MCPNet
