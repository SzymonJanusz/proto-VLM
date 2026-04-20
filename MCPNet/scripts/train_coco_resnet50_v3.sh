#!/bin/bash
#SBATCH --job-name=mcpnet_coco_r50v3
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/mcpnet_coco_r50v3_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/mcpnet_coco_r50v3_%j.err

# ResNet50 4-stage channels: 256, 512, 1024, 2048
#   concept_cha=8 (halved from v2's 16) → maximum prototype count
#   concept_per_layer: 256//8=32, 512//8=64, 1024//8=128, 2048//8=256

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

SCRATCH=/net/tscratch/people/plgabedychaj
PYTHON="${SCRATCH}/conda_envs/MCPNet/bin/python"

cd ~/proto-VLM/MCPNet

"${PYTHON}" -m pip install wandb -q

"${PYTHON}" -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 9570 train.py \
  --index COCO_resnet50_v3 \
  --model ResNet \
  --basic_model resnet50_relu \
  --parameter_path "${SCRATCH}/pretrained/resnet50.pth" \
  --devices 0 \
  --dataset_name COCO \
  --train_dataset_path "${SCRATCH}/coco_dataset/train" \
  --val_dataset_path   "${SCRATCH}/coco_dataset/val" \
  --epoch 100 \
  --optimizer adam \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --lr_scheduler 40 \
  --margin 0.01 \
  --CCD_weight 100 \
  --concept_cha 8 8 8 8 \
  --concept_per_layer 32 64 128 256 \
  --train_batch_size 64 \
  --val_batch_size 64 \
  --train_num_workers 8 \
  --val_num_workers 8 \
  --saved_dir "${SCRATCH}/mcpnet" \
  --wandb \
  --wandb_project MCPNet
