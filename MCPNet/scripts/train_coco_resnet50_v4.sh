#!/bin/bash
#SBATCH --job-name=mcpnet_coco_r50v4
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/mcpnet_coco_r50v4_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/mcpnet_coco_r50v4_%j.err

# ResNet50 4-stage channels: 256, 512, 1024, 2048
#   concept_cha=4 (halved from v3's 8) → minimum channel width
#   concept_per_layer: 256//4=64, 512//4=128, 1024//4=256, 2048//4=512

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

SCRATCH=/net/tscratch/people/plgabedychaj
PYTHON="${SCRATCH}/conda_envs/MCPNet/bin/python"

cd ~/proto-VLM/MCPNet

"${PYTHON}" -m pip install wandb -q

"${PYTHON}" -m torch.distributed.launch --nproc_per_node=1 --master_addr 127.0.0.1 --master_port 9569 train.py \
  --index COCO_resnet50_v4 \
  --model ResNet \
  --basic_model resnet50_relu \
  --devices 0 \
  --parameter_path "${SCRATCH}/pretrained/resnet50.pth" \
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
  --concept_cha 4 4 4 4 \
  --concept_per_layer 64 128 256 512 \
  --train_batch_size 64 \
  --val_batch_size 64 \
  --train_num_workers 8 \
  --val_num_workers 8 \
  --saved_dir "${SCRATCH}/mcpnet" \
  --wandb \
  --wandb_project MCPNet
