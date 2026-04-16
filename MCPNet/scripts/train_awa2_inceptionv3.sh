#!/bin/bash
#SBATCH --job-name=mcpnet_awa2_inc
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgunhype-gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/net/tscratch/people/plgabedychaj/logs/mcpnet_awa2_inc_%j.out
#SBATCH --error=/net/tscratch/people/plgabedychaj/logs/mcpnet_awa2_inc_%j.err

# InceptionV3 hook points and channel dims (input 299x299):
#   l1: Conv2d_4a_3x3 → 192ch  → 192//32 = 6 concepts
#   l2: Mixed_5d      → 288ch  → 288//32 = 9 concepts
#   l3: Mixed_6e      → 768ch  → 768//32 = 24 concepts
#   l4: Mixed_7c      → 2048ch → 2048//32 = 64 concepts

set -e
mkdir -p /net/tscratch/people/plgabedychaj/logs

SCRATCH=/net/tscratch/people/plgabedychaj
PYTHON="${SCRATCH}/conda_envs/MCPNet/bin/python"

cd ~/proto-VLM/MCPNet

"${PYTHON}" -m pip install wandb -q

"${PYTHON}" -m torch.distributed.launch --nproc_per_node=1 --master_port 9574 train.py \
  --index AWA2_inceptionv3 \
  --model inception_net \
  --basic_model inceptionv3 \
  --parameter_path "${SCRATCH}/pretrained/inception_v3.pth" \
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
  --concept_per_layer 6 9 24 64 \
  --train_batch_size 64 \
  --val_batch_size 64 \
  --train_num_workers 8 \
  --val_num_workers 8 \
  --saved_dir "${SCRATCH}/mcpnet" \
  --wandb \
  --wandb_project MCPNet
