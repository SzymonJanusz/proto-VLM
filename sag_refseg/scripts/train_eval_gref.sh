#!/usr/bin/env bash
# Train and evaluate sag_refseg on Gref.
# Usage (from repo root): bash sag_refseg/scripts/train_eval_gref.sh [extra train args]
set -euo pipefail

python sag_refseg/train.py \
    --config sag_refseg/configs/gref.json \
    --data_root data/refcoco \
    --out_dir checkpoints/sag_refseg_Gref \
    "$@"

for split in val; do
    python sag_refseg/evaluate.py \
        --config sag_refseg/configs/gref.json \
        --data_root data/refcoco \
        --data_split "$split" \
        --ckpt checkpoints/sag_refseg_Gref/Gref/best.pth
done
