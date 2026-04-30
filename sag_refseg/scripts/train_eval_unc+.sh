#!/usr/bin/env bash
# Train and evaluate sag_refseg on RefCOCO+ (unc+ split).
# Usage (from repo root): bash sag_refseg/scripts/train_eval_unc+.sh [extra train args]
set -euo pipefail

python sag_refseg/train.py \
    --config "sag_refseg/configs/unc+.json" \
    --data_root data/refcoco \
    --out_dir checkpoints/sag_refseg_unc+ \
    "$@"

for split in val testA testB; do
    python sag_refseg/evaluate.py \
        --config "sag_refseg/configs/unc+.json" \
        --data_root data/refcoco \
        --data_split "$split" \
        --ckpt "checkpoints/sag_refseg_unc+/unc+/best.pth"
done
