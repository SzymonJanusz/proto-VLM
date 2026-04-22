#!/usr/bin/env python3
"""
CTRL-O inference on RefCOCO / RefCOCO+ val/testA/testB splits.

Metrics reported per split:
  - Acc@0.5 IoU (mask): slot mask vs GT COCO instance mask
  - Acc@0.5 IoU (box):  bbox of slot mask foreground vs GT bounding box

Usage:
  python inference_refer.py \
    --data_root /net/tscratch/people/plgabedychaj/ctrl-o/data \
    --dataset refcoco \
    --splitBy unc \
    --splits val testA testB \
    --checkpoint /net/tscratch/people/plgabedychaj/ctrl-o/pretrained_models/ctrlo/pretrained_model.ckpt \
    --config /net/tscratch/people/plgabedychaj/ctrl-o/pretrained_models/ctrlo/config.yaml \
    --image_root /net/tscratch/people/plgabedychaj/ctrl-o/data/images/mscoco/images/train2014 \
    --output_dir /net/tscratch/people/plgabedychaj/ctrl-o/results_refer
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Path setup: add CTRL-O and REFER repos to sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CTRLO_DIR = os.path.join(SCRIPT_DIR, "CTRL-O")
REFER_DIR = os.path.join(SCRIPT_DIR, "refer")

# Also support WORKDIR-based layout (when running from scratch dir)
_WORKDIR = os.environ.get("WORKDIR", "")
if _WORKDIR:
    CTRLO_DIR = os.path.join(_WORKDIR, "CTRL-O")
    REFER_DIR = os.path.join(_WORKDIR, "refer")

sys.path.insert(0, CTRLO_DIR)
sys.path.insert(0, REFER_DIR)

from refer import REFER  # noqa: E402


# ---------------------------------------------------------------------------
# Model wrapper — mirrors FeatureExtractor from ocl/cli/inference.py
# but avoids module-level side effects (LLM2Vec loads only when needed).
# ---------------------------------------------------------------------------
class CtrloModel:
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = device
        self._transform = transforms.Compose([
            transforms.Resize(
                (224, 224),
                interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._load_model(checkpoint_path, config_path)
        self._load_encoder()

    def _load_model(self, checkpoint_path: str, config_path: str):
        from ocl.cli import train as ocl_train
        from omegaconf import OmegaConf, open_dict
        config = OmegaConf.load(config_path)
        # Strip metrics that depend on the 'routed' package — not needed for inference
        with open_dict(config):
            for key in ("training_metrics", "evaluation_metrics", "losses"):
                if key in config:
                    config[key] = {}
        self.model = ocl_train.build_model_from_config(config, checkpoint_path)
        self.model = self.model.to(self.device)
        self.model.eval()

    def _load_encoder(self):
        from llm2vec import LLM2Vec
        self.l2v = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path=(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
            ),
            device_map=self.device,
            torch_dtype=torch.bfloat16,
        )

    @torch.no_grad()
    def forward(self, images: list, prompts: list) -> dict:
        """
        Args:
            images:  list of PIL Images, length B
            prompts: list of length-7 string lists, length B
                     e.g. [["a red car", "other", ...], ...]
        Returns:
            model output dict
        """
        imgs_t = torch.stack([self._transform(img) for img in images]).to(self.device)
        name_embeddings = torch.stack(
            [self.l2v.encode(p) for p in prompts]
        ).to(self.device)
        bsz = imgs_t.shape[0]

        contrastive_mask = torch.stack([
            torch.tensor([int(p != "other") for p in prompt], dtype=torch.float32)
            for prompt in prompts
        ]).to(self.device)

        n_slots = 7
        inputs = {
            "image": imgs_t,
            "bbox_centroids": torch.full(
                (bsz, n_slots, 2), -1.0, dtype=torch.float32
            ).to(self.device),
            "contrastive_loss_mask": contrastive_mask,
            "name_embedding": name_embeddings,
            "instance_bbox": torch.full(
                (bsz, n_slots, 4), -1.0, dtype=torch.float32
            ).to(self.device),
            "batch_size": bsz,
        }
        return self.model(inputs)


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def mask_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred = pred_mask > 0.5
    gt = gt_mask.astype(bool)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union + 1e-8)


def box_iou(pred_mask: np.ndarray, gt_box: list) -> float:
    """GT box format: [x, y, w, h] (COCO style)."""
    rows = np.any(pred_mask > 0.5, axis=1)
    cols = np.any(pred_mask > 0.5, axis=0)
    if not rows.any() or not cols.any():
        return 0.0
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    px, py, pw, ph = int(c0), int(r0), int(c1 - c0), int(r1 - r0)

    gx, gy, gw, gh = [int(v) for v in gt_box]
    ix = max(px, gx)
    iy = max(py, gy)
    ix2 = min(px + pw, gx + gw)
    iy2 = min(py + ph, gy + gh)
    inter = max(0, ix2 - ix) * max(0, iy2 - iy)
    union = pw * ph + gw * gh - inter
    return float(inter) / float(union + 1e-8)


# ---------------------------------------------------------------------------
# Per-split evaluation
# ---------------------------------------------------------------------------

def find_best_slot_idx(outputs, query_emb: np.ndarray) -> int:
    """Find the slot with highest cosine similarity to the query text embedding.

    Uses projector_slots (language-aligned slot embeddings) from the model output.
    Falls back to slot 0 if the key is not present.
    """
    if "projector_slots" not in outputs:
        return 0
    slot_embs = outputs["projector_slots"][0].cpu().float().numpy()  # [n_slots, dim]
    q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
    sims = [float(np.dot(s / (np.linalg.norm(s) + 1e-8), q)) for s in slot_embs]
    return int(np.argmax(sims))


def evaluate_split(
    refer: REFER,
    split: str,
    model: CtrloModel,
    image_root: str,
    limit: int = 0,
) -> tuple:
    """Returns (per_ref_results, summary_dict)."""
    ref_ids = refer.getRefIds(split=split)
    if limit > 0:
        ref_ids = ref_ids[:limit]

    per_ref = []
    correct_mask = correct_box = total = missing = 0
    sum_mask_iou = sum_box_iou = 0.0

    for ref_id in tqdm(ref_ids, desc=f"  [{split}]"):
        ref = refer.loadRefs(ref_ids=[ref_id])[0]
        img_meta = refer.loadImgs(image_ids=[ref["image_id"]])[0]
        img_path = os.path.join(image_root, img_meta["file_name"])

        if not os.path.exists(img_path):
            missing += 1
            continue

        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        gt_info = refer.getMask(ref)
        gt_mask_orig = gt_info["mask"]          # (H, W) binary at original resolution
        gt_box = refer.getRefBox(ref_id)        # [x, y, w, h]

        best_miou = best_biou = 0.0

        for sent in ref["sentences"]:
            expr = sent["raw"]
            # Query goes in slot 0; remaining 6 slots are "other"
            prompt = [expr] + ["other"] * 6

            try:
                outputs = model.forward([image], [prompt])
            except Exception as e:
                print(f"  WARNING: forward failed for ref_id={ref_id}: {e}")
                continue

            # Find best slot via cosine similarity to query embedding
            query_emb = model.l2v.encode([expr])[0].cpu().float().numpy()
            slot_idx = find_best_slot_idx(outputs, query_emb)
            masks = outputs["object_decoder"].masks_as_image  # [B, n_slots, H, W]
            slot_mask_224 = masks[0][slot_idx].cpu().numpy()

            slot_mask_orig = cv2.resize(
                slot_mask_224, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
            )

            best_miou = max(best_miou, mask_iou(slot_mask_orig, gt_mask_orig))
            best_biou = max(best_biou, box_iou(slot_mask_orig, gt_box))

        correct_mask += int(best_miou >= 0.5)
        correct_box += int(best_biou >= 0.5)
        sum_mask_iou += best_miou
        sum_box_iou += best_biou
        total += 1

        per_ref.append({
            "ref_id": ref_id,
            "split": split,
            "mask_iou": round(best_miou, 4),
            "box_iou": round(best_biou, 4),
        })

    if missing > 0:
        print(f"  WARNING: {missing}/{len(ref_ids)} refs skipped — image not found. "
              f"Check --image_root or re-run download_data.sh.")

    acc_mask = correct_mask / total if total else 0.0
    acc_box  = correct_box  / total if total else 0.0
    miou_mask = sum_mask_iou / total if total else 0.0
    miou_box  = sum_box_iou  / total if total else 0.0

    summary = {
        "acc_mask_0.5": round(acc_mask, 4),
        "acc_box_0.5":  round(acc_box, 4),
        "miou_mask":    round(miou_mask, 4),
        "miou_box":     round(miou_box, 4),
        "n": total,
        "n_skipped": missing,
    }
    print(
        f"  [{split}] Acc@0.5 mask={acc_mask:.4f}  box={acc_box:.4f} | "
        f"mIoU mask={miou_mask:.4f}  box={miou_box:.4f}  (n={total})"
    )
    return per_ref, summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Run CTRL-O inference on REFER dataset splits."
    )
    p.add_argument("--data_root", required=True,
                   help="Root dir containing refcoco/ and refcoco+/ folders")
    p.add_argument("--dataset", default="refcoco",
                   choices=["refcoco", "refcoco+"],
                   help="Which REFER dataset to evaluate")
    p.add_argument("--splitBy", default="unc",
                   help="Split provider (unc or google)")
    p.add_argument("--splits", nargs="+", default=["val", "testA", "testB"],
                   help="Which splits to run (val testA testB)")
    p.add_argument("--checkpoint", required=True,
                   help="Path to CTRL-O .ckpt file")
    p.add_argument("--config", required=True,
                   help="Path to CTRL-O config.yaml")
    p.add_argument("--image_root", required=True,
                   help="Path to COCO train2014 image directory")
    p.add_argument("--output_dir", default="./results_refer",
                   help="Directory for output JSON files")
    p.add_argument("--device", default="cuda",
                   help="Device: cuda or cpu")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of refs per split (0 = all, useful for smoke tests)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"==> Loading CTRL-O model from {args.checkpoint}")
    model = CtrloModel(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
    )

    print(f"==> Loading REFER dataset: {args.dataset} / {args.splitBy}")
    refer = REFER(args.data_root, dataset=args.dataset, splitBy=args.splitBy)

    all_per_ref = []
    summary = {}

    for split in args.splits:
        print(f"\n=== {args.dataset} / {split} ===")
        per_ref, split_summary = evaluate_split(
            refer, split, model, args.image_root, limit=args.limit
        )
        all_per_ref.extend(per_ref)
        summary[split] = split_summary

    out_file = os.path.join(args.output_dir, f"{args.dataset}_metrics.json")
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "per_ref": all_per_ref}, f, indent=2)
    print(f"\n==> Results saved to {out_file}")

    # Print final table
    print("\n--- Summary ---")
    print(f"{'Split':<10} {'Acc@0.5 mask':>14} {'Acc@0.5 box':>12} {'mIoU mask':>10} {'mIoU box':>9} {'N':>8}")
    print("-" * 70)
    for split, m in summary.items():
        print(f"{split:<10} {m['acc_mask_0.5']:>14.4f} {m['acc_box_0.5']:>12.4f} "
              f"{m['miou_mask']:>10.4f} {m['miou_box']:>9.4f} {m['n']:>8}")


if __name__ == "__main__":
    main()
