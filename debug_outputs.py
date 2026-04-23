#!/usr/bin/env python3
"""Quick diagnostic: run one forward pass and print all output keys/shapes."""

import os
import sys
import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKDIR = os.environ.get("WORKDIR", "")
CTRLO_DIR = os.path.join(_WORKDIR or SCRIPT_DIR, "CTRL-O")
REFER_DIR = os.path.join(_WORKDIR or SCRIPT_DIR, "refer")
sys.path.insert(0, CTRLO_DIR)
sys.path.insert(0, REFER_DIR)

CHECKPOINT = os.environ.get(
    "CHECKPOINT",
    "/net/tscratch/people/plgabedychaj/ctrl-o/pretrained_models/ctrlo/pretrained_model.ckpt"
)
CONFIG = os.environ.get(
    "CONFIG",
    "/net/tscratch/people/plgabedychaj/ctrl-o/pretrained_models/ctrlo/config.yaml"
)
IMAGE_ROOT = os.environ.get(
    "IMAGE_ROOT",
    "/net/tscratch/people/plgabedychaj/ctrl-o/data/images/mscoco/images/train2014"
)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading model...")
    from ocl.cli import train as ocl_train
    from omegaconf import OmegaConf, open_dict
    config = OmegaConf.load(CONFIG)
    with open_dict(config):
        for key in ("training_metrics", "evaluation_metrics", "losses"):
            if key in config:
                config[key] = {}
    model = ocl_train.build_model_from_config(config, CHECKPOINT)
    model = model.to(device)
    model.eval()

    print("Loading LLM2Vec...")
    from llm2vec import LLM2Vec
    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path=(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse"
        ),
        device_map=device,
        torch_dtype=torch.bfloat16,
    )

    # Use any image that exists
    test_img_path = None
    for fname in os.listdir(IMAGE_ROOT):
        if fname.lower().endswith(".jpg"):
            test_img_path = os.path.join(IMAGE_ROOT, fname)
            break
    if test_img_path is None:
        print("ERROR: no JPEG found in IMAGE_ROOT")
        return

    print(f"Test image: {test_img_path}")
    image = Image.open(test_img_path).convert("RGB")

    expr = "a red car"
    prompt = [expr] + ["other"] * 6

    imgs_t = transform(image).unsqueeze(0).to(device)
    name_embeddings = l2v.encode(prompt).unsqueeze(0).to(device)
    bsz = 1
    n_slots = 7

    contrastive_mask = torch.tensor([[1.0, 0., 0., 0., 0., 0., 0.]], device=device)

    inputs = {
        "image": imgs_t,
        "bbox_centroids": torch.full((bsz, n_slots, 2), -1.0, device=device),
        "contrastive_loss_mask": contrastive_mask,
        "name_embedding": name_embeddings,
        "instance_bbox": torch.full((bsz, n_slots, 4), -1.0, device=device),
        "batch_size": bsz,
    }

    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(inputs)

    print("\n=== Top-level output keys ===")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor {tuple(v.shape)} dtype={v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
        else:
            print(f"  {k}: {type(v).__name__}")

    print("\n=== projector_slots present? ===")
    print("projector_slots" in outputs)
    if "projector_slots" in outputs:
        ps = outputs["projector_slots"]
        print(f"  shape: {tuple(ps.shape)}")
        print(f"  dtype: {ps.dtype}")
        query_emb = l2v.encode([expr])[0].cpu().float().numpy()
        slot_embs = ps[0].cpu().float().numpy()
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        sims = [float(np.dot(s / (np.linalg.norm(s) + 1e-8), q)) for s in slot_embs]
        print(f"  cosine sims to query: {[round(s, 4) for s in sims]}")
        print(f"  argmax: {int(np.argmax(sims))}")

    print("\n=== object_decoder tensor attributes ===")
    od = outputs.get("object_decoder")
    if od is not None:
        for attr in vars(od) if hasattr(od, "__dict__") else []:
            try:
                val = getattr(od, attr)
                if isinstance(val, torch.Tensor):
                    print(f"  .{attr}: Tensor {tuple(val.shape)}")
                elif not callable(val) and val is not None:
                    print(f"  .{attr}: {type(val).__name__} = {val}")
            except Exception:
                pass
        # Also check common named attributes explicitly
        for attr in ("masks", "masks_as_image", "slots", "slot_features",
                     "attn", "attention", "object_features"):
            try:
                val = getattr(od, attr, None)
                if val is not None and attr not in vars(od).get("__dict__", {}):
                    if isinstance(val, torch.Tensor):
                        print(f"  .{attr} (property): Tensor {tuple(val.shape)}")
            except Exception:
                pass

if __name__ == "__main__":
    main()
