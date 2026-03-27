"""
Generate up to 5 text descriptions per Caltech-101 image using Qwen2.5-VL-3B-Instruct.

Requirements:
    venv/Scripts/pip install qwen_vl_utils

Usage:
    # Smoke-test on a few random images
    python scripts/describe_caltech101.py --split all --sample 5 --num_descriptions 1 --output_file descriptions/test.json

    # Full run with resume support
    python scripts/describe_caltech101.py --split all --resume
"""

import argparse
import json
import random
import sys
import traceback
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 5 diverse prompts — generates varied descriptions per image
# ---------------------------------------------------------------------------
DESCRIPTION_PROMPTS = [
    "Describe this image in detail.",
    "What objects and characteristics do you see in this image?",
    "Provide a detailed visual description of what is shown in this image.",
    "What is the main subject of this image and what are its key visual features?",
    "Describe the visual content, colors, and notable details of this image.",
]


def load_model(model_path: str):
    """Load Qwen2.5-VL processor and model."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    print(f"Loading processor from: {model_path}")
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"

    print(f"Loading model from: {model_path}  (this may take a few minutes)")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    print(f"Using attention implementation: {attn_impl}")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=attn_impl,
    )
    model.eval()
    print(f"Model on: {next(model.parameters()).device}")
    return processor, model


def describe_images_batch(pil_images: list, prompt: str, processor, model) -> list:
    """Generate one description per image for a batch of images."""
    from qwen_vl_utils import process_vision_info

    messages_batch = [
        [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ]}]
        for img in pil_images
    ]

    texts = [
        processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_batch
    ]

    all_image_inputs = []
    for msgs in messages_batch:
        imgs, _ = process_vision_info(msgs)
        all_image_inputs.extend(imgs)

    inputs = processor(
        text=texts, images=all_image_inputs, padding=True, return_tensors="pt"
    ).to(next(model.parameters()).device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.7,
            max_new_tokens=256,
        )

    results = []
    for i in range(len(pil_images)):
        trimmed = output_ids[i][len(inputs.input_ids[i]):]
        results.append(processor.decode(trimmed, skip_special_tokens=True).strip())
    return results


def collect_image_paths(caltech_root: Path, splits: list) -> list:
    """Return sorted list of image paths for the given splits."""
    paths = []
    for split in splits:
        split_dir = caltech_root / split
        if not split_dir.exists():
            print(f"Warning: split directory not found: {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(img_path)
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate image descriptions for Caltech-101 using Qwen2.5-VL-3B-Instruct"
    )
    parser.add_argument("--caltech_root", type=str, default="caltech101",
                        help="Path to caltech101 directory (default: caltech101)")
    parser.add_argument("--split", type=str, default="all",
                        choices=["train", "val", "test", "all"],
                        help="Which split(s) to process (default: all)")
    parser.add_argument("--output_file", type=str,
                        default="descriptions/caltech101_descriptions.json",
                        help="Output JSON file path")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="HuggingFace model ID or local checkpoint path")
    parser.add_argument("--num_descriptions", type=int, default=5,
                        choices=range(1, 6), metavar="1-5",
                        help="Number of descriptions per image (default: 5)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip images already present in output file")
    parser.add_argument("--sample", type=int, default=None, metavar="N",
                        help="Randomly sample N images instead of processing all")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for --sample (default: 42)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of images to process in parallel (default: 8)")
    args = parser.parse_args()

    caltech_root = Path(args.caltech_root)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    image_paths = collect_image_paths(caltech_root, splits)
    print(f"\nFound {len(image_paths)} images across split(s): {splits}")

    # Load existing results if resuming
    results: dict = {}
    if args.resume and output_file.exists():
        for enc in ("utf-8", "cp1250", "latin-1"):
            try:
                with open(output_file, "r", encoding=enc) as f:
                    results = json.load(f)
                break
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
        else:
            print(f"Warning: could not parse {output_file} — starting fresh")
        print(f"Resuming: {len(results)} images already processed")

    prompts_to_use = DESCRIPTION_PROMPTS[: args.num_descriptions]
    todo = [p for p in image_paths if str(p) not in results]
    if args.sample is not None:
        rng = random.Random(args.seed)
        todo = rng.sample(todo, min(args.sample, len(todo)))
        print(f"Randomly sampled {len(todo)} images (seed={args.seed})")
    print(f"Images to process: {len(todo)}")

    if not todo:
        print("Nothing to do.")
        return

    processor, model = load_model(args.model_path)

    for batch_start in tqdm(range(0, len(todo), args.batch_size), desc="Generating descriptions", unit="batch"):
        batch_paths = todo[batch_start:batch_start + args.batch_size]

        pil_images = []
        valid_paths = []
        for img_path in batch_paths:
            try:
                pil_images.append(Image.open(img_path).convert("RGB"))
                valid_paths.append(img_path)
            except Exception as e:
                print(f"  Skip {img_path}: {e}")

        if not pil_images:
            continue

        batch_descriptions = [[] for _ in valid_paths]
        for prompt in prompts_to_use:
            try:
                descs = describe_images_batch(pil_images, prompt, processor, model)
                for i, desc in enumerate(descs):
                    batch_descriptions[i].append(desc)
            except Exception as e:
                print(f"  Error on batch prompt: {e}")
                traceback.print_exc()
                for i in range(len(pil_images)):
                    batch_descriptions[i].append("")
                break

        for img_path, descriptions in zip(valid_paths, batch_descriptions):
            results[str(img_path)] = descriptions

        # Save after every batch (crash-safe)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(results)} images saved to {output_file}")


if __name__ == "__main__":
    main()
