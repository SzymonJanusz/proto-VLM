"""
Example usage of ProtoCLIP model.

This script demonstrates:
1. Creating the model
2. Forward pass with images and text
3. Accessing prototype information
4. Computing similarities
"""

import torch
from ProtoPNet.models.hybrid_model import ProtoCLIP


def main():
    print("=" * 60)
    print("ProtoCLIP Example Usage")
    print("=" * 60)

    # 1. Create model
    print("\n1. Creating ProtoCLIP model...")
    model = ProtoCLIP(
        num_prototypes=200,
        image_backbone='resnet50',
        text_model='openai/clip-vit-base-patch32',
        embedding_dim=512,
        freeze_text_encoder=True,
        temperature=0.07,
        pooling_mode='max'
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # 2. Prepare example inputs
    print("\n2. Preparing example inputs...")
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)  # Random images for demo
    texts = [
        'a photo of a cat',
        'a photo of a dog',
        'an image of a bird',
        'a picture of a car'
    ]
    print(f"   Batch size: {batch_size}")
    print(f"   Image shape: {images.shape}")
    print(f"   Text examples: {texts}")

    # 3. Forward pass
    print("\n3. Running forward pass...")
    with torch.no_grad():  # No gradients for inference
        model.eval()

        # Basic forward pass
        logits_per_image, logits_per_text, image_emb, text_emb = model(images, texts)

        print(f"   Image embeddings shape: {image_emb.shape}")
        print(f"   Text embeddings shape: {text_emb.shape}")
        print(f"   Logits shape: {logits_per_image.shape}")

    # 4. Check embeddings are normalized
    print("\n4. Verifying embeddings are L2-normalized...")
    image_norms = image_emb.norm(dim=-1)
    text_norms = text_emb.norm(dim=-1)
    print(f"   Image embedding norms: {image_norms}")
    print(f"   Text embedding norms: {text_norms}")
    print(f"   ✓ All norms should be ~1.0")

    # 5. Compute similarities
    print("\n5. Computing image-text similarities...")
    print("   Similarity matrix (image × text):")
    print(f"   {logits_per_image}")
    print(f"\n   Diagonal elements (matching pairs): {logits_per_image.diagonal()}")

    # 6. Get prototype information
    print("\n6. Accessing prototype information...")
    prototypes = model.get_prototypes()
    print(f"   Prototype vectors shape: {prototypes.shape}")
    print(f"   Number of prototypes: {model.image_encoder.num_prototypes}")
    print(f"   Prototype dimension: {model.image_encoder.feature_dim}")

    # 7. Forward pass with prototype similarities
    print("\n7. Getting prototype activation maps...")
    with torch.no_grad():
        logits_i, logits_t, img_emb, txt_emb, proto_sims = model(
            images, texts, return_similarities=True
        )
        print(f"   Prototype similarity maps shape: {proto_sims.shape}")
        print(f"   Shape breakdown: (batch={proto_sims.shape[0]}, "
              f"prototypes={proto_sims.shape[1]}, "
              f"height={proto_sims.shape[2]}, width={proto_sims.shape[3]})")

        # Get max activation per prototype
        pooled_sims = model.image_encoder.pooling(proto_sims)
        print(f"\n   Pooled similarities shape: {pooled_sims.shape}")
        print(f"   Max activation per prototype (first image):")
        print(f"   {pooled_sims[0, :10]}... (showing first 10)")

    # 8. Model architecture summary
    print("\n8. Model Architecture Summary:")
    print("   " + "=" * 56)
    print("   IMAGE ENCODER:")
    print("     • ResNet-50 backbone (layer3)")
    print(f"     • {model.image_encoder.num_prototypes} learnable prototypes")
    print(f"     • Weighted pooling ({model.image_encoder.pooling.pooling_mode})")
    print("     • Projection head (200 → 1024 → 512)")
    print("     • L2 normalization")
    print("\n   TEXT ENCODER:")
    print("     • CLIP text encoder (frozen)")
    print("     • L2 normalization")
    print("\n   TRAINING:")
    print("     • Contrastive loss (InfoNCE)")
    print("     • Temperature parameter (learnable)")
    print("   " + "=" * 56)

    print("\n✓ Example completed successfully!")
    print("\nNext steps:")
    print("  1. Prepare your image-text dataset")
    print("  2. Implement data loaders")
    print("  3. Run 3-stage training (warmup → projection → fine-tuning)")
    print("  4. Evaluate on retrieval tasks")
    print("  5. Visualize prototype activations")


if __name__ == '__main__':
    main()
