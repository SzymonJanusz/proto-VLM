"""
Hybrid ProtoPNet-CLIP model combining prototype-based image encoder
with CLIP's text encoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .protopnet_encoder import ProtoPNetEncoder
from .clip_text_encoder import CLIPTextEncoder


class ProtoCLIP(nn.Module):
    """
    Hybrid model: ProtoPNet image encoder + CLIP text encoder.

    Trained with contrastive learning on image-text pairs.

    Args:
        num_prototypes: Number of prototypes for image encoder
        image_backbone: ResNet architecture for image encoder
        text_model: CLIP text model name
        embedding_dim: Shared embedding dimension (512 for CLIP)
        freeze_text_encoder: Whether to freeze text encoder initially
        temperature: Temperature parameter for contrastive loss
    """

    def __init__(
        self,
        num_prototypes=200,
        image_backbone='resnet50',
        text_model="openai/clip-vit-base-patch32",
        embedding_dim=512,
        freeze_text_encoder=True,
        temperature=0.07,
        pooling_mode='max'
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # Image encoder (ProtoPNet-based)
        self.image_encoder = ProtoPNetEncoder(
            num_prototypes=num_prototypes,
            backbone_arch=image_backbone,
            embedding_dim=embedding_dim,
            pooling_mode=pooling_mode,
            pretrained_backbone=True
        )

        # Text encoder (CLIP)
        self.text_encoder = CLIPTextEncoder(
            model_name=text_model,
            freeze=freeze_text_encoder
        )

        # Learnable temperature (can be fixed or learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / temperature))

    def encode_image(self, images, return_similarities=False):
        """Encode images to embeddings"""
        return self.image_encoder(images, return_similarities=return_similarities)

    def encode_text(self, text_inputs):
        """Encode text to embeddings"""
        return self.text_encoder(text_inputs)

    def forward(self, images, text_inputs, return_similarities=False):
        """
        Forward pass for contrastive learning.

        Args:
            images: Batch of images (B, 3, 224, 224)
            text_inputs: Batch of text descriptions (list of B strings)
            return_similarities: Whether to return prototype similarity maps

        Returns:
            logits_per_image: Similarity matrix (B, B) from image perspective
            logits_per_text: Similarity matrix (B, B) from text perspective
            image_embeddings: Image embeddings (B, embedding_dim)
            text_embeddings: Text embeddings (B, embedding_dim)
            similarities: (optional) Prototype similarity maps
        """
        # Encode images and text
        if return_similarities:
            image_embeddings, proto_similarities = self.encode_image(
                images, return_similarities=True
            )
        else:
            image_embeddings = self.encode_image(images)
            proto_similarities = None

        text_embeddings = self.encode_text(text_inputs)

        # Compute cosine similarity (already normalized)
        # logits = embeddings @ embeddings.T * temperature
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeddings @ text_embeddings.t()
        logits_per_text = logits_per_image.t()

        if return_similarities:
            return logits_per_image, logits_per_text, image_embeddings, text_embeddings, proto_similarities
        else:
            return logits_per_image, logits_per_text, image_embeddings, text_embeddings

    def get_prototypes(self):
        """Get prototype vectors"""
        return self.image_encoder.get_prototypes()

    def set_prototypes(self, new_prototypes):
        """Update prototype vectors"""
        self.image_encoder.set_prototypes(new_prototypes)
