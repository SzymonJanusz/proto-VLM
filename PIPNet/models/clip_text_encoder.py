"""
CLIP text encoder wrapper for hybrid model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import CLIPModel


class CLIPTextEncoder(nn.Module):
    """
    Wrapper around CLIP's pretrained text encoder.

    Args:
        model_name: Hugging Face model name
        freeze: Whether to freeze text encoder weights
        max_length: Maximum text sequence length
    """

    def __init__(
        self,
        model_name="openai/clip-vit-base-patch32",
        freeze=True,
        max_length=77
    ):
        super().__init__()

        self.model_name = model_name
        self.max_length = max_length

        # Load tokenizer and model
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        # self.model = CLIPTextModel.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).text_model

        # self.embedding_dim = self.model.config.hidden_size  # 512 for base CLIP
        self.embedding_dim = self.model.config.hidden_size


        # Freeze if specified
        if freeze:
            self.freeze()

    def freeze(self):
        """Freeze all text encoder parameters"""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    def unfreeze(self):
        """Unfreeze text encoder for fine-tuning"""
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.train()

    def forward(self, text_inputs):
        """
        Encode text to embeddings.

        Args:
            text_inputs: Either:
                - List of strings: ['a photo of a cat', 'a photo of a dog']
                - Already tokenized dict from tokenizer

        Returns:
            text_embeddings: L2-normalized embeddings (B, embedding_dim)
        """
        # Tokenize if input is list of strings
        if isinstance(text_inputs, list):
            inputs = self.tokenizer(
                text_inputs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            # Move to same device as model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = text_inputs

        # Get text embeddings
        outputs = self.model(**inputs)

        # Use pooled output (representation of [CLS] token)
        text_embeddings = outputs.pooler_output  # (B, embedding_dim)

        # L2 normalize (CLIP uses normalized embeddings)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)

        return text_embeddings

    def encode_text(self, text_list):
        """Convenience method for encoding text"""
        return self.forward(text_list)
