"""
Text Decoders for CLIP Embeddings

Provides pluggable interfaces for generating natural language text from
CLIP embeddings. Supports multiple decoder backends:

- ClipCapDecoder: Uses ClipCap (CLIP prefix + GPT-2)
- SimpleGPT2Decoder: Minimal GPT-2 with learned mapping
- (More decoders can be added)

These decoders enable free-form text generation (not vocabulary matching)
from abstract visual features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Optional
from PIL import Image


class AbstractTextDecoder(ABC):
    """
    Abstract base class for CLIP embedding → text decoders.

    All decoders must implement the decode() method which takes a
    CLIP embedding and generates natural language text.
    """

    @abstractmethod
    def decode(
        self,
        clip_embedding: torch.Tensor,
        num_captions: int = 5,
        max_length: int = 30
    ) -> List[str]:
        """
        Decode CLIP embedding to natural language text.

        Args:
            clip_embedding: (512,) - L2-normalized CLIP embedding
            num_captions: Number of captions to generate
            max_length: Maximum caption length in tokens

        Returns:
            List of generated text strings
        """
        pass


class SimpleGPT2Decoder(AbstractTextDecoder):
    """
    Simple GPT-2 decoder with learned linear mapping from CLIP to GPT-2 space.

    This is a minimal implementation that directly maps CLIP embeddings
    to GPT-2's input space and generates text autoregressively.

    Not as sophisticated as ClipCap but easier to use and doesn't require
    pre-trained ClipCap weights.

    Args:
        device: Computation device
        gpt2_model: GPT-2 model name (default: 'gpt2')

    Example:
        >>> decoder = SimpleGPT2Decoder(device='cuda')
        >>> clip_embedding = torch.randn(512).cuda()
        >>> clip_embedding = F.normalize(clip_embedding, dim=-1)
        >>> captions = decoder.decode(clip_embedding, num_captions=3)
        >>> print(captions[0])
    """

    def __init__(
        self,
        device: str = 'cuda',
        gpt2_model: str = 'gpt2'
    ):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.device = device

        print(f"Loading GPT-2 model: {gpt2_model}...")
        self.gpt2 = GPT2LMHeadModel.from_pretrained(gpt2_model).to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model)
        self.gpt2.eval()

        # Learn a simple mapping: CLIP (512) → GPT-2 embedding space (768)
        self.mapping = nn.Linear(512, 768).to(device)
        nn.init.xavier_normal_(self.mapping.weight)

        print(f"  Loaded SimpleGPT2Decoder on {device}")

    def decode(
        self,
        clip_embedding: torch.Tensor,
        num_captions: int = 5,
        max_length: int = 30
    ) -> List[str]:
        """
        Generate captions from CLIP embedding using GPT-2.

        Args:
            clip_embedding: (512,) - CLIP embedding
            num_captions: Number of captions to generate
            max_length: Maximum caption length

        Returns:
            List of generated caption strings
        """
        # Ensure correct shape
        if clip_embedding.dim() == 1:
            clip_embedding = clip_embedding.unsqueeze(0)  # (1, 512)

        clip_embedding = clip_embedding.to(self.device)

        with torch.no_grad():
            # Map CLIP → GPT-2 embedding space
            prefix_embed = self.mapping(clip_embedding)  # (1, 768)
            prefix_embed = prefix_embed.unsqueeze(1)  # (1, 1, 768)

            captions = []
            for _ in range(num_captions):
                # Generate from prefix embedding
                outputs = self.gpt2.generate(
                    inputs_embeds=prefix_embed,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.eos_token_id
                )

                # Decode tokens to text
                caption = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                captions.append(caption.strip())

        return captions


class ClipCapDecoder(AbstractTextDecoder):
    """
    ClipCap decoder using pretrained CLIP prefix + GPT-2 architecture.

    ClipCap uses a mapping network to convert CLIP embeddings into
    a sequence of prefix embeddings that guide GPT-2 text generation.

    Paper: "ClipCap: CLIP Prefix for Image Captioning"
    GitHub: https://github.com/rmokady/CLIP_prefix_caption

    Args:
        model_path: Path to pretrained ClipCap weights
        device: Computation device
        prefix_length: Number of prefix tokens (default: 10)

    Example:
        >>> decoder = ClipCapDecoder(
        ...     model_path='pretrained_checkpoints/clipcap/model.pt',
        ...     device='cuda'
        ... )
        >>> clip_embedding = torch.randn(512).cuda()
        >>> captions = decoder.decode(clip_embedding, num_captions=5)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = 'cuda',
        prefix_length: int = 10,
        clip_dim: int = 512
    ):
        from transformers import GPT2LMHeadModel, GPT2Tokenizer

        self.device = device
        self.prefix_length = prefix_length
        self.clip_dim = clip_dim

        print(f"Loading ClipCap decoder...")

        # Load GPT-2 (will load fine-tuned weights if available in checkpoint)
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store for later checkpoint loading
        self._model_path = model_path

        # ClipCap MLP Mapper: CLIP (512) → prefix embeddings (prefix_length * 768)
        # Official architecture: 2-layer MLP
        gpt2_embedding_size = self.gpt2.config.n_embd  # 768 for GPT-2

        self.clip_project = self._build_mlp_mapper(
            clip_dim,
            gpt2_embedding_size,
            prefix_length
        ).to(device)

        # Load pretrained weights if provided
        if model_path is not None:
            self._load_clipcap_weights(model_path)
        else:
            print("  Warning: No ClipCap weights provided. Using random initialization.")
            print("  For best results, download pretrained weights:")
            print("  Run: python scripts/download_clipcap_weights.py")

        self.gpt2.eval()
        print(f"  Loaded ClipCapDecoder on {device}")

    def _build_mlp_mapper(self, clip_dim, gpt2_dim, prefix_length):
        """
        Build MLP mapper matching ClipCap official architecture.

        Official ClipCap uses 2-layer MLP:
        - Linear(512 → 512*prefix_length)  [for simplicity]
        - ReLU
        - Linear(512*prefix_length → 768*prefix_length)

        But the actual pretrained model uses:
        - Linear(512 → 3840)
        - ReLU
        - Linear(3840 → 7680)  where 7680 = 10 * 768
        """
        output_dim = prefix_length * gpt2_dim  # 10 * 768 = 7680

        # Match official architecture: 2-layer MLP
        # First layer maps to an intermediate dimension
        hidden_dim = (clip_dim + output_dim) // 2  # (512 + 7680) // 2 = 4096, but model uses 3840

        # For pretrained compatibility, use exact dimensions from checkpoint
        if clip_dim == 512 and output_dim == 7680:
            # Official COCO model dimensions
            hidden_dim = 3840

        layers = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),  # 512 → 3840
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 3840 → 7680
        )

        return layers

    def _load_clipcap_weights(self, model_path: str):
        """Load pretrained ClipCap weights from official checkpoint."""
        try:
            print(f"  Loading weights from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

            # Official ClipCap checkpoint structure:
            # - 'clip_project.model.0.weight': First linear layer
            # - 'clip_project.model.0.bias'
            # - 'clip_project.model.2.weight': Second linear layer (after ReLU at index 1)
            # - 'clip_project.model.2.bias'
            # - 'gpt.*': Fine-tuned GPT-2 model weights

            if not isinstance(checkpoint, dict):
                raise ValueError(f"Expected checkpoint to be dict, got {type(checkpoint)}")

            # 1. Extract and load clip_project weights
            clip_project_weights = {
                k: v for k, v in checkpoint.items()
                if k.startswith('clip_project.')
            }

            if not clip_project_weights:
                raise ValueError("No 'clip_project.*' keys found in checkpoint")

            print(f"  Found {len(clip_project_weights)} clip_project weights")

            # Remove 'clip_project.model.' prefix to match our Sequential structure
            # Official keys: 'clip_project.model.0.weight' → Our keys: '0.weight'
            mapper_state = {}
            for k, v in clip_project_weights.items():
                # Remove 'clip_project.model.' prefix
                if k.startswith('clip_project.model.'):
                    key = k.replace('clip_project.model.', '')
                else:
                    # Fallback: just remove 'clip_project.'
                    key = k.replace('clip_project.', '')
                mapper_state[key] = v

            # Load into clip_project module
            missing_keys, unexpected_keys = self.clip_project.load_state_dict(mapper_state, strict=True)

            if missing_keys:
                raise ValueError(f"Missing keys in clip_project: {missing_keys}")
            if unexpected_keys:
                print(f"  Warning: Unexpected keys: {unexpected_keys}")

            print(f"  Successfully loaded ClipCap mapper weights!")

            # 2. Extract and load GPT-2 weights (fine-tuned)
            gpt_weights = {
                k: v for k, v in checkpoint.items()
                if k.startswith('gpt.')
            }

            if gpt_weights:
                print(f"  Found {len(gpt_weights)} GPT-2 weights (fine-tuned)")

                # Remove 'gpt.' prefix to match GPT2LMHeadModel structure
                # Official keys: 'gpt.transformer.h.0.ln_1.weight' → 'transformer.h.0.ln_1.weight'
                gpt_state = {}
                for k, v in gpt_weights.items():
                    key = k.replace('gpt.', '')
                    gpt_state[key] = v

                # Load into GPT-2 model
                missing, unexpected = self.gpt2.load_state_dict(gpt_state, strict=False)

                if missing:
                    print(f"  Note: {len(missing)} GPT-2 keys not in checkpoint (using pretrained)")
                if unexpected:
                    print(f"  Warning: {len(unexpected)} unexpected GPT-2 keys")

                print(f"  Successfully loaded fine-tuned GPT-2 weights!")
            else:
                print(f"  No GPT-2 weights in checkpoint, using base GPT-2")

        except Exception as e:
            print(f"  Error loading weights from {model_path}:")
            print(f"  {type(e).__name__}: {e}")
            print("  Using random initialization instead.")
            print()
            print("  To download official weights:")
            print("  python scripts/download_clipcap_weights.py")

    def decode(
        self,
        clip_embedding: torch.Tensor,
        num_captions: int = 5,
        max_length: int = 30
    ) -> List[str]:
        """
        Generate captions from CLIP embedding using ClipCap.

        Args:
            clip_embedding: (512,) - CLIP embedding
            num_captions: Number of captions to generate
            max_length: Maximum caption length

        Returns:
            List of generated caption strings
        """
        # Ensure correct shape
        if clip_embedding.dim() == 1:
            clip_embedding = clip_embedding.unsqueeze(0)  # (1, 512)

        clip_embedding = clip_embedding.to(self.device)

        with torch.no_grad():
            # Map CLIP embedding to GPT-2 prefix
            prefix_projections = self.clip_project(clip_embedding)  # (1, prefix_length * 768)
            prefix_projections = prefix_projections.view(
                1, self.prefix_length, 768
            )  # (1, prefix_length, 768)

            captions = []

            # Generate diverse captions with sampling
            # Repeat prefix_projections for batch generation
            batch_prefix = prefix_projections.repeat(num_captions, 1, 1)

            outputs = self.gpt2.generate(
                inputs_embeds=batch_prefix,
                max_length=max_length,
                do_sample=True,
                num_return_sequences=num_captions,
                top_p=0.95,
                temperature=1.0,
                repetition_penalty=1.5,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Decode all captions
            for output in outputs:
                caption = self.tokenizer.decode(output, skip_special_tokens=True)
                captions.append(caption.strip())

        return captions


def get_text_decoder(
    decoder_type: str = 'simple_gpt2',
    device: str = 'cuda',
    **kwargs
) -> AbstractTextDecoder:
    """
    Factory function to create text decoders.

    Args:
        decoder_type: Type of decoder ('simple_gpt2', 'clipcap')
        device: Computation device
        **kwargs: Additional arguments for specific decoders
            - model_path: For ClipCapDecoder
            - gpt2_model: For SimpleGPT2Decoder

    Returns:
        AbstractTextDecoder instance

    Examples:
        >>> # Simple GPT-2 decoder
        >>> decoder = get_text_decoder('simple_gpt2', device='cuda')

        >>> # ClipCap decoder with pretrained weights
        >>> decoder = get_text_decoder(
        ...     'clipcap',
        ...     device='cuda',
        ...     model_path='pretrained_checkpoints/clipcap/model.pt'
        ... )
    """
    decoder_type = decoder_type.lower()

    if decoder_type == 'simple_gpt2':
        gpt2_model = kwargs.get('gpt2_model', 'gpt2')
        return SimpleGPT2Decoder(device=device, gpt2_model=gpt2_model)

    elif decoder_type == 'clipcap':
        model_path = kwargs.get('model_path', None)
        prefix_length = kwargs.get('prefix_length', 10)
        return ClipCapDecoder(
            model_path=model_path,
            device=device,
            prefix_length=prefix_length
        )

    else:
        raise ValueError(
            f"Unknown decoder type: {decoder_type}. "
            f"Supported: 'simple_gpt2', 'clipcap'"
        )
