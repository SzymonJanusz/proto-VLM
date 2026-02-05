"""
Vision-Language Model (VLM) Based Prototype Interpretation

Uses multimodal VLMs to generate natural language descriptions of prototype patches
directly from visual features, without relying on prefabricated label vocabularies.

This module provides a flexible abstraction layer for different VLM backends:
- BLIP: Fast, good quality, balanced choice
- GIT: Very fast, simple captions
- LLaVA: High quality, complex reasoning (requires more memory)
"""

import torch
from PIL import Image
from typing import Optional, Dict
from abc import ABC, abstractmethod


class VLMInterpreter(ABC):
    """Abstract base class for VLM-based prototype interpretation."""

    @abstractmethod
    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50
    ) -> str:
        """
        Generate a caption for an image.

        Args:
            image: PIL Image to caption
            max_length: Maximum caption length in tokens

        Returns:
            Generated caption
        """
        pass

    @abstractmethod
    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100
    ) -> str:
        """
        Answer a question about an image.

        Args:
            image: PIL Image
            question: Text question about the image
            max_length: Maximum answer length in tokens

        Returns:
            Generated answer
        """
        pass


class BLIPInterpreter(VLMInterpreter):
    """
    BLIP-based interpreter using Salesforce's BLIP model.
    Good balance of quality and speed, runs locally.

    Memory: ~5GB VRAM
    Speed: ~1-2 seconds per image
    """

    def __init__(
        self,
        model_name: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda"
    ):
        """
        Initialize BLIP interpreter.

        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda' or 'cpu')
        """
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
        except ImportError:
            raise ImportError(
                "BLIP requires transformers. Install with: pip install transformers"
            )

        self.device = device
        self.model_name = model_name

        print(f"Loading BLIP model: {model_name}...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.model.eval()

        print(f"  Loaded BLIP model on {device}")

    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50
    ) -> str:
        """
        Generate caption for image using BLIP.

        Args:
            image: PIL Image
            max_length: Maximum caption length

        Returns:
            Generated caption
        """
        inputs = self.processor(image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100
    ) -> str:
        """
        Answer a question about the image using BLIP.

        BLIP supports conditional captioning by providing text as context.

        Args:
            image: PIL Image
            question: Question about the image
            max_length: Maximum answer length

        Returns:
            Generated answer
        """
        # BLIP can do conditional generation with text prompt
        inputs = self.processor(image, question, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=3
            )

        answer = self.processor.decode(out[0], skip_special_tokens=True)
        return answer.strip()


class GITInterpreter(VLMInterpreter):
    """
    GIT (Generative Image-to-Text) based interpreter.
    Microsoft's image captioning model, very fast but simpler outputs.

    Memory: ~3GB VRAM
    Speed: ~0.5-1 second per image
    """

    def __init__(
        self,
        model_name: str = "microsoft/git-large-coco",
        device: str = "cuda"
    ):
        """
        Initialize GIT interpreter.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
        """
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
        except ImportError:
            raise ImportError(
                "GIT requires transformers. Install with: pip install transformers"
            )

        self.device = device
        self.model_name = model_name

        print(f"Loading GIT model: {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()

        print(f"  Loaded GIT model on {device}")

    def generate_caption(
        self,
        image: Image.Image,
        max_length: int = 50
    ) -> str:
        """
        Generate caption for image using GIT.

        Args:
            image: PIL Image
            max_length: Maximum caption length

        Returns:
            Generated caption
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_length=max_length,
                num_beams=3
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption.strip()

    def answer_question(
        self,
        image: Image.Image,
        question: str,
        max_length: int = 100
    ) -> str:
        """
        Answer a question about the image.

        Note: GIT doesn't natively support question answering,
        so we just return a caption with the question as context.

        Args:
            image: PIL Image
            question: Question about the image
            max_length: Maximum answer length

        Returns:
            Generated answer (caption)
        """
        # GIT doesn't support conditional generation well,
        # so we just return a caption
        return self.generate_caption(image, max_length)


class LLaVAInterpreter(VLMInterpreter):
    """
    LLaVA-based interpreter using open-source vision-language model.
    Highest quality, can follow complex instructions and reason about images.

    Memory: ~14GB VRAM (or ~7GB with 8-bit quantization)
    Speed: ~3-5 seconds per image

    Note: This is a placeholder stub. Full implementation requires
    LLaVA dependencies and more complex setup.
    """

    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        device: str = "cuda",
        load_in_8bit: bool = False
    ):
        """
        Initialize LLaVA interpreter.

        Args:
            model_name: HuggingFace model name
            device: Device to run on
            load_in_8bit: Whether to load in 8-bit (saves memory)
        """
        raise NotImplementedError(
            "LLaVA interpreter is not yet implemented. "
            "Use 'blip' or 'git' backends instead. "
            "To add LLaVA support, implement this class following the BLIPInterpreter pattern."
        )

    def generate_caption(self, image: Image.Image, max_length: int = 50) -> str:
        raise NotImplementedError("LLaVA interpreter not implemented")

    def answer_question(self, image: Image.Image, question: str, max_length: int = 100) -> str:
        raise NotImplementedError("LLaVA interpreter not implemented")


def get_vlm_interpreter(
    backend: str = "blip",
    device: str = "cuda",
    **kwargs
) -> VLMInterpreter:
    """
    Factory function to create VLM interpreter.

    Args:
        backend: VLM backend ('blip', 'git', or 'llava')
        device: Device for local models ('cuda' or 'cpu')
        **kwargs: Additional arguments for specific backends
            - model_name: Custom model name
            - load_in_8bit: For LLaVA, load in 8-bit mode

    Returns:
        VLMInterpreter instance

    Examples:
        >>> # Use BLIP (default, recommended)
        >>> vlm = get_vlm_interpreter('blip')

        >>> # Use GIT (faster, simpler)
        >>> vlm = get_vlm_interpreter('git')

        >>> # Use custom BLIP model
        >>> vlm = get_vlm_interpreter('blip', model_name='Salesforce/blip-image-captioning-base')

        >>> # Use CPU instead of GPU
        >>> vlm = get_vlm_interpreter('blip', device='cpu')
    """
    backend = backend.lower()

    if backend == "blip":
        model_name = kwargs.get('model_name', 'Salesforce/blip-image-captioning-large')
        return BLIPInterpreter(model_name=model_name, device=device)

    elif backend == "git":
        model_name = kwargs.get('model_name', 'microsoft/git-large-coco')
        return GITInterpreter(model_name=model_name, device=device)

    elif backend == "llava":
        model_name = kwargs.get('model_name', 'llava-hf/llava-1.5-7b-hf')
        load_in_8bit = kwargs.get('load_in_8bit', False)
        return LLaVAInterpreter(
            model_name=model_name,
            device=device,
            load_in_8bit=load_in_8bit
        )

    else:
        raise ValueError(
            f"Unknown VLM backend: {backend}. "
            f"Supported backends: 'blip', 'git', 'llava'"
        )
