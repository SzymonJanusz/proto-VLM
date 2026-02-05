"""
CLIP-based prototype interpretation utilities.

This module provides tools to interpret ProtoPNet prototypes using CLIP's
pre-trained image encoder. No additional training required.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
import pickle


class PrototypeImageGenerator:
    """Creates synthetic images from prototype patches for CLIP interpretation."""

    def __init__(self, projection_info: Dict, image_size: int = 224):
        """
        Initialize the generator with projection information.

        Args:
            projection_info: Dictionary containing projection metadata
                Expected keys: 'image_path', 'patch_h', 'patch_w', etc.
            image_size: Size of output images (default: 224 for CLIP)
        """
        self.projection_info = projection_info
        self.image_size = image_size

        # Extract prototype information
        self.image_paths = projection_info.get('image_path', [])
        self.patch_h = projection_info.get('patch_h', [])
        self.patch_w = projection_info.get('patch_w', [])
        self.num_prototypes = len(self.image_paths)

    @classmethod
    def from_file(cls, projection_info_path: str, image_size: int = 224):
        """
        Load projection info from pickle file.

        Args:
            projection_info_path: Path to projection_info.pkl
            image_size: Size of output images

        Returns:
            PrototypeImageGenerator instance
        """
        with open(projection_info_path, 'rb') as f:
            projection_info = pickle.load(f)
        return cls(projection_info, image_size)

    def extract_patch(
        self,
        prototype_idx: int,
        patch_size: int = 64,
        feature_size: int = 14
    ) -> Image.Image:
        """
        Extract patch from original training image at prototype location.

        Args:
            prototype_idx: Index of the prototype
            patch_size: Desired patch size in pixels (e.g., 64)
            feature_size: Feature map size (default: 14 for 14x14)

        Returns:
            PIL Image of extracted patch
        """
        if prototype_idx >= self.num_prototypes:
            raise ValueError(f"Prototype index {prototype_idx} out of range")

        # Get prototype information
        image_path = self.image_paths[prototype_idx]
        patch_h = self.patch_h[prototype_idx]
        patch_w = self.patch_w[prototype_idx]

        # Load and resize image
        img = Image.open(image_path).convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        # Calculate receptive field center in pixel space
        stride = self.image_size / feature_size  # 224/14 = 16
        center_h = (patch_h + 0.5) * stride
        center_w = (patch_w + 0.5) * stride

        # Extract patch around center
        half_size = patch_size // 2
        top = int(center_h - half_size)
        left = int(center_w - half_size)
        bottom = int(center_h + half_size)
        right = int(center_w + half_size)

        # Handle boundaries
        top = max(0, top)
        left = max(0, left)
        bottom = min(self.image_size, bottom)
        right = min(self.image_size, right)

        # Crop patch
        patch = img.crop((left, top, right, bottom))

        # Resize to exact patch_size if needed
        if patch.size != (patch_size, patch_size):
            patch = patch.resize((patch_size, patch_size), Image.BILINEAR)

        return patch

    def create_synthetic_image(
        self,
        patch: Image.Image,
        background: str = 'noise',
        noise_std: float = 0.1
    ) -> Image.Image:
        """
        Create synthetic image with patch on background.

        Args:
            patch: PIL Image of prototype patch
            background: Background type ('noise', 'gray', 'random', 'average')
            noise_std: Standard deviation for noise background

        Returns:
            PIL Image ready for CLIP encoding
        """
        # Create background
        if background == 'noise':
            # Gaussian noise with ImageNet-like statistics
            mean = np.array([0.485, 0.456, 0.406])
            bg = np.random.normal(mean, noise_std, (self.image_size, self.image_size, 3))
            bg = np.clip(bg, 0, 1)
        elif background == 'gray':
            bg = np.full((self.image_size, self.image_size, 3), 0.5)
        elif background == 'random':
            bg = np.random.rand(self.image_size, self.image_size, 3)
        elif background == 'average':
            # Use ImageNet mean
            bg = np.full((self.image_size, self.image_size, 3), [0.485, 0.456, 0.406])
        else:
            raise ValueError(f"Unknown background type: {background}")

        # Convert to PIL Image
        bg_img = Image.fromarray((bg * 255).astype(np.uint8))

        # Paste patch in center
        patch_size = patch.size[0]
        paste_loc = (
            (self.image_size - patch_size) // 2,
            (self.image_size - patch_size) // 2
        )
        bg_img.paste(patch, paste_loc)

        return bg_img

    def generate_multiple_sizes(
        self,
        prototype_idx: int,
        sizes: List[int] = [32, 64, 128],
        background: str = 'noise'
    ) -> List[Tuple[int, Image.Image]]:
        """
        Generate synthetic images at multiple patch sizes.

        Args:
            prototype_idx: Index of the prototype
            sizes: List of patch sizes to generate
            background: Background type

        Returns:
            List of (patch_size, synthetic_image) tuples
        """
        results = []
        for size in sizes:
            patch = self.extract_patch(prototype_idx, patch_size=size)
            synthetic = self.create_synthetic_image(patch, background=background)
            results.append((size, synthetic))
        return results

    def get_original_image(self, prototype_idx: int) -> Image.Image:
        """
        Get the original training image for a prototype.

        Args:
            prototype_idx: Index of the prototype

        Returns:
            PIL Image of the original training image
        """
        if prototype_idx >= self.num_prototypes:
            raise ValueError(f"Prototype index {prototype_idx} out of range")

        image_path = self.image_paths[prototype_idx]
        img = Image.open(image_path).convert('RGB')
        return img

    def get_patch_location(self, prototype_idx: int) -> Tuple[int, int]:
        """
        Get the patch location (h, w) for a prototype.

        Args:
            prototype_idx: Index of the prototype

        Returns:
            Tuple of (h, w) patch coordinates
        """
        if prototype_idx >= self.num_prototypes:
            raise ValueError(f"Prototype index {prototype_idx} out of range")

        return (self.patch_h[prototype_idx], self.patch_w[prototype_idx])

    def get_image_path(self, prototype_idx: int) -> str:
        """
        Get the image path for a prototype.

        Args:
            prototype_idx: Index of the prototype

        Returns:
            Path to the original training image
        """
        if prototype_idx >= self.num_prototypes:
            raise ValueError(f"Prototype index {prototype_idx} out of range")

        return self.image_paths[prototype_idx]


class CLIPPrototypeInterpreter:
    """Interprets prototypes using CLIP image encoder."""

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize CLIP interpreter.

        Args:
            clip_model_name: HuggingFace CLIP model name
            device: Device to run on ('cuda' or 'cpu')
        """
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()

        # Cache for text embeddings
        self._text_embedding_cache = {}

    def encode_image(self, image: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Encode image(s) with CLIP.

        Args:
            image: PIL Image or list of PIL Images

        Returns:
            Normalized image embeddings (N, 512) or (512,)
        """
        with torch.no_grad():
            inputs = self.clip_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            image_features = self.clip_model.get_image_features(**inputs)
            # L2 normalize
            image_features = F.normalize(image_features, p=2, dim=-1)

            # Return single vector if single image
            if isinstance(image, Image.Image):
                return image_features[0]
            return image_features

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions with CLIP.

        Args:
            texts: List of text descriptions

        Returns:
            Normalized text embeddings (N, 512)
        """
        # Check cache
        cache_key = tuple(texts)
        if cache_key in self._text_embedding_cache:
            return self._text_embedding_cache[cache_key]

        with torch.no_grad():
            inputs = self.clip_processor(text=texts, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            text_features = self.clip_model.get_text_features(**inputs)
            # L2 normalize
            text_features = F.normalize(text_features, p=2, dim=-1)

        # Cache the result
        self._text_embedding_cache[cache_key] = text_features

        return text_features

    def find_nearest_texts(
        self,
        image_embedding: torch.Tensor,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find top-K nearest text descriptions to image embedding.

        Args:
            image_embedding: CLIP image embedding (512,)
            candidates: List of candidate text descriptions
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples, sorted by similarity
        """
        # Encode candidate texts
        text_embeddings = self.encode_texts(candidates)

        # Compute cosine similarities
        similarities = (image_embedding @ text_embeddings.T).cpu().numpy()

        # Get top-K indices
        top_k = min(top_k, len(candidates))
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results
        results = [(candidates[idx], float(similarities[idx])) for idx in top_indices]
        return results

    def interpret_prototype(
        self,
        synthetic_image: Image.Image,
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Complete interpretation pipeline for a prototype.

        Args:
            synthetic_image: Synthetic image with prototype patch
            candidates: List of candidate text descriptions
            top_k: Number of top results to return

        Returns:
            List of (text, similarity_score) tuples
        """
        # Encode image
        img_embedding = self.encode_image(synthetic_image)

        # Find nearest texts
        results = self.find_nearest_texts(img_embedding, candidates, top_k)

        return results

    def interpret_with_multiple_images(
        self,
        synthetic_images: List[Image.Image],
        candidates: List[str],
        top_k: int = 5,
        aggregate: str = 'mean'
    ) -> List[Tuple[str, float]]:
        """
        Interpret prototype using multiple synthetic images (e.g., different patch sizes).

        Args:
            synthetic_images: List of synthetic images
            candidates: List of candidate text descriptions
            top_k: Number of top results to return
            aggregate: How to aggregate results ('mean', 'max', 'vote')

        Returns:
            List of (text, similarity_score) tuples
        """
        # Encode all images
        img_embeddings = self.encode_image(synthetic_images)  # (N, 512)

        # Encode candidate texts
        text_embeddings = self.encode_texts(candidates)  # (M, 512)

        # Compute similarities for all images
        similarities = (img_embeddings @ text_embeddings.T).cpu().numpy()  # (N, M)

        # Aggregate similarities
        if aggregate == 'mean':
            aggregated_sim = similarities.mean(axis=0)
        elif aggregate == 'max':
            aggregated_sim = similarities.max(axis=0)
        elif aggregate == 'vote':
            # Vote based on top-1 for each image
            votes = np.zeros(len(candidates))
            for sim in similarities:
                top_idx = np.argmax(sim)
                votes[top_idx] += 1
            aggregated_sim = votes / len(synthetic_images)
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")

        # Get top-K
        top_k = min(top_k, len(candidates))
        top_indices = np.argsort(aggregated_sim)[::-1][:top_k]

        results = [(candidates[idx], float(aggregated_sim[idx])) for idx in top_indices]
        return results

    def clear_cache(self):
        """Clear the text embedding cache."""
        self._text_embedding_cache.clear()


class VLMPrototypeEnricher:
    """
    Enriches CLIP interpretations with VLM-generated descriptions.

    Uses Vision-Language Models to provide:
    1. Reasoning explanations for why CLIP matches make sense
    2. Feature analysis (colors, textures, shapes, parts)
    3. Free-form descriptions
    4. Relevance to ground truth class
    """

    def __init__(self, vlm_interpreter):
        """
        Initialize enricher with a VLM interpreter.

        Args:
            vlm_interpreter: VLMInterpreter instance (from vlm_interpretation.py)
        """
        self.vlm = vlm_interpreter

    def enrich_clip_results(
        self,
        image: Image.Image,
        clip_results: List[Tuple[str, float]],
        ground_truth: Optional[str] = None,
        top_k_for_reasoning: int = 5
    ) -> Dict[str, any]:
        """
        Generate VLM enrichment for CLIP results.

        Args:
            image: Prototype patch image
            clip_results: List of (text, similarity_score) from CLIP
            ground_truth: Optional ground truth class name
            top_k_for_reasoning: Number of top CLIP results to include in reasoning prompt

        Returns:
            Dictionary with:
            {
                'detailed_description': str,
                'concept_identification': str,
                'discriminative_reasoning': str
            }
        """
        results = {}

        # Generate detailed visual description (replaces: visual_features + free_description)
        results['detailed_description'] = self._generate_detailed_description(image)

        # Identify what object/concept the patch represents (NEW)
        results['concept_identification'] = self._identify_concept(image, clip_results[:3])

        # Explain discriminative features (replaces: reasoning + relevance_to_class)
        results['discriminative_reasoning'] = self._explain_discriminative_features(
            image, clip_results[:top_k_for_reasoning], ground_truth
        )

        return results

    def _generate_detailed_description(self, image: Image.Image) -> str:
        """
        Generate comprehensive visual description integrating colors, textures, shapes, and parts.
        Replaces: _analyze_visual_features() + _generate_free_description()

        Args:
            image: Prototype patch image

        Returns:
            Detailed visual description (3-5 sentences)
        """
        question = (
            "Provide a detailed description of this image patch, covering: "
            "1) Main colors and their distribution, "
            "2) Textures and surface properties, "
            "3) Shapes and geometric patterns, "
            "4) Specific objects or parts visible. "
            "Focus on concrete visual elements that make this patch distinctive. "
            "Write 3-5 sentences."
        )

        try:
            description = self.vlm.answer_question(image, question, max_length=200)
            return description
        except Exception as e:
            return f"Error generating description: {str(e)}"

    def _identify_concept(
        self,
        image: Image.Image,
        top_clip_results: List[Tuple[str, float]]
    ) -> str:
        """
        Identify what object/concept the patch represents.
        NEW functionality - provides concise concept identification.

        Args:
            image: Prototype patch image
            top_clip_results: Top CLIP interpretations (for context)

        Returns:
            Concise object/concept identification (1 sentence)
        """
        # Include CLIP results as hints
        clip_hints = ", ".join([f'"{text}"' for text, _ in top_clip_results])

        question = (
            f"Based on this image patch, identify in ONE concise sentence what "
            f"object, part, or visual concept it represents. "
            f"Context (similar concepts): {clip_hints}. "
            f"Be specific."
        )

        try:
            concept = self.vlm.answer_question(image, question, max_length=50)
            return concept
        except Exception as e:
            return f"Error: {str(e)}"

    def _explain_discriminative_features(
        self,
        image: Image.Image,
        top_clip_results: List[Tuple[str, float]],
        ground_truth: Optional[str] = None
    ) -> str:
        """
        Explain why this patch is discriminative for classification.
        Replaces: _generate_reasoning() + _explain_relevance()

        Args:
            image: Prototype patch image
            top_clip_results: Top CLIP interpretations
            ground_truth: Optional ground truth class name

        Returns:
            Explanation of discriminative features (2-3 sentences)
        """
        if ground_truth:
            # Classification-focused reasoning with ground truth
            question = (
                f"This patch is from a '{ground_truth}' image. Explain in 2-3 sentences: "
                f"1) What visual features in this patch are distinctive for identifying "
                f"'{ground_truth}', and 2) Why these features would help a neural network "
                f"recognize '{ground_truth}' versus other classes."
            )
        else:
            # Generic discriminative reasoning without ground truth
            clip_text = ", ".join([f'"{text}"' for text, _ in top_clip_results[:3]])
            question = (
                f"The top visual interpretations for this patch are: {clip_text}. "
                f"Explain in 2-3 sentences what makes this image patch visually "
                f"distinctive and how its features could help distinguish between "
                f"different object categories."
            )

        try:
            reasoning = self.vlm.answer_question(image, question, max_length=200)
            return reasoning
        except Exception as e:
            return f"Error: {str(e)}"
