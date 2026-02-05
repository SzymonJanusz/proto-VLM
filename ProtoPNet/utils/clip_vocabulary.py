"""
Text candidate generation for CLIP-based prototype interpretation.

This module provides utilities to generate and manage vocabularies for
interpreting prototypes with CLIP.
"""

import json
from typing import List, Dict, Optional, Set
from pathlib import Path


# Default text templates for generating descriptions
DEFAULT_TEMPLATES = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a cropped photo of a {}",
    "a close-up photo of a {}",
]


def load_imagenet_classes(json_path: str) -> Dict[str, str]:
    """
    Load ImageNet class names from JSON file.

    Args:
        json_path: Path to JSON file with ImageNet class mappings
            Expected format: {"n01440764": "tench", ...}

    Returns:
        Dictionary mapping class IDs to class names
    """
    with open(json_path, 'r') as f:
        classes = json.load(f)
    return classes


def load_imagenet_class_list(json_path: str) -> List[str]:
    """
    Load ImageNet class names as a list.

    Args:
        json_path: Path to JSON file with ImageNet class mappings

    Returns:
        List of class names
    """
    classes = load_imagenet_classes(json_path)
    return list(classes.values())


def generate_text_templates(
    class_names: List[str],
    templates: Optional[List[str]] = None
) -> List[str]:
    """
    Generate diverse text descriptions from class names using templates.

    Args:
        class_names: List of class names (e.g., ["golden retriever", "cat"])
        templates: List of template strings with {} placeholder
            If None, uses DEFAULT_TEMPLATES

    Returns:
        List of generated text descriptions
    """
    if templates is None:
        templates = DEFAULT_TEMPLATES

    descriptions = []
    for class_name in class_names:
        for template in templates:
            descriptions.append(template.format(class_name))

    return descriptions


def generate_imagenet_vocabulary(
    json_path: str,
    templates: Optional[List[str]] = None,
    use_templates: bool = True
) -> List[str]:
    """
    Generate complete ImageNet vocabulary with templates.

    Args:
        json_path: Path to ImageNet class JSON file
        templates: Custom templates (if None, uses defaults)
        use_templates: If False, only returns class names without templates

    Returns:
        List of text descriptions
    """
    class_names = load_imagenet_class_list(json_path)

    if not use_templates:
        return class_names

    return generate_text_templates(class_names, templates)


def generate_custom_vocabulary(
    domain: str = 'general',
    include_colors: bool = True,
    include_textures: bool = True,
    include_shapes: bool = True,
    include_parts: bool = True
) -> List[str]:
    """
    Generate custom vocabulary for broader concept interpretation.

    Args:
        domain: Vocabulary domain ('general', 'animals', 'objects')
        include_colors: Include color descriptors
        include_textures: Include texture descriptors
        include_shapes: Include shape descriptors
        include_parts: Include part descriptors

    Returns:
        List of concept descriptions
    """
    vocabulary = []

    if include_colors:
        colors = [
            "red color", "blue color", "green color", "yellow color",
            "orange color", "purple color", "pink color", "brown color",
            "black color", "white color", "gray color", "golden color",
            "silver color", "beige color", "tan color"
        ]
        vocabulary.extend(colors)

    if include_textures:
        textures = [
            "furry texture", "smooth texture", "rough texture", "scaly texture",
            "feathered texture", "soft texture", "hard texture", "shiny surface",
            "matte surface", "metallic surface", "wooden texture", "fabric texture",
            "leather texture", "glass surface", "plastic surface"
        ]
        vocabulary.extend(textures)

    if include_shapes:
        shapes = [
            "round shape", "circular shape", "oval shape", "rectangular shape",
            "square shape", "triangular shape", "curved shape", "straight lines",
            "angular shape", "elongated shape", "compact shape", "symmetrical shape"
        ]
        vocabulary.extend(shapes)

    if domain == 'animals' and include_parts:
        parts = [
            "eye", "nose", "mouth", "ear", "paw", "tail", "leg", "wing",
            "beak", "fur", "feathers", "whiskers", "claws", "teeth",
            "head", "face", "body", "neck"
        ]
        vocabulary.extend(parts)
    elif domain == 'objects' and include_parts:
        parts = [
            "handle", "button", "wheel", "screen", "keyboard", "door",
            "window", "edge", "corner", "surface", "base", "top"
        ]
        vocabulary.extend(parts)
    elif domain == 'general' and include_parts:
        parts = [
            "edge", "corner", "center", "border", "surface", "pattern",
            "detail", "feature", "element", "component"
        ]
        vocabulary.extend(parts)

    return vocabulary


def generate_hybrid_vocabulary(
    imagenet_json_path: str,
    include_custom: bool = True,
    custom_domain: str = 'general',
    imagenet_templates: Optional[List[str]] = None
) -> List[str]:
    """
    Generate hybrid vocabulary combining ImageNet classes and custom concepts.

    Args:
        imagenet_json_path: Path to ImageNet class JSON file
        include_custom: Include custom vocabulary
        custom_domain: Domain for custom vocabulary
        imagenet_templates: Templates for ImageNet classes

    Returns:
        Combined vocabulary
    """
    # Get ImageNet vocabulary
    imagenet_vocab = generate_imagenet_vocabulary(
        imagenet_json_path,
        templates=imagenet_templates,
        use_templates=True
    )

    if not include_custom:
        return imagenet_vocab

    # Get custom vocabulary
    custom_vocab = generate_custom_vocabulary(domain=custom_domain)

    # Combine and deduplicate
    combined = list(set(imagenet_vocab + custom_vocab))

    return combined


def filter_vocabulary_by_similarity(
    vocabulary: List[str],
    query: str,
    max_candidates: int = 1000,
    clip_model=None
) -> List[str]:
    """
    Filter vocabulary to most relevant candidates using CLIP text similarity.

    This can reduce computation when the full vocabulary is very large.

    Args:
        vocabulary: Full vocabulary list
        query: Query text to compare against
        max_candidates: Maximum number of candidates to return
        clip_model: CLIPPrototypeInterpreter instance (optional)

    Returns:
        Filtered vocabulary list
    """
    if len(vocabulary) <= max_candidates:
        return vocabulary

    if clip_model is None:
        # If no CLIP model provided, just return first max_candidates
        return vocabulary[:max_candidates]

    # Encode query
    query_embedding = clip_model.encode_texts([query])[0]

    # Encode all vocabulary
    vocab_embeddings = clip_model.encode_texts(vocabulary)

    # Compute similarities
    import torch
    similarities = (query_embedding @ vocab_embeddings.T).cpu().numpy()

    # Get top candidates
    top_indices = similarities.argsort()[::-1][:max_candidates]

    return [vocabulary[idx] for idx in top_indices]


def get_vocabulary(
    vocabulary_type: str,
    imagenet_json_path: Optional[str] = None,
    templates: Optional[List[str]] = None,
    custom_domain: str = 'general'
) -> List[str]:
    """
    Convenience function to get vocabulary by type.

    Args:
        vocabulary_type: Type of vocabulary
            - 'imagenet': ImageNet classes only
            - 'custom': Custom concepts only
            - 'hybrid': Both ImageNet and custom
        imagenet_json_path: Path to ImageNet JSON (required for imagenet/hybrid)
        templates: Custom templates for ImageNet classes
        custom_domain: Domain for custom vocabulary

    Returns:
        Generated vocabulary list
    """
    if vocabulary_type == 'imagenet':
        if imagenet_json_path is None:
            raise ValueError("imagenet_json_path required for 'imagenet' vocabulary")
        return generate_imagenet_vocabulary(imagenet_json_path, templates)

    elif vocabulary_type == 'custom':
        return generate_custom_vocabulary(domain=custom_domain)

    elif vocabulary_type == 'hybrid':
        if imagenet_json_path is None:
            raise ValueError("imagenet_json_path required for 'hybrid' vocabulary")
        return generate_hybrid_vocabulary(
            imagenet_json_path,
            include_custom=True,
            custom_domain=custom_domain,
            imagenet_templates=templates
        )

    else:
        raise ValueError(f"Unknown vocabulary type: {vocabulary_type}")


def create_class_specific_vocabulary(
    class_name: str,
    imagenet_json_path: Optional[str] = None,
    templates: Optional[List[str]] = None
) -> List[str]:
    """
    Create vocabulary focused on a specific class and related concepts.

    Args:
        class_name: Name of the class (e.g., "golden retriever")
        imagenet_json_path: Path to ImageNet JSON (optional, for related classes)
        templates: Custom templates

    Returns:
        Class-specific vocabulary
    """
    if templates is None:
        templates = DEFAULT_TEMPLATES

    vocabulary = []

    # Add the class itself with templates
    for template in templates:
        vocabulary.append(template.format(class_name))

    # Add general descriptors
    custom_vocab = generate_custom_vocabulary(domain='general')
    vocabulary.extend(custom_vocab)

    # If ImageNet classes provided, add those too
    if imagenet_json_path is not None:
        imagenet_classes = load_imagenet_class_list(imagenet_json_path)
        for template in templates:
            for img_class in imagenet_classes:
                vocabulary.append(template.format(img_class))

    return vocabulary


def save_vocabulary(vocabulary: List[str], output_path: str):
    """
    Save vocabulary to a text file (one entry per line).

    Args:
        vocabulary: List of text descriptions
        output_path: Path to output file
    """
    with open(output_path, 'w') as f:
        for item in vocabulary:
            f.write(f"{item}\n")


def load_vocabulary(input_path: str) -> List[str]:
    """
    Load vocabulary from a text file.

    Args:
        input_path: Path to vocabulary file

    Returns:
        List of text descriptions
    """
    with open(input_path, 'r') as f:
        vocabulary = [line.strip() for line in f if line.strip()]
    return vocabulary
