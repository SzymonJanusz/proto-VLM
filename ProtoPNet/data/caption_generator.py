"""
Caption generator for ImageNet class names.

Generates simple text descriptions from ImageNet class labels for use with CLIP.
"""

import json
from pathlib import Path


# ImageNet class index to WordNet ID mapping (sample - you'll need the full mapping)
# This is a subset for demonstration. Full mapping available at:
# https://github.com/anishathalye/imagenet-simple-labels
IMAGENET_CLASSES = {
    0: "tench",
    1: "goldfish",
    2: "great white shark",
    3: "tiger shark",
    4: "hammerhead shark",
    # ... (1000 classes total)
}


class ImageNetCaptionGenerator:
    """
    Generate captions for ImageNet images based on class labels.

    Args:
        class_mapping_file: Path to JSON file with class index -> name mapping (optional)
        caption_templates: List of caption templates to use
        use_multiple: If True, generate multiple captions per image
    """

    def __init__(
        self,
        class_mapping_file=None,
        caption_templates=None,
        use_multiple=False
    ):
        self.use_multiple = use_multiple

        # Load class mapping
        if class_mapping_file and Path(class_mapping_file).exists():
            with open(class_mapping_file, 'r') as f:
                self.class_names = json.load(f)
        else:
            # Use built-in subset (or download full mapping)
            self.class_names = IMAGENET_CLASSES

        # Default caption templates (similar to CLIP's prompts)
        if caption_templates is None:
            self.caption_templates = [
                "a photo of a {}",
                "a picture of a {}",
                "an image of a {}",
                "a {} in the scene",
                "a cropped photo of a {}",
                "a good photo of a {}",
            ]
        else:
            self.caption_templates = caption_templates

    def generate_caption(self, class_idx, template_idx=0):
        """
        Generate a single caption for a class.

        Args:
            class_idx: ImageNet class index (0-999)
            template_idx: Which template to use (0-5)

        Returns:
            str: Generated caption
        """
        if class_idx not in self.class_names:
            # Fallback for unknown classes
            return f"an image of class {class_idx}"

        class_name = self.class_names[class_idx]
        template = self.caption_templates[template_idx % len(self.caption_templates)]

        return template.format(class_name)

    def generate_multiple_captions(self, class_idx):
        """
        Generate multiple captions for a class using different templates.

        Args:
            class_idx: ImageNet class index

        Returns:
            list: List of captions
        """
        captions = []
        for template in self.caption_templates:
            if class_idx in self.class_names:
                captions.append(template.format(self.class_names[class_idx]))

        return captions

    def __call__(self, class_idx):
        """
        Generate caption(s) for a class.

        Args:
            class_idx: ImageNet class index

        Returns:
            str or list: Single caption or list of captions
        """
        if self.use_multiple:
            return self.generate_multiple_captions(class_idx)
        else:
            return self.generate_caption(class_idx)

    def get_all_class_captions(self):
        """
        Generate captions for all ImageNet classes.

        Returns:
            dict: {class_idx: caption}
        """
        return {
            idx: self.generate_caption(idx)
            for idx in self.class_names.keys()
        }


def download_imagenet_class_mapping(save_path="./data/imagenet_classes.json"):
    """
    Download and save ImageNet class index to name mapping.

    This is a helper to get the full 1000-class mapping.

    Args:
        save_path: Where to save the mapping
    """
    import urllib.request

    # URL to ImageNet class mapping
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"

    print(f"Downloading ImageNet class mapping from {url}")

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode())

        # Convert list to dict with indices
        class_mapping = {i: name for i, name in enumerate(data)}

        # Save to file
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)

        print(f"✓ Saved class mapping to {save_path}")
        print(f"  Total classes: {len(class_mapping)}")

        return class_mapping

    except Exception as e:
        print(f"✗ Error downloading class mapping: {e}")
        print("You can manually create the mapping or use the built-in subset")
        return None


if __name__ == '__main__':
    # Example usage
    print("ImageNet Caption Generator Example")
    print("=" * 60)

    # Try to download full class mapping
    download_imagenet_class_mapping()

    # Create caption generator
    generator = ImageNetCaptionGenerator()

    # Generate some example captions
    print("\nExample captions:")
    for class_idx in [0, 1, 2, 3, 4]:
        caption = generator.generate_caption(class_idx)
        print(f"  Class {class_idx}: {caption}")

    # Generate multiple captions for one class
    print("\nMultiple captions for class 0:")
    captions = generator.generate_multiple_captions(0)
    for i, caption in enumerate(captions):
        print(f"  {i+1}. {caption}")
