"""
Prototype visualization utilities.

This module provides functions to visualize projected prototypes,
showing which training image patches each prototype corresponds to.
"""

import torch
from pathlib import Path
import pickle
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .projection import get_patch_bounding_box


def visualize_prototypes(projection_info, output_dir, num_prototypes=20, img_size=224, feature_size=14):
    """
    Generate HTML visualization of projected prototypes.

    Creates an HTML report showing each prototype's source image with
    highlighted patch region, along with metadata.

    Args:
        projection_info: Dictionary from PrototypeProjector.project_prototypes()
        output_dir: Directory to save visualization files
        num_prototypes: Number of prototypes to visualize (default: 20)
        img_size: Original image size (default: 224)
        feature_size: Feature map size (default: 14)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for prototype images
    proto_img_dir = output_dir / 'prototype_images'
    proto_img_dir.mkdir(exist_ok=True)

    # Determine which prototypes to visualize
    num_to_viz = min(num_prototypes, len(projection_info['prototype_id']))

    # Sort by distance (visualize best projections)
    sorted_indices = np.argsort(projection_info['distance'])[:num_to_viz]

    print(f"\nGenerating visualizations for {num_to_viz} prototypes...")

    # Generate individual prototype images
    proto_images = []
    for rank, proto_idx in enumerate(sorted_indices):
        img_path = projection_info['image_path'][proto_idx]
        if img_path is None or not Path(img_path).exists():
            print(f"  Skipping prototype {proto_idx}: image not found")
            continue

        patch_h = projection_info['patch_h'][proto_idx]
        patch_w = projection_info['patch_w'][proto_idx]
        distance = projection_info['distance'][proto_idx]
        class_id = projection_info['class_id'][proto_idx]

        # Load and process image
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))

            # Get patch bounding box
            top, left, bottom, right = get_patch_bounding_box(
                patch_h, patch_w, img_size, feature_size
            )

            # Draw bounding box on image
            img_with_box = img.copy()
            draw = ImageDraw.Draw(img_with_box)
            draw.rectangle([left, top, right, bottom], outline='red', width=3)

            # Save annotated image
            save_path = proto_img_dir / f'prototype_{proto_idx:03d}.jpg'
            img_with_box.save(save_path)

            proto_images.append({
                'proto_idx': proto_idx,
                'rank': rank + 1,
                'img_path': save_path.relative_to(output_dir),
                'patch_coords': (patch_h, patch_w),
                'distance': distance,
                'class_id': class_id,
                'bbox': (top, left, bottom, right),
            })

        except Exception as e:
            print(f"  Error processing prototype {proto_idx}: {e}")
            continue

    # Generate HTML report
    html_path = output_dir / 'prototype_visualization.html'
    _generate_html_report(html_path, proto_images, projection_info)

    print(f"✓ Saved visualization to: {html_path}")
    print(f"✓ Saved {len(proto_images)} prototype images to: {proto_img_dir}")


def save_prototype_grid(projection_info, output_dir, grid_size=(5, 4), img_size=224, feature_size=14):
    """
    Create a grid image showing multiple prototypes side by side.

    Args:
        projection_info: Dictionary from PrototypeProjector.project_prototypes()
        output_dir: Directory to save grid image
        grid_size: Tuple of (rows, cols) for grid layout
        img_size: Original image size (default: 224)
        feature_size: Feature map size (default: 14)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, cols = grid_size
    num_prototypes = rows * cols

    # Sort by distance (show best projections)
    sorted_indices = np.argsort(projection_info['distance'])[:num_prototypes]

    # Create grid image
    cell_size = img_size
    grid_img = Image.new('RGB', (cols * cell_size, rows * cell_size), color='white')

    for idx, proto_idx in enumerate(sorted_indices):
        img_path = projection_info['image_path'][proto_idx]
        if img_path is None or not Path(img_path).exists():
            continue

        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            img = img.resize((img_size, img_size))

            # Draw bounding box
            patch_h = projection_info['patch_h'][proto_idx]
            patch_w = projection_info['patch_w'][proto_idx]
            top, left, bottom, right = get_patch_bounding_box(
                patch_h, patch_w, img_size, feature_size
            )

            draw = ImageDraw.Draw(img)
            draw.rectangle([left, top, right, bottom], outline='red', width=3)

            # Paste into grid
            row = idx // cols
            col = idx % cols
            grid_img.paste(img, (col * cell_size, row * cell_size))

        except Exception as e:
            print(f"  Error adding prototype {proto_idx} to grid: {e}")
            continue

    # Save grid
    save_path = output_dir / 'prototype_grid.jpg'
    grid_img.save(save_path, quality=95)
    print(f"✓ Saved prototype grid to: {save_path}")


def _generate_html_report(html_path, proto_images, projection_info):
    """
    Generate HTML report for prototype visualization.

    Args:
        html_path: Path to save HTML file
        proto_images: List of prototype image metadata dictionaries
        projection_info: Full projection info dictionary
    """
    stats = projection_info['projection_stats']

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Prototype Projection Visualization</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .stats {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stats h2 {{
            margin-top: 0;
            color: #4CAF50;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 4px;
        }}
        .stat-label {{
            font-weight: bold;
            color: #555;
        }}
        .stat-value {{
            color: #333;
        }}
        .prototype-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .prototype-card {{
            background-color: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .prototype-card img {{
            width: 100%;
            border-radius: 4px;
            border: 2px solid #ddd;
        }}
        .prototype-info {{
            margin-top: 10px;
        }}
        .info-row {{
            margin: 5px 0;
            font-size: 14px;
        }}
        .label {{
            font-weight: bold;
            color: #555;
        }}
        .value {{
            color: #333;
        }}
        .rank {{
            background-color: #4CAF50;
            color: white;
            padding: 5px 10px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 10px;
        }}
    </style>
</head>
<body>
    <h1>Prototype Projection Visualization</h1>

    <div class="stats">
        <h2>Projection Statistics</h2>
        <div class="stat-row">
            <span class="stat-label">Total Batches Processed:</span>
            <span class="stat-value">{stats['total_batches']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Total Patches Searched:</span>
            <span class="stat-value">{stats['total_patches']:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Number of Prototypes:</span>
            <span class="stat-value">{len(projection_info['prototype_id'])}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Mean Distance:</span>
            <span class="stat-value">{stats['mean_distance']:.4f}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Median Distance:</span>
            <span class="stat-value">{stats['median_distance']:.4f}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Distance Range:</span>
            <span class="stat-value">[{stats['min_distance']:.4f}, {stats['max_distance']:.4f}]</span>
        </div>
    </div>

    <h2>Top {len(proto_images)} Prototypes (by distance)</h2>
    <div class="prototype-grid">
"""

    # Add prototype cards
    for proto_data in proto_images:
        html_content += f"""
        <div class="prototype-card">
            <div class="rank">Rank #{proto_data['rank']}</div>
            <img src="{proto_data['img_path']}" alt="Prototype {proto_data['proto_idx']}">
            <div class="prototype-info">
                <div class="info-row">
                    <span class="label">Prototype ID:</span>
                    <span class="value">{proto_data['proto_idx']}</span>
                </div>
                <div class="info-row">
                    <span class="label">Patch Coordinates:</span>
                    <span class="value">({proto_data['patch_coords'][0]}, {proto_data['patch_coords'][1]})</span>
                </div>
                <div class="info-row">
                    <span class="label">Distance:</span>
                    <span class="value">{proto_data['distance']:.4f}</span>
                </div>
                <div class="info-row">
                    <span class="label">Class ID:</span>
                    <span class="value">{proto_data['class_id']}</span>
                </div>
                <div class="info-row">
                    <span class="label">Bounding Box:</span>
                    <span class="value">[{proto_data['bbox'][0]}, {proto_data['bbox'][1]}, {proto_data['bbox'][2]}, {proto_data['bbox'][3]}]</span>
                </div>
            </div>
        </div>
"""

    html_content += """
    </div>
</body>
</html>
"""

    # Write HTML file
    with open(html_path, 'w') as f:
        f.write(html_content)
