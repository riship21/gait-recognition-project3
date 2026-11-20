from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def preprocess_silhouette(image_path, output_size=(128, 128)):
    image = Image.open(image_path).convert("L")
    img = np.array(image)

    # Find foreground pixels
    coords = np.argwhere(img > 0)

    if len(coords) == 0:
        return None

    # Find silhouette bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop around the person
    cropped = img[y_min:y_max + 1, x_min:x_max + 1]

    cropped_img = Image.fromarray(cropped)

    # Resize while preserving aspect ratio
    cropped_img.thumbnail(output_size)

    # Create black canvas
    canvas = Image.new("L", output_size, 0)

    # Center silhouette
    x_offset = (output_size[0] - cropped_img.width) // 2
    y_offset = (output_size[1] - cropped_img.height) // 2

    canvas.paste(cropped_img, (x_offset, y_offset))

    return canvas


# Everything below here runs ONLY when preprocess.py
# is executed directly.
if __name__ == "__main__":

    sequence_path = Path(
        "data/raw/CASIA-B/GaitDatasetB-silh/001/nm-01/090"
    )

    frames = sorted(sequence_path.glob("*.png"))

    print(f"Number of frames: {len(frames)}")

    indices = [
        0,
        len(frames) // 4,
        len(frames) // 2,
        3 * len(frames) // 4,
        len(frames) - 1
    ]

    selected = [frames[i] for i in indices]

    fig, axes = plt.subplots(2, len(selected), figsize=(12, 6))

    for i, frame_path in enumerate(selected):

        # Original
        original = Image.open(frame_path).convert("L")

        axes[0, i].imshow(original, cmap="gray")
        axes[0, i].set_title(frame_path.name)
        axes[0, i].axis("off")

        # Preprocessed
        processed = preprocess_silhouette(frame_path)

        axes[1, i].imshow(processed, cmap="gray")
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Preprocessed")

    plt.tight_layout()

    plt.savefig(
        "results/figures/subject001_preprocessing_comparison.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()