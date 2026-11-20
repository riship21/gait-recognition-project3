from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from preprocess import preprocess_silhouette


def generate_gei(sequence_path, output_size=(128, 128)):
    frames = sorted(sequence_path.glob("*.png"))

    if not frames:
        raise FileNotFoundError(f"No PNG frames found in {sequence_path}")

    processed_frames = []

    for frame_path in frames:
        processed = preprocess_silhouette(frame_path, output_size=output_size)

        if processed is not None:
            processed_array = np.array(processed, dtype=np.float32)
            processed_frames.append(processed_array)

    if not processed_frames:
        raise ValueError("No valid processed frames found.")

    gei = np.mean(processed_frames, axis=0)

    return gei, len(processed_frames)


if __name__ == "__main__":
    sequence_path = Path(
        "data/raw/CASIA-B/GaitDatasetB-silh/001/nm-01/090"
    )

    gei, num_frames = generate_gei(sequence_path)

    print(f"Number of frames used for GEI: {num_frames}")
    print(f"GEI shape: {gei.shape}")

    plt.figure(figsize=(5, 5))
    plt.imshow(gei, cmap="gray")
    plt.title("Gait Energy Image (GEI)\nSubject 001 | nm-01 | 090")
    plt.axis("off")

    plt.savefig(
        "results/figures/subject001_nm01_090_gei.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.show()