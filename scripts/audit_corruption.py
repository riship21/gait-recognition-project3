"""Audit the Project 3 corruption protocol on CASIA-B test probes.

This script does NOT modify the dataset.

Protocol:
- subjects 075-124
- probes:
    nm-05, nm-06
    bg-01, bg-02
    cl-01, cl-02
- 30% of frames corrupted
- lower 50% of foreground removed
- master RNG seed 2026
- deterministic per-sequence seeds saved to CSV
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from src.corruption.occlusion import lower_body_occlusion
from src.sampling.adaptive import adaptive_quality_weight
from src.sampling.quality import silhouette_quality_scores


DATASET = (
    Path.home()
    / "COSC594"
    / "gait-recognition-project2"
    / "data"
    / "processed"
    / "CASIA-B-pkl-compatible"
)

OUTPUT = Path("results/tables/corruption_audit.csv")

SUBJECTS = [f"{i:03d}" for i in range(75, 125)]

PROBES = (
    "nm-05",
    "nm-06",
    "bg-01",
    "bg-02",
    "cl-01",
    "cl-02",
)

VIEWS = (
    "000",
    "018",
    "036",
    "054",
    "072",
    "090",
    "108",
    "126",
    "144",
    "162",
    "180",
)

MASTER_SEED = 2026
FRAME_FRACTION = 0.30
REMOVAL_FRACTION = 0.50
QUALITY_THRESHOLD = 0.50


def condition_from_sequence(sequence: str) -> str:
    return sequence.split("-", 1)[0].upper()


def summarize(rows, label):
    if not rows:
        return

    total_corrupted = sum(
        row["corrupted_frames"]
        for row in rows
    )

    total_detected = sum(
        row["corrupted_frames_below_threshold"]
        for row in rows
    )

    total_uncorrupted = sum(
        row["uncorrupted_frames"]
        for row in rows
    )

    total_false_low = sum(
        row["uncorrupted_frames_below_threshold"]
        for row in rows
    )

    clean_weights = np.array(
        [row["clean_adaptive_weight"] for row in rows],
        dtype=np.float64,
    )

    corrupted_weights = np.array(
        [row["corrupted_adaptive_weight"] for row in rows],
        dtype=np.float64,
    )

    increases = corrupted_weights - clean_weights

    selected_quality = np.array(
        [row["corrupted_frame_quality_mean"] for row in rows],
        dtype=np.float64,
    )

    unselected_quality = np.array(
        [row["uncorrupted_frame_quality_mean"] for row in rows],
        dtype=np.float64,
    )

    corruption_fraction = np.array(
        [row["actual_corruption_fraction"] for row in rows],
        dtype=np.float64,
    )

    print(f"\n=== {label} ===")
    print("Sequences:", len(rows))

    print(
        "Actual corrupted-frame fraction:"
        f" mean={corruption_fraction.mean():.4f}"
        f" min={corruption_fraction.min():.4f}"
        f" max={corruption_fraction.max():.4f}"
    )

    detection_rate = (
        total_detected / total_corrupted
        if total_corrupted
        else 0.0
    )

    false_low_rate = (
        total_false_low / total_uncorrupted
        if total_uncorrupted
        else 0.0
    )

    print(
        "Corrupted frames below quality threshold:"
        f" {total_detected}/{total_corrupted}"
        f" ({100 * detection_rate:.2f}%)"
    )

    print(
        "Uncorrupted frames below quality threshold:"
        f" {total_false_low}/{total_uncorrupted}"
        f" ({100 * false_low_rate:.2f}%)"
    )

    print(
        "Mean quality:"
        f" corrupted={selected_quality.mean():.4f}"
        f" uncorrupted={unselected_quality.mean():.4f}"
    )

    print(
        "Clean adaptive weight:"
        f" mean={clean_weights.mean():.4f}"
        f" median={np.median(clean_weights):.4f}"
        f" max={clean_weights.max():.4f}"
    )

    print(
        "Corrupted adaptive weight:"
        f" mean={corrupted_weights.mean():.4f}"
        f" median={np.median(corrupted_weights):.4f}"
        f" min={corrupted_weights.min():.4f}"
        f" max={corrupted_weights.max():.4f}"
    )

    print(
        "Adaptive-weight increase:"
        f" mean={increases.mean():.4f}"
        f" median={np.median(increases):.4f}"
    )

    increased = np.mean(increases > 0)

    print(
        "Sequences with increased adaptive weight:"
        f" {100 * increased:.2f}%"
    )


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)

    expected = []
    missing = []

    for subject in SUBJECTS:
        for sequence in PROBES:
            for view in VIEWS:
                path = (
                    DATASET
                    / subject
                    / sequence
                    / view
                    / f"{view}.pkl"
                )

                if path.is_file():
                    expected.append(path)
                else:
                    missing.append(path)

    print("Expected theoretical probe files:", 50 * 6 * 11)
    print("Existing probe files:", len(expected))
    print("Missing probe files:", len(missing))

    if missing:
        print("\nFirst missing files:")
        for path in missing[:10]:
            print(" ", path.relative_to(DATASET))

    if not expected:
        raise RuntimeError("No probe files found.")

    rng = np.random.default_rng(MASTER_SEED)

    rows = []

    for path in sorted(expected):
        relative = path.relative_to(DATASET)

        subject = relative.parts[0]
        sequence = relative.parts[1]
        view = relative.parts[2]
        condition = condition_from_sequence(sequence)

        with path.open("rb") as file:
            clean = pickle.load(file)

        clean = np.asarray(clean)

        if clean.ndim != 3:
            raise RuntimeError(
                f"Unexpected shape {clean.shape}: {path}"
            )

        sequence_seed = int(
            rng.integers(
                0,
                2**32 - 1,
            )
        )

        corrupted, corrupted_indices = lower_body_occlusion(
            clean,
            frame_fraction=FRAME_FRACTION,
            removal_fraction=REMOVAL_FRACTION,
            seed=sequence_seed,
        )

        clean_scores = silhouette_quality_scores(clean)
        corrupted_scores = silhouette_quality_scores(corrupted)

        selected = np.asarray(
            corrupted_indices,
            dtype=np.int64,
        )

        selected_set = set(corrupted_indices)

        unselected = np.asarray(
            [
                i
                for i in range(len(clean))
                if i not in selected_set
            ],
            dtype=np.int64,
        )

        selected_low = int(
            np.sum(
                corrupted_scores[selected]
                < QUALITY_THRESHOLD
            )
        )

        unselected_low = int(
            np.sum(
                corrupted_scores[unselected]
                < QUALITY_THRESHOLD
            )
        )

        rows.append({
            "subject": subject,
            "sequence": sequence,
            "condition": condition,
            "view": view,
            "frames": len(clean),
            "sequence_seed": sequence_seed,
            "corrupted_frames": len(selected),
            "uncorrupted_frames": len(unselected),
            "actual_corruption_fraction": (
                len(selected) / len(clean)
            ),
            "clean_low_quality_fraction": float(
                np.mean(
                    clean_scores
                    < QUALITY_THRESHOLD
                )
            ),
            "corrupted_low_quality_fraction": float(
                np.mean(
                    corrupted_scores
                    < QUALITY_THRESHOLD
                )
            ),
            "corrupted_frames_below_threshold": selected_low,
            "uncorrupted_frames_below_threshold": unselected_low,
            "corrupted_frame_quality_mean": float(
                corrupted_scores[selected].mean()
            ),
            "uncorrupted_frame_quality_mean": float(
                corrupted_scores[unselected].mean()
            ),
            "clean_adaptive_weight": float(
                adaptive_quality_weight(clean)
            ),
            "corrupted_adaptive_weight": float(
                adaptive_quality_weight(corrupted)
            ),
            "weight_increase": float(
                adaptive_quality_weight(corrupted)
                - adaptive_quality_weight(clean)
            ),
            "corrupted_indices": ";".join(
                str(i)
                for i in corrupted_indices
            ),
        })

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    summarize(
        rows,
        "ALL PROBES",
    )

    for condition in ("NM", "BG", "CL"):
        summarize(
            [
                row
                for row in rows
                if row["condition"] == condition
            ],
            condition,
        )

    print(
        f"\nSaved {len(rows)} rows to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
