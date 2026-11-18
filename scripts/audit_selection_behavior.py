"""Audit frame-selection behavior under controlled corruption.

Uses the frozen 30%-frame / 50%-lower-body corruption dataset and
the same 15-frame selection methods used in the OpenGait experiments.

For every corrupted probe sequence, this script measures:
- actual corrupted frames selected
- low-quality frames selected
- selected-frame quality
- temporal span
- mean pairwise temporal separation

No dataset files are modified.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path

import numpy as np

from src.sampling.adaptive import adaptive_quality_weight
from src.sampling.quality import silhouette_quality_scores
from src.sampling.selector import select_frames


DATASET = Path(
    "data/generated/CASIA-B-occlusion-f30-r50"
)

CORRUPTION_AUDIT = Path(
    "results/tables/corruption_audit.csv"
)

OUTPUT = Path(
    "results/tables/selection_behavior_audit.csv"
)

BUDGET = 15
QUALITY_THRESHOLD = 0.50

METHODS = (
    ("first_n", "first_n", 0.5),
    ("uniform", "uniform", 0.5),
    ("diversity", "diversity", 0.5),
    ("quality", "quality", 0.5),
    ("combined_0.25", "combined", 0.25),
    ("adaptive", "adaptive", 0.5),
    ("combined_0.5", "combined", 0.50),
    ("combined_0.9", "combined", 0.90),
)


def temporal_metrics(
    indices: list[int],
    total_frames: int,
) -> tuple[float, float]:
    """Return temporal span and mean pairwise separation.

    Both values are normalized by the full sequence duration.
    """
    if (
        total_frames <= 1
        or len(indices) <= 1
    ):
        return 0.0, 0.0

    values = np.asarray(
        indices,
        dtype=np.float64,
    )

    denominator = total_frames - 1

    span = float(
        (values.max() - values.min())
        / denominator
    )

    normalized = values / denominator

    differences = np.abs(
        normalized[:, None]
        - normalized[None, :]
    )

    upper = differences[
        np.triu_indices(
            len(normalized),
            k=1,
        )
    ]

    pairwise = float(
        upper.mean()
        if len(upper)
        else 0.0
    )

    return span, pairwise


def print_summary(
    rows: list[dict],
    title: str,
):
    print(f"\n=== {title} ===")

    header = (
        f"{'method':<15}"
        f"{'corrupt%':>11}"
        f"{'zero-corrupt%':>15}"
        f"{'lowQ%':>10}"
        f"{'quality':>11}"
        f"{'span':>10}"
        f"{'pairwise':>11}"
    )

    print(header)
    print("-" * len(header))

    for label, _, _ in METHODS:
        subset = [
            row
            for row in rows
            if row["method"] == label
        ]

        if not subset:
            continue

        selected_total = sum(
            row["selected_count"]
            for row in subset
        )

        corrupted_total = sum(
            row["selected_corrupted_count"]
            for row in subset
        )

        low_quality_total = sum(
            row["selected_low_quality_count"]
            for row in subset
        )

        corrupted_rate = (
            corrupted_total / selected_total
            if selected_total
            else 0.0
        )

        low_quality_rate = (
            low_quality_total / selected_total
            if selected_total
            else 0.0
        )

        zero_corruption_rate = np.mean([
            row["selected_corrupted_count"] == 0
            for row in subset
        ])

        quality = np.mean([
            row["selected_quality_mean"]
            for row in subset
        ])

        span = np.mean([
            row["temporal_span_fraction"]
            for row in subset
        ])

        pairwise = np.mean([
            row["mean_pairwise_temporal_distance"]
            for row in subset
        ])

        print(
            f"{label:<15}"
            f"{100 * corrupted_rate:>10.2f}%"
            f"{100 * zero_corruption_rate:>14.2f}%"
            f"{100 * low_quality_rate:>9.2f}%"
            f"{quality:>11.4f}"
            f"{span:>10.4f}"
            f"{pairwise:>11.4f}"
        )


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)

    if not CORRUPTION_AUDIT.is_file():
        raise FileNotFoundError(
            CORRUPTION_AUDIT
        )

    with CORRUPTION_AUDIT.open(
        newline="",
        encoding="utf-8",
    ) as file:
        audit_rows = list(
            csv.DictReader(file)
        )

    rows = []

    for audit in audit_rows:
        subject = audit["subject"]
        sequence = audit["sequence"]
        condition = audit["condition"]
        view = audit["view"]

        path = (
            DATASET
            / subject
            / sequence
            / view
            / f"{view}.pkl"
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open("rb") as file:
            frames = np.asarray(
                pickle.load(file)
            )

        if frames.ndim != 3:
            raise RuntimeError(
                f"Unexpected shape {frames.shape}: {path}"
            )

        corrupted_indices = {
            int(value)
            for value in audit[
                "corrupted_indices"
            ].split(";")
            if value
        }

        quality_scores = (
            silhouette_quality_scores(
                frames
            )
        )

        adaptive_weight = (
            adaptive_quality_weight(
                frames
            )
        )

        for (
            label,
            method,
            quality_weight,
        ) in METHODS:

            _, indices = select_frames(
                frames,
                budget=BUDGET,
                method=method,
                quality_weight=quality_weight,
            )

            indices_array = np.asarray(
                indices,
                dtype=np.int64,
            )

            selected_corrupted_count = sum(
                index in corrupted_indices
                for index in indices
            )

            selected_low_quality_count = int(
                np.sum(
                    quality_scores[
                        indices_array
                    ]
                    < QUALITY_THRESHOLD
                )
            )

            selected_quality_mean = float(
                np.mean(
                    quality_scores[
                        indices_array
                    ]
                )
            )

            span, pairwise = temporal_metrics(
                indices,
                len(frames),
            )

            rows.append({
                "subject": subject,
                "sequence": sequence,
                "condition": condition,
                "view": view,
                "frames": len(frames),
                "method": label,
                "budget": BUDGET,
                "selected_count": len(indices),
                "selected_corrupted_count": (
                    selected_corrupted_count
                ),
                "selected_corrupted_fraction": (
                    selected_corrupted_count
                    / len(indices)
                ),
                "selected_low_quality_count": (
                    selected_low_quality_count
                ),
                "selected_low_quality_fraction": (
                    selected_low_quality_count
                    / len(indices)
                ),
                "selected_quality_mean": (
                    selected_quality_mean
                ),
                "temporal_span_fraction": span,
                "mean_pairwise_temporal_distance": (
                    pairwise
                ),
                "adaptive_weight": (
                    adaptive_weight
                ),
                "selected_indices": ";".join(
                    str(index)
                    for index in indices
                ),
            })

    expected_rows = (
        len(audit_rows)
        * len(METHODS)
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            "Unexpected number of audit rows."
        )

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
            fieldnames=list(
                rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "Probe sequences:",
        len(audit_rows),
    )

    print(
        "Methods:",
        len(METHODS),
    )

    print(
        "Detailed rows:",
        len(rows),
    )

    print_summary(
        rows,
        "ALL CORRUPTED PROBES",
    )

    for condition in (
        "NM",
        "BG",
        "CL",
    ):
        print_summary(
            [
                row
                for row in rows
                if row["condition"]
                == condition
            ],
            condition,
        )

    print(
        f"\nSaved {len(rows)} rows "
        f"to {OUTPUT}"
    )


if __name__ == "__main__":
    main()
