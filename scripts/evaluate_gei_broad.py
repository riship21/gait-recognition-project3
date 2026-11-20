"""Broader Project 1 GEI baseline evaluation for Project 3.

Uses the exact GEI implementation validated against the original
Project 1 matched 90-degree result.

Protocol:
- CASIA-B subjects 075-124
- gallery: nm-01 through nm-04
- probes:
    NM: nm-05, nm-06
    BG: bg-01, bg-02
    CL: cl-01, cl-02
- all 11 camera viewpoints
- same-view gallery/probe matching
- Euclidean nearest-neighbor
- Rank-1 identification accuracy

The raw CASIA-B silhouettes are not redistributed.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "scripts" / "gei_baseline"),
)

from gei import generate_gei


DATA = Path(
    os.environ.get(
        "CASIAB_RAW_ROOT",
        str(
            Path.home()
            / "COSC594"
            / "casiab-raw-heldout"
        ),
    )
)

OUTPUT = PROJECT_ROOT / "results" / "tables"

SUBJECTS = [
    f"{i:03d}"
    for i in range(75, 125)
]

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

GALLERY = (
    "nm-01",
    "nm-02",
    "nm-03",
    "nm-04",
)

PROBES = {
    "NM": (
        "nm-05",
        "nm-06",
    ),
    "BG": (
        "bg-01",
        "bg-02",
    ),
    "CL": (
        "cl-01",
        "cl-02",
    ),
}


def main():
    if not DATA.is_dir():
        raise FileNotFoundError(DATA)

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache = {}
    missing = []

    def feature(
        subject: str,
        sequence: str,
        view: str,
    ):
        key = (
            subject,
            sequence,
            view,
        )

        if key in cache:
            return cache[key]

        path = (
            DATA
            / subject
            / sequence
            / view
        )

        try:
            gei, frames = generate_gei(path)

        except (
            FileNotFoundError,
            ValueError,
        ) as error:
            missing.append({
                "subject": subject,
                "sequence": sequence,
                "view": view,
                "reason": str(error),
            })

            cache[key] = None
            return None

        value = (
            gei.astype(
                np.float32
            ).flatten()
            / 255.0,
            frames,
        )

        cache[key] = value

        return value

    cell_rows = []
    prediction_rows = []

    for view in VIEWS:
        print(
            f"Processing gallery view {view}...",
            flush=True,
        )

        gallery_features = []
        gallery_labels = []
        gallery_sequences = []

        for subject in SUBJECTS:
            for sequence in GALLERY:
                result = feature(
                    subject,
                    sequence,
                    view,
                )

                if result is None:
                    continue

                vector, _ = result

                gallery_features.append(
                    vector
                )

                gallery_labels.append(
                    subject
                )

                gallery_sequences.append(
                    sequence
                )

        if not gallery_features:
            raise RuntimeError(
                f"No gallery samples for view {view}."
            )

        gallery_features = np.stack(
            gallery_features
        )

        gallery_labels = np.asarray(
            gallery_labels
        )

        gallery_sequences = np.asarray(
            gallery_sequences
        )

        for condition, probe_sequences in PROBES.items():
            correct = 0
            total = 0

            for subject in SUBJECTS:
                for probe_sequence in probe_sequences:
                    result = feature(
                        subject,
                        probe_sequence,
                        view,
                    )

                    if result is None:
                        continue

                    vector, frames = result

                    distances = np.linalg.norm(
                        gallery_features - vector,
                        axis=1,
                    )

                    index = int(
                        np.argmin(distances)
                    )

                    predicted_subject = str(
                        gallery_labels[index]
                    )

                    predicted_sequence = str(
                        gallery_sequences[index]
                    )

                    matched = (
                        predicted_subject
                        == subject
                    )

                    correct += int(
                        matched
                    )

                    total += 1

                    prediction_rows.append({
                        "condition": condition,
                        "view": view,
                        "true_subject": subject,
                        "probe_sequence": probe_sequence,
                        "probe_frames": frames,
                        "predicted_subject": predicted_subject,
                        "matched_gallery_sequence": predicted_sequence,
                        "distance": float(
                            distances[index]
                        ),
                        "correct": matched,
                    })

            if total == 0:
                raise RuntimeError(
                    f"No probes for {condition} view {view}."
                )

            rank1 = (
                100.0
                * correct
                / total
            )

            cell_rows.append({
                "condition": condition,
                "view": view,
                "gallery_samples": len(
                    gallery_features
                ),
                "probe_samples": total,
                "correct": correct,
                "rank1_percent": rank1,
            })

            print(
                f"  {condition}: "
                f"{correct}/{total} "
                f"= {rank1:.2f}%",
                flush=True,
            )

    # Mandatory backward-compatibility check.
    matched_090_nm = next(
        row
        for row in cell_rows
        if (
            row["condition"] == "NM"
            and row["view"] == "090"
        )
    )

    if (
        matched_090_nm["probe_samples"] != 100
        or abs(
            matched_090_nm[
                "rank1_percent"
            ]
            - 95.0
        ) > 1e-9
    ):
        raise RuntimeError(
            "Broader evaluator failed the original "
            "90-degree NM validation."
        )

    condition_rows = []

    for condition in PROBES:
        subset = [
            row
            for row in cell_rows
            if row["condition"] == condition
        ]

        total_correct = sum(
            row["correct"]
            for row in subset
        )

        total_probes = sum(
            row["probe_samples"]
            for row in subset
        )

        macro_rank1 = float(
            np.mean([
                row["rank1_percent"]
                for row in subset
            ])
        )

        weighted_rank1 = (
            100.0
            * total_correct
            / total_probes
        )

        condition_rows.append({
            "condition": condition,
            "views": len(subset),
            "total_correct": total_correct,
            "total_probes": total_probes,
            "macro_view_rank1_percent": macro_rank1,
            "weighted_rank1_percent": weighted_rank1,
        })

    with (
        OUTPUT
        / "gei_broad_same_view.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                cell_rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            cell_rows
        )

    with (
        OUTPUT
        / "gei_broad_condition_summary.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                condition_rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            condition_rows
        )

    with (
        OUTPUT
        / "gei_broad_predictions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                prediction_rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            prediction_rows
        )

    missing_path = (
        OUTPUT
        / "gei_broad_missing.csv"
    )

    if missing:
        with missing_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(
                    missing[0].keys()
                ),
                lineterminator="\n",
            )

            writer.writeheader()
            writer.writerows(
                missing
            )

    print("\n=== CONDITION SUMMARY ===")

    for row in condition_rows:
        print(
            f"{row['condition']}: "
            f"macro={row['macro_view_rank1_percent']:.2f}% "
            f"weighted={row['weighted_rank1_percent']:.2f}% "
            f"({row['total_correct']}/"
            f"{row['total_probes']})"
        )

    print()
    print(
        "90-degree NM validation: "
        "PASS (95.00%)"
    )

    print(
        "Missing sequences:",
        len(missing),
    )

    print(
        "Detailed predictions:",
        len(prediction_rows),
    )

    print(
        "Results saved to:",
        OUTPUT,
    )


if __name__ == "__main__":
    main()
