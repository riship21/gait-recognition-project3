"""Generate nested corruption-severity datasets for Project 3.

The already-frozen 30% corruption experiment is the anchor.

Severity relationship for every probe sequence:

    15% corrupted frames
        subset of
    30% corrupted frames
        subset of
    50% corrupted frames

The 30% indices and sequence seeds come directly from:
    results/tables/corruption_audit.csv

The 30% experiment itself is never modified.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np

from src.corruption.occlusion import lower_body_occlusion


SOURCE = (
    Path.home()
    / "COSC594"
    / "gait-recognition-project2"
    / "data"
    / "processed"
    / "CASIA-B-pkl-compatible"
)

ANCHOR_AUDIT = Path(
    "results/tables/corruption_audit.csv"
)

TEST_SUBJECTS = {
    f"{i:03d}"
    for i in range(75, 125)
}

PROBES = {
    "nm-05",
    "nm-06",
    "bg-01",
    "bg-02",
    "cl-01",
    "cl-02",
}

ANCHOR_FRACTION = 0.30
REMOVAL_FRACTION = 0.50


def target_count(
    total_frames: int,
    fraction: float,
) -> int:
    """Match the rounding rule used by lower_body_occlusion."""
    count = int(
        np.floor(
            total_frames * fraction
            + 0.5
        )
    )

    return max(
        1,
        min(total_frames, count),
    )


def load_anchor_audit():
    rows = {}

    with ANCHOR_AUDIT.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for row in reader:
            key = (
                row["subject"],
                row["sequence"],
                row["view"],
            )

            if key in rows:
                raise RuntimeError(
                    f"Duplicate anchor key: {key}"
                )

            rows[key] = row

    return rows


def make_relative_symlink(
    source: Path,
    destination: Path,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target = os.path.relpath(
        source,
        start=destination.parent,
    )

    destination.symlink_to(target)


def nested_indices(
    total_frames: int,
    anchor_indices: list[int],
    target_fraction: float,
    seed: int,
) -> list[int]:
    """Create indices nested around the frozen 30% anchor."""
    anchor = sorted(anchor_indices)

    desired = target_count(
        total_frames,
        target_fraction,
    )

    if target_fraction < ANCHOR_FRACTION:
        if desired > len(anchor):
            raise RuntimeError(
                "Mild target exceeds anchor size."
            )

        rng = np.random.default_rng(
            seed + 15015
        )

        selected = rng.choice(
            np.asarray(anchor),
            size=desired,
            replace=False,
        )

        return sorted(
            int(index)
            for index in selected
        )

    if target_fraction > ANCHOR_FRACTION:
        if desired < len(anchor):
            raise RuntimeError(
                "Severe target is smaller than anchor."
            )

        anchor_set = set(anchor)

        candidates = [
            index
            for index in range(total_frames)
            if index not in anchor_set
        ]

        additional_count = (
            desired - len(anchor)
        )

        if additional_count > len(candidates):
            raise RuntimeError(
                "Not enough frames to extend anchor."
            )

        rng = np.random.default_rng(
            seed + 50050
        )

        additional = rng.choice(
            np.asarray(candidates),
            size=additional_count,
            replace=False,
        )

        return sorted(
            anchor
            + [
                int(index)
                for index in additional
            ]
        )

    return anchor


def build_nested_corruption(
    clean: np.ndarray,
    anchor_corrupted: np.ndarray,
    anchor_indices: list[int],
    selected_indices: list[int],
    seed: int,
) -> np.ndarray:
    """Construct mild/severe corruption around frozen 30% frames."""
    anchor_set = set(anchor_indices)
    selected_set = set(selected_indices)

    if selected_set <= anchor_set:
        # Mild case:
        # use exactly the already-defined corrupted versions
        # for the chosen subset of anchor frames.
        corrupted = clean.copy()

        selected_array = np.asarray(
            selected_indices,
            dtype=np.int64,
        )

        corrupted[selected_array] = (
            anchor_corrupted[
                selected_array
            ]
        )

        return corrupted

    if not anchor_set <= selected_set:
        raise RuntimeError(
            "Severe corruption does not contain "
            "the full 30% anchor."
        )

    # Severe case starts from the exact frozen 30% result.
    corrupted = anchor_corrupted.copy()

    extra_indices = sorted(
        selected_set - anchor_set
    )

    if extra_indices:
        extra_array = np.asarray(
            extra_indices,
            dtype=np.int64,
        )

        # A 100% corruption rate on this extracted subset
        # applies the exact same lower-body corruption
        # implementation to every additional frame.
        extra_corrupted, local_indices = (
            lower_body_occlusion(
                clean[extra_array],
                frame_fraction=1.0,
                removal_fraction=REMOVAL_FRACTION,
                seed=seed + 50050,
            )
        )

        expected_local = list(
            range(len(extra_indices))
        )

        if local_indices != expected_local:
            raise RuntimeError(
                "Expected every extra frame "
                "to be corrupted."
            )

        corrupted[extra_array] = (
            extra_corrupted
        )

    return corrupted


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frame-fraction",
        required=True,
        type=float,
        choices=(0.15, 0.50),
    )

    args = parser.parse_args()

    frame_fraction = (
        args.frame_fraction
    )

    frame_percent = int(
        round(
            frame_fraction * 100
        )
    )

    removal_percent = int(
        round(
            REMOVAL_FRACTION * 100
        )
    )

    output = (
        Path("data")
        / "generated"
        / (
            f"CASIA-B-occlusion-"
            f"f{frame_percent:02d}-"
            f"r{removal_percent:02d}"
        )
    )

    manifest = Path(
        "results/tables/"
        f"corruption_manifest_"
        f"f{frame_percent:02d}_"
        f"r{removal_percent:02d}.csv"
    )

    if not SOURCE.is_dir():
        raise FileNotFoundError(
            SOURCE
        )

    if not ANCHOR_AUDIT.is_file():
        raise FileNotFoundError(
            ANCHOR_AUDIT
        )

    if output.exists():
        raise RuntimeError(
            f"Output already exists: {output}"
        )

    anchor_rows = (
        load_anchor_audit()
    )

    source_files = []

    for path in SOURCE.rglob("*.pkl"):
        relative = path.relative_to(
            SOURCE
        )

        if len(relative.parts) < 4:
            continue

        subject = relative.parts[0]

        if subject in TEST_SUBJECTS:
            source_files.append(path)

    source_files.sort()

    corrupted_count = 0
    linked_count = 0
    manifest_rows = []

    for source_path in source_files:
        relative = (
            source_path.relative_to(
                SOURCE
            )
        )

        subject = relative.parts[0]
        sequence = relative.parts[1]
        view = relative.parts[2]

        destination = (
            output / relative
        )

        if sequence not in PROBES:
            make_relative_symlink(
                source_path,
                destination,
            )

            linked_count += 1
            continue

        key = (
            subject,
            sequence,
            view,
        )

        if key not in anchor_rows:
            raise RuntimeError(
                f"Missing anchor row: {key}"
            )

        anchor_row = (
            anchor_rows[key]
        )

        seed = int(
            anchor_row[
                "sequence_seed"
            ]
        )

        frozen_anchor_indices = [
            int(value)
            for value in anchor_row[
                "corrupted_indices"
            ].split(";")
            if value
        ]

        with source_path.open(
            "rb"
        ) as file:
            clean = np.asarray(
                pickle.load(file)
            )

        # Independently reconstruct the frozen 30% result.
        anchor_corrupted, recomputed_anchor = (
            lower_body_occlusion(
                clean,
                frame_fraction=ANCHOR_FRACTION,
                removal_fraction=REMOVAL_FRACTION,
                seed=seed,
            )
        )

        if (
            recomputed_anchor
            != frozen_anchor_indices
        ):
            raise RuntimeError(
                "30% anchor no longer reproduces "
                f"for {key}."
            )

        selected_indices = (
            nested_indices(
                len(clean),
                frozen_anchor_indices,
                frame_fraction,
                seed,
            )
        )

        corrupted = (
            build_nested_corruption(
                clean,
                anchor_corrupted,
                frozen_anchor_indices,
                selected_indices,
                seed,
            )
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open(
            "wb"
        ) as file:
            pickle.dump(
                corrupted,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        manifest_rows.append({
            "subject": subject,
            "sequence": sequence,
            "condition": (
                sequence
                .split("-", 1)[0]
                .upper()
            ),
            "view": view,
            "frames": len(clean),
            "sequence_seed": seed,
            "frame_fraction": (
                frame_fraction
            ),
            "anchor_fraction": (
                ANCHOR_FRACTION
            ),
            "removal_fraction": (
                REMOVAL_FRACTION
            ),
            "corrupted_frames": (
                len(
                    selected_indices
                )
            ),
            "actual_corruption_fraction": (
                len(selected_indices)
                / len(clean)
            ),
            "corrupted_indices": (
                ";".join(
                    str(index)
                    for index
                    in selected_indices
                )
            ),
        })

        corrupted_count += 1

    manifest.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with manifest.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                manifest_rows[0].keys()
            ),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(
            manifest_rows
        )

    total = (
        corrupted_count
        + linked_count
    )

    print(
        "PASS: Nested severity "
        "dataset generated."
    )

    print(
        "Frame fraction:",
        frame_fraction,
    )

    print(
        "Output:",
        output.resolve(),
    )

    print(
        "Total files:",
        total,
    )

    print(
        "Corrupted probe files:",
        corrupted_count,
    )

    print(
        "Clean symlinked files:",
        linked_count,
    )

    print(
        "Manifest rows:",
        len(manifest_rows),
    )

    print(
        "Manifest:",
        manifest,
    )


if __name__ == "__main__":
    main()
