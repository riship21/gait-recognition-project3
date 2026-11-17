"""Generate the frozen Project 3 corrupted CASIA-B evaluation dataset.

Clean files are symlinked to the original Project 2 dataset.
Probe files are independently written with controlled lower-body occlusion.

The exact per-sequence seeds come from:
    results/tables/corruption_audit.csv
"""

from __future__ import annotations

import csv
import os
import pickle
import shutil
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

AUDIT = Path("results/tables/corruption_audit.csv")

OUTPUT = (
    Path("data")
    / "generated"
    / "CASIA-B-occlusion-f30-r50"
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

FRAME_FRACTION = 0.30
REMOVAL_FRACTION = 0.50


def load_audit():
    mapping = {}

    with AUDIT.open(
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

            if key in mapping:
                raise RuntimeError(
                    f"Duplicate audit key: {key}"
                )

            mapping[key] = row

    return mapping


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


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)

    if not AUDIT.is_file():
        raise FileNotFoundError(AUDIT)

    if OUTPUT.exists():
        raise RuntimeError(
            f"Output already exists: {OUTPUT}\n"
            "Remove it manually before regenerating."
        )

    audit = load_audit()

    source_files = []

    for path in SOURCE.rglob("*.pkl"):
        relative = path.relative_to(SOURCE)

        if len(relative.parts) < 4:
            continue

        subject = relative.parts[0]

        if subject in TEST_SUBJECTS:
            source_files.append(path)

    source_files.sort()

    corrupted_count = 0
    linked_count = 0

    used_audit_keys = set()

    for source_path in source_files:
        relative = source_path.relative_to(SOURCE)

        subject = relative.parts[0]
        sequence = relative.parts[1]
        view = relative.parts[2]

        destination = OUTPUT / relative

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

        if key not in audit:
            raise RuntimeError(
                f"Probe exists but audit entry is missing: {key}"
            )

        row = audit[key]

        seed = int(row["sequence_seed"])

        with source_path.open("rb") as file:
            clean = np.asarray(
                pickle.load(file)
            )

        corrupted, indices = lower_body_occlusion(
            clean,
            frame_fraction=FRAME_FRACTION,
            removal_fraction=REMOVAL_FRACTION,
            seed=seed,
        )

        expected_indices = [
            int(value)
            for value in row["corrupted_indices"].split(";")
            if value
        ]

        if indices != expected_indices:
            raise RuntimeError(
                "Generated corruption does not match audit "
                f"for {key}."
            )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with destination.open("wb") as file:
            pickle.dump(
                corrupted,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        corrupted_count += 1
        used_audit_keys.add(key)

    unused_audit = set(audit) - used_audit_keys

    if unused_audit:
        raise RuntimeError(
            f"{len(unused_audit)} audit rows were not used."
        )

    total = corrupted_count + linked_count

    print("PASS: Corrupted evaluation dataset generated.")
    print("Output:", OUTPUT.resolve())
    print("Total files:", total)
    print("Corrupted probe files:", corrupted_count)
    print("Clean symlinked files:", linked_count)
    print("Audit rows used:", len(used_audit_keys))

    if corrupted_count != len(audit):
        raise RuntimeError(
            "Corrupted-file count does not equal audit-row count."
        )


if __name__ == "__main__":
    main()
