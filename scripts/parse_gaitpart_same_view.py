"""Parse Project 3 same-view GaitPart results from an OpenGait log."""

from __future__ import annotations

import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

LOG = (
    ROOT
    / "results"
    / "raw"
    / "gaitpart_full_same_view.log"
)

OUTPUT = (
    ROOT
    / "results"
    / "tables"
    / "gaitpart_broad_same_view.csv"
)

PATTERN = re.compile(
    r"PROJECT3_SAME_VIEW,"
    r"(NM|BG|CL),"
    r"(\d{3}),"
    r"(\d+),"
    r"(\d+),"
    r"([0-9.]+)"
)

EXPECTED_VIEWS = {
    "000", "018", "036", "054", "072", "090",
    "108", "126", "144", "162", "180",
}

EXPECTED_CONDITIONS = {"NM", "BG", "CL"}


def main():
    text = LOG.read_text(
        encoding="utf-8",
        errors="replace",
    )

    matches = PATTERN.findall(text)

    if len(matches) != 33:
        raise RuntimeError(
            f"Expected 33 same-view rows, found {len(matches)}."
        )

    rows = []

    for condition, view, gallery, probes, reported_rank1 in matches:
        gallery = int(gallery)
        probes = int(probes)
        reported_rank1 = float(reported_rank1)

        # OpenGait reports Rank-1 rounded to two decimals.
        correct = round(
            reported_rank1
            * probes
            / 100.0
        )

        exact_rank1 = (
            100.0
            * correct
            / probes
        )

        rows.append({
            "condition": condition,
            "view": view,
            "gallery_samples": gallery,
            "probe_samples": probes,
            "correct": correct,
            "rank1_percent": exact_rank1,
        })

    keys = {
        (r["condition"], r["view"])
        for r in rows
    }

    expected = {
        (condition, view)
        for condition in EXPECTED_CONDITIONS
        for view in EXPECTED_VIEWS
    }

    if keys != expected:
        raise RuntimeError(
            "Same-view condition/view cells are incomplete."
        )

    nm90 = next(
        r
        for r in rows
        if (
            r["condition"] == "NM"
            and r["view"] == "090"
        )
    )

    if nm90["rank1_percent"] != 100.0:
        raise RuntimeError(
            "Expected validated NM 090 result of 100.00%."
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )

        writer.writeheader()
        writer.writerows(rows)

    print("PASS: parsed 33 same-view cells from raw OpenGait log")
    print("PASS: all 11 views present for NM, BG, and CL")
    print("PASS: NM 090 = 100.00%")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
