"""Plot clean-to-severe corruption results for Project 3."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt


INPUT = Path(
    "results/tables/severity_results.csv"
)

OUTPUT_DIR = Path(
    "results/figures"
)

METHODS = (
    "uniform",
    "diversity",
    "combined_0.5",
    "adaptive",
)

METRICS = (
    ("nm_rank1", "NM Rank-1 (%)", "severity_nm.png"),
    ("bg_rank1", "BG Rank-1 (%)", "severity_bg.png"),
    ("cl_rank1", "CL Rank-1 (%)", "severity_cl.png"),
)


def main():
    with INPUT.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric, ylabel, filename in METRICS:
        plt.figure(figsize=(7, 5))

        for method in METHODS:
            subset = sorted(
                (
                    row
                    for row in rows
                    if row["method"] == method
                ),
                key=lambda row: int(
                    row["severity_pct"]
                ),
            )

            x = [
                int(row["severity_pct"])
                for row in subset
            ]

            y = [
                float(row[metric])
                for row in subset
            ]

            plt.plot(
                x,
                y,
                marker="o",
                label=method,
            )

        plt.xlabel(
            "Corrupted probe frames (%)"
        )

        plt.ylabel(ylabel)

        plt.title(
            f"{ylabel} vs. Corruption Severity"
        )

        plt.xticks(
            [0, 15, 30, 50]
        )

        plt.grid(
            True,
            alpha=0.25,
        )

        plt.legend()

        plt.tight_layout()

        output = (
            OUTPUT_DIR / filename
        )

        plt.savefig(
            output,
            dpi=200,
        )

        plt.close()

        print("Saved:", output)


if __name__ == "__main__":
    main()
