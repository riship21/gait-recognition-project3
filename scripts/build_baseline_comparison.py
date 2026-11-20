from pathlib import Path
import csv
from collections import defaultdict

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

GEI_FILE = TABLES / "gei_broad_same_view.csv"
GAITPART_FILE = TABLES / "gaitpart_broad_same_view.csv"

COMPARISON_FILE = TABLES / "gei_vs_gaitpart_same_view.csv"
SUMMARY_FILE = TABLES / "baseline_comparison_summary.csv"

CONDITIONS = ("NM", "BG", "CL")
VIEWS = (
    "000", "018", "036", "054", "072", "090",
    "108", "126", "144", "162", "180",
)


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def index_rows(rows):
    return {
        (r["condition"], r["view"]): r
        for r in rows
    }


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)

    gei_rows = read_rows(GEI_FILE)
    gait_rows = read_rows(GAITPART_FILE)

    gei = index_rows(gei_rows)
    gait = index_rows(gait_rows)

    if set(gei) != set(gait):
        raise RuntimeError("GEI and GaitPart cell sets differ.")

    comparison = []

    for condition in CONDITIONS:
        for view in VIEWS:
            key = (condition, view)

            g = gei[key]
            p = gait[key]

            gei_gallery = int(g["gallery_samples"])
            gait_gallery = int(p["gallery_samples"])

            gei_probes = int(g["probe_samples"])
            gait_probes = int(p["probe_samples"])

            if gei_gallery != gait_gallery:
                raise RuntimeError(
                    f"Gallery-count mismatch at {condition} {view}: "
                    f"{gei_gallery} vs {gait_gallery}"
                )

            if gei_probes != gait_probes:
                raise RuntimeError(
                    f"Probe-count mismatch at {condition} {view}: "
                    f"{gei_probes} vs {gait_probes}"
                )

            gei_acc = float(g["rank1_percent"])
            gait_acc = float(p["rank1_percent"])

            comparison.append({
                "condition": condition,
                "view": view,
                "gallery_samples": gei_gallery,
                "probe_samples": gei_probes,
                "gei_correct": int(g["correct"]),
                "gaitpart_correct": int(p["correct"]),
                "gei_rank1_percent": gei_acc,
                "gaitpart_rank1_percent": gait_acc,
                "gaitpart_minus_gei_pp": gait_acc - gei_acc,
            })

    with COMPARISON_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=comparison[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(comparison)

    summary = []

    grouped = defaultdict(list)

    for row in comparison:
        grouped[row["condition"]].append(row)

    for condition in CONDITIONS:
        rows = grouped[condition]

        probes = sum(r["probe_samples"] for r in rows)
        gei_correct = sum(r["gei_correct"] for r in rows)
        gait_correct = sum(r["gaitpart_correct"] for r in rows)

        gei_acc = 100.0 * gei_correct / probes
        gait_acc = 100.0 * gait_correct / probes

        summary.append({
            "condition": condition,
            "probe_samples": probes,
            "gei_correct": gei_correct,
            "gaitpart_correct": gait_correct,
            "gei_rank1_percent": gei_acc,
            "gaitpart_rank1_percent": gait_acc,
            "gaitpart_minus_gei_pp": gait_acc - gei_acc,
        })

    with SUMMARY_FILE.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(summary)

    # One figure per condition keeps each plot readable.
    for condition in CONDITIONS:
        rows = grouped[condition]

        x = [r["view"] for r in rows]
        gei_y = [r["gei_rank1_percent"] for r in rows]
        gait_y = [r["gaitpart_rank1_percent"] for r in rows]

        plt.figure(figsize=(9, 5))
        plt.plot(x, gei_y, marker="o", label="GEI")
        plt.plot(x, gait_y, marker="o", label="GaitPart")
        plt.xlabel("Camera view")
        plt.ylabel("Rank-1 accuracy (%)")
        plt.title(f"Same-view GEI vs GaitPart — {condition}")
        plt.ylim(0, 105)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()

        path = FIGURES / f"gei_vs_gaitpart_{condition.lower()}.png"
        plt.savefig(path, dpi=200)
        plt.close()

    print("PASS: GEI/GaitPart keys match")
    print("PASS: gallery/probe counts match for all 33 cells")
    print()

    print("=== BROAD BASELINE COMPARISON ===")

    for row in summary:
        print(
            f"{row['condition']}: "
            f"GEI={row['gei_rank1_percent']:.2f}%  "
            f"GaitPart={row['gaitpart_rank1_percent']:.2f}%  "
            f"delta={row['gaitpart_minus_gei_pp']:+.2f} pp"
        )

    print()
    print("Comparison:", COMPARISON_FILE)
    print("Summary:", SUMMARY_FILE)
    print("Figures:", FIGURES)


if __name__ == "__main__":
    main()
