"""Build the consolidated Project 3 corruption-severity results table."""

import csv
from pathlib import Path


SOURCES = (
    (0, Path("results/tables/clean_15_results.csv")),
    (15, Path("results/tables/corrupt_f15_results.csv")),
    (30, Path("results/tables/corrupt_15_results.csv")),
    (50, Path("results/tables/corrupt_f50_results.csv")),
)

METHODS = (
    "uniform",
    "diversity",
    "combined_0.5",
    "adaptive",
)

OUTPUT = Path(
    "results/tables/severity_results.csv"
)


def read_results(path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as file:
        return {
            row["method"]: row
            for row in csv.DictReader(file)
        }


def main():
    rows = []

    for severity, path in SOURCES:
        if not path.is_file():
            raise FileNotFoundError(path)

        results = read_results(path)

        for method in METHODS:
            if method not in results:
                raise RuntimeError(
                    f"{method} missing from {path}"
                )

            row = results[method]

            rows.append({
                "severity_pct": severity,
                "method": method,
                "budget": int(row["budget"]),
                "nm_rank1": float(row["nm_rank1"]),
                "bg_rank1": float(row["bg_rank1"]),
                "cl_rank1": float(row["cl_rank1"]),
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

    print(f"PASS: Wrote {len(rows)} severity rows.")
    print("Output:", OUTPUT)

    print("\n=== NM ===")
    for row in rows:
        print(
            f"{row['severity_pct']:>2}% "
            f"{row['method']:<13} "
            f"{row['nm_rank1']:>5.2f}"
        )


if __name__ == "__main__":
    main()
