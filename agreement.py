"""Check how well two people agree when labelling the real test set.

If two careful annotators only agree 60% of the time, no model can be
expected to score much higher, and the label definitions need tightening.

Usage:
    python agreement.py data/real_test/labels.csv
    python agreement.py labels.csv --a label_a --b label_b --disagreements disagreements.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from doomscroll.metrics import cohen_kappa
from doomscroll.schema import LABELS


# One-letter shortcuts annotators can type instead of the full label.
SHORTCUTS = {"r": LABELS[0], "b": LABELS[1], "n": LABELS[2], "e": LABELS[3]}


def normalize_label(value: str) -> str:
    value = (value or "").strip()
    return SHORTCUTS.get(value.lower(), value)


def agreement_report(rows: list[dict], col_a: str, col_b: str) -> dict:
    pairs = [(normalize_label(r.get(col_a)), normalize_label(r.get(col_b))) for r in rows]
    pairs = [(x, y) for x, y in pairs if x and y]
    bad = sorted({x for pair in pairs for x in pair if x not in LABELS})
    if bad:
        raise ValueError(f"unknown labels in CSV: {bad}")
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    n = len(pairs)
    return {
        "n_both_labelled": n,
        "n_rows": len(rows),
        "raw_agreement": sum(x == y for x, y in pairs) / n if n else 0.0,
        "cohen_kappa": cohen_kappa(a, b),
        "counts_a": dict(Counter(a)),
        "counts_b": dict(Counter(b)),
        "confusions": dict(Counter(f"{x} | {y}" for x, y in pairs if x != y).most_common()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("--a", default="label_a")
    p.add_argument("--b", default="label_b")
    p.add_argument("--disagreements", type=Path, help="write rows where A and B differ to this CSV")
    args = p.parse_args()

    with args.csv.open(encoding="utf-8-sig", newline="") as f:  # -sig: Excel adds a BOM
        rows = list(csv.DictReader(f))
    rep = agreement_report(rows, args.a, args.b)

    print(f"rows labelled by both: {rep['n_both_labelled']} / {rep['n_rows']}")
    print(f"raw agreement:         {rep['raw_agreement']:.3f}")
    print(f"Cohen's kappa:         {rep['cohen_kappa']:.3f}  (>0.6 substantial, >0.8 near-perfect)")
    for pair, count in rep["confusions"].items():
        print(f"  {count:>3}  A|B = {pair}")

    if args.disagreements:
        diff = [r for r in rows if normalize_label(r.get(args.a)) and normalize_label(r.get(args.b))
                and normalize_label(r[args.a]) != normalize_label(r[args.b])]
        with args.disagreements.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(diff)
        print(f"wrote {len(diff)} disagreements to {args.disagreements}")


if __name__ == "__main__":
    main()
