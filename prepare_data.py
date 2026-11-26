"""Clean the raw dataset and split it into train / val / test.

Steps (each one is logged):
  1. load one or more JSONL files with input_text / label / reasoning
  2. drop malformed rows (empty text, unknown label)
  3. drop exact duplicates (after lowercasing / whitespace cleanup)
  4. apply data/audit.csv: hand-made relabel / drop decisions
  5. group near-duplicates so each group stays inside one split
  6. stratified split (same label mix in every split), seeded

Usage:
    python prepare_data.py
    python prepare_data.py --input dataset.jsonl more_data.jsonl --out-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from doomscroll.data import (
    apply_audit,
    dedup_exact,
    load_audit,
    load_jsonl,
    near_duplicate_groups,
    stratified_group_split,
    validate_rows,
    write_jsonl,
)
from doomscroll.schema import LABELS


def prepare(inputs: list[Path], audit_path: Path, out_dir: Path, threshold: float,
            fractions: tuple[float, float, float], seed: int) -> dict:
    raw = [row for path in inputs for row in load_jsonl(path)]
    rows, problems = validate_rows(raw)
    rows, exact_dups = dedup_exact(rows)

    audit = load_audit(audit_path)
    rows, audit_log = apply_audit(rows, audit)
    links = [(e["id"].strip(), e["group_with"].strip()) for e in audit if (e.get("group_with") or "").strip()]

    groups = near_duplicate_groups(rows, threshold=threshold, links=links)
    splits = stratified_group_split(rows, groups, fractions=fractions, seed=seed)

    for name, split_rows in splits.items():
        write_jsonl(split_rows, out_dir / f"{name}.jsonl")

    group_sizes = Counter(groups)
    stats = {
        "inputs": [str(p) for p in inputs],
        "raw_rows": len(raw),
        "malformed_dropped": problems,
        "exact_duplicates_dropped": len(exact_dups),
        "audit_log": audit_log,
        "near_dup_threshold": threshold,
        "near_dup_groups_with_2plus": sum(1 for s in group_sizes.values() if s > 1),
        "rows_in_near_dup_groups": sum(s for s in group_sizes.values() if s > 1),
        "seed": seed,
        "fractions": fractions,
        "final_rows": len(rows),
        "splits": {name: dict(Counter(r["label"] for r in rs)) | {"total": len(rs)} for name, rs in splits.items()},
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, nargs="+", default=[Path("dataset.jsonl")])
    p.add_argument("--audit", type=Path, default=Path("data/audit.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--near-dup-threshold", type=float, default=0.2)
    p.add_argument("--fractions", type=float, nargs=3, default=(0.7, 0.15, 0.15), metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stats = prepare(args.input, args.audit, args.out_dir, args.near_dup_threshold, tuple(args.fractions), args.seed)

    print(f"raw rows:            {stats['raw_rows']}")
    print(f"malformed dropped:   {len(stats['malformed_dropped'])}")
    print(f"exact dups dropped:  {stats['exact_duplicates_dropped']}")
    for line in stats["audit_log"]:
        print(f"audit: {line}")
    print(f"near-dup groups:     {stats['near_dup_groups_with_2plus']} ({stats['rows_in_near_dup_groups']} rows)")
    print(f"final rows:          {stats['final_rows']}")
    header = f"{'split':<6}" + "".join(f"{label[:12]:>14}" for label in LABELS) + f"{'total':>8}"
    print(header)
    for name, counts in stats["splits"].items():
        print(f"{name:<6}" + "".join(f"{counts.get(label, 0):>14}" for label in LABELS) + f"{counts['total']:>8}")
    print(f"wrote {args.out_dir}/{{train,val,test}}.jsonl and stats.json")


if __name__ == "__main__":
    main()
