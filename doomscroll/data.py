"""Loading, cleaning, de-duplicating and splitting the dataset."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from doomscroll.schema import LABELS

# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def write_jsonl(rows: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# Normalisation and ids
# --------------------------------------------------------------------------- #

_WS_RE = re.compile(r"\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def normalize_text(text: str) -> str:
    """Lowercase, unicode-normalise (curly quotes -> straight) and collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return _WS_RE.sub(" ", text).strip().lower()


def text_id(text: str) -> str:
    """Stable 10-char id for a post, used by data/audit.csv."""
    return hashlib.sha1(normalize_text(text).encode("utf-8")).hexdigest()[:10]


# --------------------------------------------------------------------------- #
# Validation, exact de-dup, audit
# --------------------------------------------------------------------------- #


def validate_rows(rows: list[dict]) -> tuple[list[dict], list[str]]:
    """Keep rows with non-empty text/reasoning and a known label."""
    good, problems = [], []
    for i, row in enumerate(rows):
        text = str(row.get("input_text", "")).strip()
        label = str(row.get("label", "")).strip()
        reasoning = str(row.get("reasoning", "")).strip()
        if not text:
            problems.append(f"row {i}: empty input_text")
        elif label not in LABELS:
            problems.append(f"row {i}: unknown label {label!r}")
        elif not reasoning:
            problems.append(f"row {i}: empty reasoning")
        else:
            good.append({"id": text_id(text), "input_text": text, "label": label, "reasoning": reasoning})
    return good, problems


def dedup_exact(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Drop rows whose normalised text was already seen (keeps the first)."""
    seen, kept, removed = set(), [], []
    for row in rows:
        if row["id"] in seen:
            removed.append(row)
        else:
            seen.add(row["id"])
            kept.append(row)
    return kept, removed


AUDIT_ACTIONS = {"keep", "relabel", "drop"}


def load_audit(path: str | Path) -> list[dict]:
    """Read data/audit.csv (columns: id, action, new_label, note)."""
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        entries = [dict(r) for r in csv.DictReader(f)]
    for e in entries:
        action = e.get("action", "").strip()
        if action not in AUDIT_ACTIONS:
            raise ValueError(f"audit {e.get('id')}: action must be one of {sorted(AUDIT_ACTIONS)}")
        if action == "relabel" and e.get("new_label", "").strip() not in LABELS:
            raise ValueError(f"audit {e.get('id')}: relabel needs a valid new_label")
    return entries


def apply_audit(rows: list[dict], audit: list[dict]) -> tuple[list[dict], list[str]]:
    """Apply relabel/drop decisions by id. Returns (rows, human-readable log)."""
    by_id = {e["id"].strip(): e for e in audit}
    out, log, used = [], [], set()
    for row in rows:
        entry = by_id.get(row["id"])
        if entry is None:
            out.append(row)
            continue
        used.add(row["id"])
        action = entry["action"].strip()
        if action == "drop":
            log.append(f"drop {row['id']}: {entry.get('note', '')}")
            continue
        if action == "relabel":
            new = entry["new_label"].strip()
            log.append(f"relabel {row['id']}: {row['label']} -> {new}")
            row = {**row, "label": new, "reasoning": entry.get("new_reasoning", "").strip() or row["reasoning"]}
        out.append(row)
    for missing in sorted(set(by_id) - used):
        log.append(f"warning: audit id {missing} not found in data")
    return out, log


# --------------------------------------------------------------------------- #
# Near-duplicate grouping
# --------------------------------------------------------------------------- #


# Very common words carry no topic signal, so they are ignored when comparing posts.
_STOPWORDS = set(
    "a an the and or but if to of in on at for with is are was were be been it its this that these "
    "those you your i my me we our they their he she his her him them so just not no do does did "
    "have has had will can as by from up out about what when who how why all any more most very "
    "too than then there here really actually literally like get got im dont".split()
)


def content_words(text: str) -> set[str]:
    """Set of meaningful words (3+ letters, not a stopword) in a post."""
    return {w for w in _WORD_RE.findall(normalize_text(text)) if len(w) > 2 and w not in _STOPWORDS}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def near_duplicate_groups(
    rows: list[dict], threshold: float = 0.2, links: list[tuple[str, str]] | None = None
) -> list[int]:
    """Give each row a group number; rows whose content-word Jaccard >= threshold
    (directly or through a chain) share a group.

    The split keeps a whole group on one side, so a test post never has a
    near-copy in the training set (which would inflate test scores).
    """
    n = len(rows)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    words = [content_words(r["input_text"]) for r in rows]
    for i in range(n):
        for j in range(i + 1, n):
            if jaccard(words[i], words[j]) >= threshold:
                parent[find(i)] = find(j)

    index = {r["id"]: i for i, r in enumerate(rows)}
    for a, b in links or []:
        if a in index and b in index:
            parent[find(index[a])] = find(index[b])

    roots: dict[int, int] = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


# --------------------------------------------------------------------------- #
# Stratified, group-aware split
# --------------------------------------------------------------------------- #


def stratified_group_split(
    rows: list[dict],
    groups: list[int],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Split into train/val/test, keeping each label's proportion roughly equal
    in every split and never splitting a near-duplicate group.

    Each group is assigned by its most common label. Within a label, groups
    are shuffled (seeded) and each goes to the split furthest below its target.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError("fractions must sum to 1")
    names = ("train", "val", "test")

    members: dict[int, list[int]] = defaultdict(list)
    for idx, g in enumerate(groups):
        members[g].append(idx)

    by_label: dict[str, list[int]] = defaultdict(list)
    for g, idxs in members.items():
        majority = Counter(rows[i]["label"] for i in idxs).most_common(1)[0][0]
        by_label[majority].append(g)

    rng = random.Random(seed)
    out: dict[str, list[dict]] = {name: [] for name in names}
    for label in sorted(by_label):
        label_groups = sorted(by_label[label])
        rng.shuffle(label_groups)
        # Big groups first so they don't overshoot the small splits at the end.
        label_groups.sort(key=lambda g: -len(members[g]))
        total = sum(len(members[g]) for g in label_groups)
        targets = [f * total for f in fractions]
        counts = [0, 0, 0]
        for g in label_groups:
            size = len(members[g])
            # Split with the largest remaining deficit (ties -> earlier split).
            k = max(range(3), key=lambda s: (targets[s] - counts[s], -s))
            counts[k] += size
            out[names[k]].extend(rows[i] | {"group": g} for i in members[g])

    for name in names:
        out[name].sort(key=lambda r: r["id"])
    return out
