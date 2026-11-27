"""Fast CPU tests for everything that doesn't need a GPU. Run: pytest -q"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agreement import agreement_report  # noqa: E402
from doomscroll.data import (  # noqa: E402
    apply_audit,
    dedup_exact,
    near_duplicate_groups,
    normalize_text,
    stratified_group_split,
    text_id,
    validate_rows,
)
from doomscroll.metrics import bootstrap_ci, cohen_kappa, confusion_matrix, macro_f1, per_class  # noqa: E402
from doomscroll.schema import LABELS, UNPARSED, format_target, parse_output, to_prompt_completion  # noqa: E402
from evaluate import render_report  # noqa: E402
from prepare_data import prepare  # noqa: E402

RAGE, BRAIN, NEUTRAL, EDU = LABELS


def make_rows(n_per_label: int = 10) -> list[dict]:
    rows = []
    for li, label in enumerate(LABELS):
        for i in range(n_per_label):
            text = f"post {li}-{i} about topic{li * 100 + i} word{i} thing{li}"
            rows.append({"input_text": text, "label": label, "reasoning": "because"})
    return validate_rows(rows)[0]


# ----------------------------- schema ---------------------------------------


def test_parse_clean_json():
    pred = parse_output('{"label": "neutral", "reasoning": "A life update."}')
    assert (pred.label, pred.reasoning, pred.valid_json) == (NEUTRAL, "A life update.", True)


def test_parse_json_inside_fences_and_chatter():
    raw = 'Sure!\n```json\n{"label": "educational", "reasoning": "Explains X."}\n```'
    pred = parse_output(raw)
    assert pred.label == EDU and pred.valid_json


def test_parse_invalid_label_falls_back_to_text_match():
    pred = parse_output('{"label": "rage"} I think this is high-dopamine ragebait')
    assert pred.label == RAGE and not pred.valid_json


def test_parse_garbage_is_unparsed():
    # The kind of degenerate output the v0 model produced.
    assert parse_output("brain Post Post Post Post Post").label == UNPARSED
    assert parse_output("").label == UNPARSED


def test_target_roundtrips_through_parser():
    target = format_target(BRAIN, "Meme soup.")
    pred = parse_output(target)
    assert (pred.label, pred.reasoning, pred.valid_json) == (BRAIN, "Meme soup.", True)


def test_format_target_rejects_unknown_label():
    with pytest.raises(ValueError):
        format_target("spam", "x")


def test_prompt_completion_format():
    ex = to_prompt_completion({"input_text": "hello", "label": NEUTRAL, "reasoning": "r"})
    assert [m["role"] for m in ex["prompt"]] == ["system", "user"]
    assert ex["completion"][0]["role"] == "assistant"
    assert json.loads(ex["completion"][0]["content"])["label"] == NEUTRAL
    assert "hello" in ex["prompt"][1]["content"]


# ----------------------------- data -----------------------------------------


def test_normalize_and_id_ignore_case_whitespace_quotes():
    assert normalize_text("  It’s   FINE\n") == "it's fine"
    assert text_id("It’s fine") == text_id("it's   FINE")


def test_validate_drops_bad_rows():
    good, problems = validate_rows([
        {"input_text": "ok", "label": NEUTRAL, "reasoning": "r"},
        {"input_text": "", "label": NEUTRAL, "reasoning": "r"},
        {"input_text": "x", "label": "spam", "reasoning": "r"},
    ])
    assert len(good) == 1 and len(problems) == 2


def test_dedup_exact():
    rows = validate_rows([{"input_text": t, "label": NEUTRAL, "reasoning": "r"} for t in ["A b", "a  B", "c"]])[0]
    kept, removed = dedup_exact(rows)
    assert len(kept) == 2 and len(removed) == 1


def test_apply_audit_relabel_and_drop():
    rows = validate_rows([{"input_text": t, "label": NEUTRAL, "reasoning": "r"} for t in ["one", "two", "three"]])[0]
    audit = [
        {"id": text_id("one"), "action": "relabel", "new_label": EDU, "note": ""},
        {"id": text_id("two"), "action": "drop", "new_label": "", "note": "junk"},
        {"id": "missing000", "action": "keep", "new_label": "", "note": ""},
    ]
    out, log = apply_audit(rows, audit)
    assert {r["input_text"]: r["label"] for r in out} == {"one": EDU, "three": NEUTRAL}
    assert any("not found" in line for line in log)


def test_near_duplicates_grouped_and_links_respected():
    rows = validate_rows([
        {"input_text": "Tried the new ramen spot downtown, broth rich, pork tender, worth the wait", "label": NEUTRAL, "reasoning": "r"},
        {"input_text": "Finally tried the new ramen spot downtown! Broth rich, pork tender. Worth the wait", "label": NEUTRAL, "reasoning": "r"},
        {"input_text": "Photosynthesis converts sunlight into chemical energy in chloroplasts", "label": EDU, "reasoning": "r"},
        {"input_text": "Volcanoes form where tectonic plates diverge or subduct", "label": EDU, "reasoning": "r"},
    ])[0]
    groups = near_duplicate_groups(rows)
    assert groups[0] == groups[1] and groups[2] != groups[3]
    linked = near_duplicate_groups(rows, links=[(rows[2]["id"], rows[3]["id"])])
    assert linked[2] == linked[3]


def test_split_is_stratified_deterministic_and_disjoint():
    rows = make_rows(20)
    groups = list(range(len(rows)))
    a = stratified_group_split(rows, groups, seed=1)
    b = stratified_group_split(rows, groups, seed=1)
    assert a == b
    ids = [r["id"] for split in a.values() for r in split]
    assert len(ids) == len(set(ids)) == len(rows)
    for label in LABELS:
        assert sum(r["label"] == label for r in a["test"]) == 3  # 15% of 20


def test_split_keeps_groups_together():
    rows = make_rows(10)
    groups = [i // 3 for i in range(len(rows))]  # groups of 3
    splits = stratified_group_split(rows, groups, seed=0)
    where = {}
    for name, split_rows in splits.items():
        for r in split_rows:
            assert where.setdefault(r["group"], name) == name


def test_prepare_data_end_to_end(tmp_path):
    data = tmp_path / "data.jsonl"
    rows = [{"input_text": r["input_text"], "label": r["label"], "reasoning": "r"} for r in make_rows(10)]
    rows.append(dict(rows[0]))  # exact duplicate
    data.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    stats = prepare([data], tmp_path / "no_audit.csv", tmp_path / "out", 0.9, (0.7, 0.15, 0.15), 42)
    assert stats["exact_duplicates_dropped"] == 1
    assert sum(s["total"] for s in stats["splits"].values()) == 40
    assert (tmp_path / "out" / "test.jsonl").exists()


# ----------------------------- metrics --------------------------------------


def test_macro_f1_matches_hand_calculation():
    y_true = [RAGE, RAGE, BRAIN, NEUTRAL, EDU, EDU]
    y_pred = [RAGE, BRAIN, BRAIN, NEUTRAL, EDU, UNPARSED]
    pc = per_class(y_true, y_pred)
    assert pc[RAGE] == {"precision": 1.0, "recall": 0.5, "f1": pytest.approx(2 / 3), "support": 2}
    assert pc[BRAIN]["precision"] == 0.5
    assert pc[EDU]["recall"] == 0.5  # unparsed counts as a miss
    expected = (2 / 3 + 2 / 3 + 1.0 + 2 / 3) / 4
    assert macro_f1(y_true, y_pred) == pytest.approx(expected)


def test_macro_f1_agrees_with_sklearn():
    from sklearn.metrics import f1_score

    y_true = [RAGE, BRAIN, NEUTRAL, EDU] * 5 + [RAGE, RAGE]
    y_pred = [RAGE, NEUTRAL, NEUTRAL, EDU] * 5 + [BRAIN, RAGE]
    assert macro_f1(y_true, y_pred) == pytest.approx(f1_score(y_true, y_pred, labels=LABELS, average="macro"))


def test_confusion_matrix_counts_unparsed():
    cm = confusion_matrix([RAGE, RAGE, EDU], [RAGE, UNPARSED, NEUTRAL])
    assert cm[RAGE][RAGE] == 1 and cm[RAGE][UNPARSED] == 1 and cm[EDU][NEUTRAL] == 1


def test_bootstrap_ci_brackets_point_estimate_and_is_seeded():
    y_true = [RAGE, BRAIN, NEUTRAL, EDU] * 10
    y_pred = [RAGE, BRAIN, NEUTRAL, NEUTRAL] * 10
    lo, hi = bootstrap_ci(y_true, y_pred, macro_f1, n_boot=300, seed=3)
    assert lo <= macro_f1(y_true, y_pred) <= hi
    assert (lo, hi) == bootstrap_ci(y_true, y_pred, macro_f1, n_boot=300, seed=3)


def test_cohen_kappa_known_values():
    a = [RAGE, RAGE, BRAIN, BRAIN]
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, [RAGE, BRAIN, RAGE, BRAIN]) == pytest.approx(0.0)
    rows = [{"label_a": x, "label_b": y} for x, y in zip(a, [RAGE, RAGE, BRAIN, NEUTRAL])]
    rep = agreement_report(rows, "label_a", "label_b")
    assert rep["raw_agreement"] == 0.75
    # annotators may type one-letter shortcuts (r/b/n/e)
    short = [{"label_a": RAGE, "label_b": "r"}, {"label_a": BRAIN, "label_b": " B "}, {"label_a": EDU, "label_b": "n"}]
    assert agreement_report(short, "label_a", "label_b")["raw_agreement"] == pytest.approx(2 / 3)


# ----------------------------- baseline / report / notebook -----------------


def test_tfidf_baseline_learns_easy_data():
    from doomscroll.baseline import fit_predict

    train = ["skibidi rizz gyatt fr fr", "you are all idiots wake up", "nice walk in the park today",
             "photosynthesis converts light to energy"] * 5
    labels = [BRAIN, RAGE, NEUTRAL, EDU] * 5
    assert fit_predict(train, labels, ["gyatt skibidi rizz", "how photosynthesis converts energy"]) == [BRAIN, EDU]


def test_report_renders_all_systems():
    m = {"macro_f1": 0.5, "macro_f1_ci95": (0.4, 0.6), "accuracy": 0.5, "accuracy_ci95": (0.4, 0.6),
         "valid_json_rate": 1.0, "per_class": per_class([RAGE], [RAGE]),
         "confusion": confusion_matrix([RAGE], [RAGE]), "errors": []}
    report = render_report({"info": {"test_path": "t.jsonl", "n_test": 1, "n_boot": 10},
                            "systems": {"finetuned": m, "tfidf": m | {"train_cv": {
                                "folds": 5, "macro_f1_mean": 0.9, "macro_f1_std": 0.02}}}})
    assert "| finetuned | 0.500 |" in report and "0.900 ± 0.020" in report


def test_colab_notebook_is_valid_and_clones_main():
    nb = json.loads((ROOT / "notebooks" / "train_colab.ipynb").read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "-b main" in code and "train.py" in code and "evaluate.py" in code
    assert "HF_TOKEN" in code and "hf_" not in code.replace("HF_", "")  # no hard-coded token
