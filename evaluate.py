"""Compare the fine-tuned model against two baselines on the held-out test set.

Systems:
  finetuned  - your LoRA adapter (--model)
  zero-shot  - the same base model with the same prompt but no fine-tuning
  tfidf      - TF-IDF + logistic regression trained on train+val (CPU, seconds)

For each system it reports accuracy and macro-F1 with 95% bootstrap
confidence intervals, per-class scores, a confusion matrix and how often
the output was valid JSON. Writes:
  <out-dir>/metrics.json, <out-dir>/report.md, <out-dir>/predictions_<system>.jsonl

Usage:
    python evaluate.py --model outputs/lora                     # all three systems
    python evaluate.py --systems tfidf                           # CPU-only, no model needed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doomscroll.data import load_jsonl, write_jsonl
from doomscroll.metrics import summarize
from doomscroll.schema import LABELS

SYSTEMS = ("finetuned", "zero-shot", "tfidf")
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def run_tfidf(train_rows: list[dict], test_rows: list[dict], seed: int) -> tuple[list[dict], dict]:
    from doomscroll.baseline import cross_val_macro_f1, fit_predict

    preds = fit_predict([r["input_text"] for r in train_rows], [r["label"] for r in train_rows],
                        [r["input_text"] for r in test_rows], seed)
    cv = cross_val_macro_f1([r["input_text"] for r in train_rows], [r["label"] for r in train_rows],
                            [r.get("group", i) for i, r in enumerate(train_rows)], seed=seed)
    rows = [r | {"pred_label": p, "valid_json": True} for r, p in zip(test_rows, preds)]
    return rows, {"train_cv": cv}


def run_llm(model_path: str, test_rows: list[dict], batch_size: int) -> list[dict]:
    from predict import load_classifier, predict

    model, tokenizer = load_classifier(model_path)
    results = predict(model, tokenizer, [r["input_text"] for r in test_rows], batch_size)
    rows = [r | {"pred_label": pred.label, "pred_reasoning": pred.reasoning,
                 "valid_json": pred.valid_json, "raw_output": raw}
            for r, (pred, raw) in zip(test_rows, results)]

    del model
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    return rows


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def render_report(metrics: dict) -> str:
    """Markdown report from the metrics.json structure."""
    lines = ["# Evaluation report", ""]
    info = metrics["info"]
    lines.append(f"Test set: `{info['test_path']}` ({info['n_test']} posts). "
                 f"95% CIs from {info['n_boot']} bootstrap resamples.")
    lines += ["", "## Summary", "",
              "| System | Macro-F1 | 95% CI | Accuracy | 95% CI | Valid JSON |",
              "|---|---|---|---|---|---|"]
    for name, m in metrics["systems"].items():
        lo, hi = m["macro_f1_ci95"]
        alo, ahi = m["accuracy_ci95"]
        valid = _pct(m["valid_json_rate"]) + "%" if "valid_json_rate" in m and name != "tfidf" else "n/a"
        lines.append(f"| {name} | {m['macro_f1']:.3f} | {lo:.3f}-{hi:.3f} | {m['accuracy']:.3f} | "
                     f"{alo:.3f}-{ahi:.3f} | {valid} |")
    if "tfidf" in metrics["systems"] and "train_cv" in metrics["systems"]["tfidf"]:
        cv = metrics["systems"]["tfidf"]["train_cv"]
        lines += ["", f"TF-IDF {cv['folds']}-fold group CV macro-F1 on train+val: "
                      f"{cv['macro_f1_mean']:.3f} ± {cv['macro_f1_std']:.3f}"]

    for name, m in metrics["systems"].items():
        lines += ["", f"## {name}", "", "| Label | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
        for label in LABELS:
            c = m["per_class"][label]
            lines.append(f"| {label} | {c['precision']:.3f} | {c['recall']:.3f} | {c['f1']:.3f} | {c['support']} |")
        cols = list(next(iter(m["confusion"].values())).keys())
        lines += ["", "Confusion matrix (rows = true, columns = predicted):", "",
                  "| true \\ pred | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
        for true_label, row in m["confusion"].items():
            lines.append(f"| {true_label} | " + " | ".join(str(row[c]) for c in cols) + " |")
        if m.get("errors"):
            lines += ["", "Some mistakes:", ""]
            for e in m["errors"]:
                lines.append(f"- true **{e['label']}**, predicted **{e['pred_label']}**: {e['text']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", default=",".join(SYSTEMS), help="comma list from: " + ", ".join(SYSTEMS))
    p.add_argument("--model", help="fine-tuned adapter dir or Hub repo (needed for 'finetuned')")
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="model for 'zero-shot'")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--test", type=Path, help="test JSONL (default: <data-dir>/test.jsonl)")
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    unknown = set(systems) - set(SYSTEMS)
    if unknown:
        p.error(f"unknown systems: {sorted(unknown)}")
    if "finetuned" in systems and not args.model:
        p.error("'finetuned' needs --model")

    test_path = args.test or args.data_dir / "test.jsonl"
    test_rows = load_jsonl(test_path)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict = {"info": {"test_path": str(test_path), "n_test": len(test_rows), "n_boot": args.n_boot,
                              "model": args.model, "base_model": args.base_model}, "systems": {}}

    for name in systems:
        print(f"== {name}")
        extra: dict = {}
        if name == "tfidf":
            train_rows = load_jsonl(args.data_dir / "train.jsonl") + load_jsonl(args.data_dir / "val.jsonl")
            rows, extra = run_tfidf(train_rows, test_rows, args.seed)
        else:
            rows = run_llm(args.model if name == "finetuned" else args.base_model, test_rows, args.batch_size)

        write_jsonl(rows, args.out_dir / f"predictions_{name}.jsonl")
        m = summarize([r["label"] for r in rows], [r["pred_label"] for r in rows],
                      [bool(r["valid_json"]) for r in rows], n_boot=args.n_boot, seed=args.seed)
        m["errors"] = [{"label": r["label"], "pred_label": r["pred_label"], "text": r["input_text"][:160]}
                       for r in rows if r["label"] != r["pred_label"]][:8]
        m.update(extra)
        metrics["systems"][name] = m
        lo, hi = m["macro_f1_ci95"]
        print(f"   macro-F1 {m['macro_f1']:.3f} (95% CI {lo:.3f}-{hi:.3f}), accuracy {m['accuracy']:.3f}")

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.out_dir / "report.md").write_text(render_report(metrics), encoding="utf-8")
    print(f"wrote {args.out_dir}/metrics.json and report.md")


if __name__ == "__main__":
    main()
