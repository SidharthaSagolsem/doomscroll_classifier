"""Run the classifier (fine-tuned adapter or plain base model) on posts.

Usage:
    # fine-tuned adapter (local folder or Hugging Face repo)
    python predict.py --model outputs/lora --text "POV: skibidi rizz at 3am fr fr"
    python predict.py --model <user>/doomscroll-qwen2.5-1.5b-lora --input posts.jsonl --output preds.jsonl

    # zero-shot base model (no fine-tuning), same prompt
    python predict.py --model Qwen/Qwen2.5-1.5B-Instruct --text "..."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from doomscroll.data import load_jsonl, write_jsonl
from doomscroll.schema import Prediction, build_prompt_messages, parse_output


def is_adapter(model_path: str) -> bool:
    """True if model_path (local dir or Hub repo) holds a PEFT/LoRA adapter."""
    if Path(model_path, "adapter_config.json").exists():
        return True
    if Path(model_path).exists():
        return False
    try:
        from huggingface_hub import file_exists

        return file_exists(model_path, "adapter_config.json")
    except Exception:  # noqa: BLE001 - offline / not found -> treat as a full model
        return False


def _hub_repo_exists(repo_id: str) -> bool:
    try:
        from huggingface_hub import repo_exists

        return repo_exists(repo_id)
    except Exception:  # noqa: BLE001 - offline: let from_pretrained report the real error
        return True


def load_classifier(model_path: str, load_in_4bit: bool | None = None):
    """Return (model, tokenizer). Adapters load their base model automatically."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cuda = torch.cuda.is_available()
    if load_in_4bit is None:
        load_in_4bit = cuda
    # Native bf16 needs compute capability >= 8.0; a T4 (7.5) uses fp16.
    native_bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    dtype = (torch.bfloat16 if native_bf16 else torch.float16) if cuda else torch.float32
    kwargs = {"dtype": dtype, "device_map": "auto" if cuda else None}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True
        )

    if not Path(model_path).exists() and not _hub_repo_exists(model_path):
        raise SystemExit(
            f"Model not found: {model_path!r} is neither a local folder nor a Hugging Face repo. "
            "If it's outputs/lora, training probably failed: check the train.py output."
        )
    if is_adapter(model_path):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"  # decoder-only models must pad on the left for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_raw(model, tokenizer, texts: list[str], batch_size: int = 16, max_new_tokens: int = 128) -> list[str]:
    """Greedy-decode the model's raw answer for each text."""
    import torch

    outputs: list[str] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(build_prompt_messages(t), add_generation_prompt=True, tokenize=False)
            for t in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  generated {min(start + batch_size, len(texts))}/{len(texts)}", flush=True)
    return outputs


def predict(model, tokenizer, texts: list[str], batch_size: int = 16) -> list[tuple[Prediction, str]]:
    raws = generate_raw(model, tokenizer, texts, batch_size)
    return [(parse_output(raw), raw) for raw in raws]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="adapter dir / Hub repo, or a base model id for zero-shot")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", action="append", help="a post to classify (repeatable)")
    src.add_argument("--input", type=Path, help="JSONL with an input_text field")
    p.add_argument("--output", type=Path, help="write predictions JSONL here")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()

    rows = [{"input_text": t} for t in args.text] if args.text else load_jsonl(args.input)
    model, tokenizer = load_classifier(args.model, load_in_4bit=False if args.no_4bit else None)
    results = predict(model, tokenizer, [r["input_text"] for r in rows], args.batch_size)

    out = []
    for row, (pred, raw) in zip(rows, results):
        out.append(row | {"pred_label": pred.label, "pred_reasoning": pred.reasoning,
                          "valid_json": pred.valid_json, "raw_output": raw})
    if args.output:
        write_jsonl(out, args.output)
        print(f"wrote {len(out)} predictions to {args.output}")
    else:
        for r in out:
            print(json.dumps({"text": r["input_text"][:80], "label": r["pred_label"],
                              "reasoning": r["pred_reasoning"], "valid_json": r["valid_json"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
