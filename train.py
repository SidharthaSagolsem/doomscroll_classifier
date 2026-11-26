"""QLoRA fine-tune of Qwen2.5-1.5B-Instruct on the doomscroll dataset.

What happens:
  * The base model is loaded in 4-bit (on a CUDA GPU) to fit in a T4's 15 GB.
  * Small LoRA adapter matrices are added to every linear layer; only those
    (~1% of the weights) are trained.
  * Data is in prompt-completion format, so the loss is computed ONLY on the
    JSON answer, not on the prompt.
  * At the end the adapter (~70 MB, not the 3 GB base model) is saved to
    --output-dir and, with --push-to-hub, uploaded to Hugging Face so it
    survives the Colab session ending.

Usage (Colab):
    python train.py --push-to-hub --hub-repo <user>/doomscroll-qwen2.5-1.5b-lora
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from doomscroll.data import load_jsonl
from doomscroll.schema import to_prompt_completion

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/lora"))
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-4bit", action="store_true", help="Load the base model in full precision.")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-repo", default=None, help="e.g. your-username/doomscroll-qwen2.5-1.5b-lora")
    p.add_argument("--private", action="store_true", help="Create the Hub repo as private.")
    args = p.parse_args(argv)
    if args.push_to_hub and not args.hub_repo:
        p.error("--push-to-hub needs --hub-repo")
    return args


def load_split(data_dir: Path, name: str) -> list[dict]:
    path = data_dir / f"{name}.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run `python prepare_data.py` first.")
    return [to_prompt_completion(r) for r in load_jsonl(path)]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # Heavy imports here so `--help` and the tests don't need a GPU stack.
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if args.push_to_hub and not (os.environ.get("HF_TOKEN") or _hf_logged_in()):
        raise SystemExit("Not logged in to Hugging Face. Set HF_TOKEN or run huggingface_hub.login() first.")

    train_ds = Dataset.from_list(load_split(args.data_dir, "train"))
    val_ds = Dataset.from_list(load_split(args.data_dir, "val"))
    print(f"train={len(train_ds)} val={len(val_ds)}")

    cuda = torch.cuda.is_available()
    use_4bit = cuda and not args.no_4bit
    bf16 = cuda and torch.cuda.is_bf16_supported()  # T4: False -> fp16
    compute_dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"cuda={cuda} 4bit={use_4bit} dtype={compute_dtype if cuda else torch.float32}")

    quant = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
        if use_4bit
        else None
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    # Strings (not torch.dtype objects) so the config stays JSON-serialisable.
    model_kwargs = {"dtype": ("bfloat16" if bf16 else "float16") if cuda else "float32"}
    if cuda:
        model_kwargs["device_map"] = "auto"

    config = SFTConfig(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,  # float < 1 = fraction of total steps
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="no",  # we save the final adapter ourselves
        max_length=args.max_length,
        completion_only_loss=True,  # loss only on the JSON answer
        packing=False,
        gradient_checkpointing=cuda,
        fp16=cuda and not bf16,
        bf16=bf16,
        report_to="none",
        seed=args.seed,
        model_init_kwargs=model_kwargs,
    )

    trainer = SFTTrainer(
        model=args.base_model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        quantization_config=quant,
    )
    trainer.model.print_trainable_parameters()

    result = trainer.train()
    eval_metrics = trainer.evaluate()
    print("train:", result.metrics)
    print("val:", eval_metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(args.output_dir)  # adapter only
    tokenizer.save_pretrained(args.output_dir)
    run_info = {
        "base_model": args.base_model,
        "args": {k: str(v) for k, v in vars(args).items()},
        "train_metrics": result.metrics,
        "val_metrics": eval_metrics,
        "log_history": trainer.state.log_history,
    }
    (args.output_dir / "training_run.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")
    print(f"saved adapter to {args.output_dir}")

    if args.push_to_hub:
        trainer.model.push_to_hub(args.hub_repo, private=args.private,
                                  commit_message="Upload doomscroll LoRA adapter")
        tokenizer.push_to_hub(args.hub_repo, private=args.private)
        print(f"pushed to https://huggingface.co/{args.hub_repo}")


def _hf_logged_in() -> bool:
    try:
        from huggingface_hub import whoami

        whoami()
        return True
    except Exception:  # noqa: BLE001 - any failure means "not logged in"
        return False


if __name__ == "__main__":
    main()
