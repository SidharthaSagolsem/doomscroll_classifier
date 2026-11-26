# Run v0 (Sept 2026 Colab) post-mortem

## What happened

- Qwen2.5-1.5B-Instruct was fine-tuned on Colab with the 160-row `dataset.jsonl`.
- Inference produced degenerate output such as `brain Post Post Post ...` instead of a label.
- There was no held-out test set and no metric, so there was no way to tell whether it learned anything.
- The trained model was saved only to Colab's `/content`, which is wiped when the runtime ends. That copy was lost.
  - A merged full-weight export (`model.safetensors`, about 3 GB, dated 2026-09-05) survived on a local machine. It isn't committed (weights don't belong in git) and isn't used by the new pipeline. For a v0 score, upload it to Colab and run `evaluate.py --base-model <that folder> --systems zero-shot`. v0 was trained with a different prompt, so expect a low valid-JSON rate.
- The original training notebook was not kept, so the causes below are informed guesses based on the symptoms.

## Likely causes of "Post Post Post"

| Symptom | Likely cause | Fix in the new pipeline |
|---|---|---|
| Model repeats words from the prompt (`Post`) | Loss computed on the whole sequence, including the prompt, so the model learned to continue the prompt | `train.py` uses TRL prompt-completion data with `completion_only_loss=True`: loss is on the JSON answer only |
| Never stops generating | Training text without the chat template's end-of-turn token, so the model never learned to emit `<|im_end|>` | TRL applies Qwen's chat template, which ends each answer with `<|im_end|>` |
| Prompt at inference ≠ prompt at training | Hand-written prompt strings in two places | `doomscroll/schema.py` is the one source for the prompt; train, predict and evaluate all import it |
| Output not parseable | No output contract | The model is trained to emit `{"label", "reasoning"}` JSON; `parse_output` handles fences and chatter and reports a valid-JSON rate |
| Lost model | Saved only to `/content` | `train.py --push-to-hub` uploads the adapter to Hugging Face at the end of training |
| No idea if it worked | No evaluation | `prepare_data.py` holds out a test split; `evaluate.py` reports macro-F1 with bootstrap CIs vs zero-shot and TF-IDF baselines |

## Lessons

1. Save anything you can't regenerate in 5 minutes off the VM (Hugging Face Hub or Drive) before the session ends.
2. Decide the metric and the baselines before training.
3. Smoke-test generation on 2 or 3 examples right after training (notebook step 7) before running the full evaluation.
