# Doomscroll & Ragebait Classifier

A QLoRA fine-tune of **Qwen2.5-1.5B-Instruct** that reads a social media post and answers with JSON:

```json
{"label": "brainrot/low-attention", "reasoning": "Meme-slang soup with no information."}
```

Labels: `high-dopamine ragebait` · `brainrot/low-attention` · `neutral` · `educational`

The training data is synthetic, generated with the Gemini API.

**Model:** _not published yet. A link to the Hugging Face adapter goes here after training._

## Results

Held-out synthetic test set (about 23 posts, near-duplicates kept out of train). Macro-F1 with 95% bootstrap CI.

| System | Macro-F1 | 95% CI | Accuracy | Valid JSON |
|---|---|---|---|---|
| Fine-tuned (QLoRA) | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| Zero-shot Qwen2.5-1.5B-Instruct | _tbd_ | _tbd_ | _tbd_ | _tbd_ |
| TF-IDF + logistic regression | _tbd_ | _tbd_ | _tbd_ | n/a |

With a test set this small, differences inside the confidence intervals aren't meaningful. See [docs/data_audit.md](docs/data_audit.md) for the data's limits, and [data/real_test/](data/real_test/README.md) for the real-post test set in progress.

## Pipeline

```
generate_dataset.py ─► dataset.jsonl ─► prepare_data.py ─► data/processed/{train,val,test}.jsonl
                                              ▲                    │
                                     data/audit.csv                ├─► train.py ─► LoRA adapter ─► Hugging Face Hub
                                                                   └─► evaluate.py (finetuned / zero-shot / tfidf)
                                                                         └─► results/metrics.json, report.md
```

| File | What it does |
|---|---|
| `generate_dataset.py` | Generates labelled synthetic posts with Gemini (rate-limited for the free tier) |
| `prepare_data.py` | Validates, de-duplicates, applies `data/audit.csv`, groups near-duplicates, makes a stratified split |
| `train.py` | QLoRA fine-tune with TRL `SFTTrainer` + PEFT (4-bit on CUDA). Loss only on the JSON answer. `--push-to-hub` |
| `predict.py` | Classifies posts with an adapter (local or Hub) or a base model |
| `evaluate.py` | Macro-F1 and accuracy with bootstrap CIs, per-class scores, confusion matrix, valid-JSON rate |
| `agreement.py` | Cohen's kappa between two annotators for the real test set |
| `doomscroll/` | Shared code: labels and prompt (`schema.py`), data utils, metrics, TF-IDF baseline |
| `notebooks/train_colab.ipynb` | One-click Colab run: clone `main` → prepare → train → push → evaluate → download results |

## Setup (local, CPU)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

`requirements-train.txt` (torch stack, pinned: transformers 5.24.1, trl 1.19.0, peft 0.22.0) is for the Colab GPU only.

### Generate more data

Copy `.env.example` to `.env` and add a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey):

```powershell
python generate_dataset.py --num-batches 35
```

### Prepare data and run the CPU baseline

```powershell
python prepare_data.py
python evaluate.py --systems tfidf
```

## Training (Colab)

1. Create a Hugging Face **write** token (huggingface.co → Settings → Access Tokens).
2. Open `notebooks/train_colab.ipynb` in Colab (File → Open notebook → GitHub).
3. Runtime → Change runtime type → **T4 GPU**.
4. Secrets (key icon): add `HF_TOKEN` and turn on notebook access. Set `HF_REPO` in the first cell.
5. Runtime → Run all. The adapter is pushed to the Hub and `results.zip` is downloaded.

## History

The first model (Sept 2026) produced `brain Post Post Post...`, was never evaluated, and was lost when the Colab runtime ended. See [docs/run_v0_notes.md](docs/run_v0_notes.md) for what went wrong and what this pipeline changes.
