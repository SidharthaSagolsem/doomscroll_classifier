# Doomscroll & Ragebait Classifier

A QLoRA fine-tune of **Qwen2.5-1.5B-Instruct** that reads a social media post and answers with JSON:

```json
{"label": "brainrot/low-attention", "reasoning": "Meme-slang soup with no information."}
```

Labels: `high-dopamine ragebait` · `brainrot/low-attention` · `neutral` · `educational`

The training data is synthetic, generated with the Gemini API.

**Model:** [SidharthaSgl/doomscroll-qwen2.5-1.5b-lora](https://huggingface.co/SidharthaSgl/doomscroll-qwen2.5-1.5b-lora) (LoRA adapter for `Qwen/Qwen2.5-1.5B-Instruct`)

```powershell
python predict.py --model SidharthaSgl/doomscroll-qwen2.5-1.5b-lora --text "POV: the rizzler at 3am fr fr"
```

## Results

### Real posts (the number that matters)

249 real YouTube comments and Shorts/video titles from 2020–2025, taken from [AmaanP314/youtube-comment-sentiment](https://huggingface.co/datasets/AmaanP314/youtube-comment-sentiment) (CC-BY-SA-4.0) and labelled by hand. None were seen in training. Same adapter as below; evaluated 2026-11-27.

| System | Macro-F1 | 95% CI | Accuracy | 95% CI | Valid JSON |
|---|---|---|---|---|---|
| **Fine-tuned (QLoRA)** | **0.737** | 0.675–0.791 | **0.743** | 0.687–0.799 | **100%** |
| Zero-shot Qwen2.5-1.5B-Instruct | 0.416 | 0.356–0.474 | 0.486 | 0.426–0.546 | 99.2% |
| TF-IDF + logistic regression | 0.527 | 0.467–0.582 | 0.538 | 0.478–0.594 | n/a |

Labels: Claude labelled all 249 posts. A human independently re-labelled 87 of them, with 74% raw agreement and Cohen's κ 0.64 ("substantial"). The disagreements were resolved with the rules in [data/real_test/README.md](data/real_test/README.md). On just those 87 human-checked posts the ranking is the same: fine-tuned 0.718 (CI 0.608–0.812), TF-IDF 0.518, zero-shot 0.443.

What this shows:
- **Fine-tuning generalises; keyword matching doesn't.** TF-IDF falls from 0.91 on synthetic posts to 0.53 on real ones. The fine-tuned model falls less (0.92 → 0.74) and now clearly beats TF-IDF (the CIs don't overlap).
- **Best at ragebait (F1 0.82) and brainrot (0.80).** Zero-shot Qwen calls 44 of 58 brainrot posts "ragebait", and fine-tuning fixes that.
- **Weakest at neutral (0.66) and educational (0.67).** The model treats posts *about* learning ("great explanation, thanks!") as educational, and short how-to tips ("you can't log out on a phone, remove the Google account instead") as brainrot. More real-style neutral and educational training examples are the next fix.

### Synthetic test set

Held-out synthetic test set: 23 posts, with near-duplicates kept out of train. Macro-F1 and accuracy have 95% bootstrap CIs (1000 resamples). Run of 2026-11-26 on a Colab T4: 3 epochs, LoRA r=16, about 2 minutes of training.

| System | Macro-F1 | 95% CI | Accuracy | 95% CI | Valid JSON |
|---|---|---|---|---|---|
| **Fine-tuned (QLoRA)** | **0.923** | 0.775–1.000 | **0.913** | 0.783–1.000 | **100%** |
| Zero-shot Qwen2.5-1.5B-Instruct | 0.616 | 0.458–0.714 | 0.652 | 0.478–0.826 | 91.3% |
| TF-IDF + logistic regression | 0.908 | 0.726–1.000 | 0.913 | 0.783–1.000 | n/a |

TF-IDF 5-fold group cross-validation on train+val: macro-F1 0.925 ± 0.014.

What this shows:
- **Fine-tuning works.** The model beats its own zero-shot version by a clear margin (the CIs don't overlap). The biggest gain is brainrot: zero-shot F1 0.00 → fine-tuned 0.86. Every output is valid JSON, unlike v0's `Post Post Post...`.
- **It doesn't (yet) beat TF-IDF.** 0.923 vs 0.908 is well inside the noise of a 23-post test set. The synthetic posts are easy to separate by surface words, so a cheap baseline already does well.
- **Remaining errors:** 2 of 7 ragebait posts were called brainrot (the "lasagna in the sink" post flagged as borderline in the data audit, and a tipping-culture rant).

With a test set this small, differences inside the confidence intervals aren't meaningful. See [docs/data_audit.md](docs/data_audit.md) for the data's limits.

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
