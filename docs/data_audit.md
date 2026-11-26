# Data audit: `dataset.jsonl` (160 synthetic posts)

Audit done by reading all 160 rows, plus automatic checks in `prepare_data.py`.
Decisions that change the data live in [`data/audit.csv`](../data/audit.csv), so they are applied the same way every run.

## Summary

| Check | Result |
|---|---|
| Rows | 160 (Gemini-generated, 16 batches of 10) |
| Labels | ragebait 48, brainrot 42, neutral 36, educational 34 |
| Malformed rows / unknown labels | 0 |
| Exact duplicates (after normalising case, spacing and quotes) | 0 |
| Near-duplicate clusters | 9 clusters covering 20 rows (6 found automatically, 3 added by hand) |
| Clear label errors | none found. 4 borderline rows are documented and kept |

## 1. Near-duplicates (the main problem)

Each Gemini call was nudged toward one theme, so later batches often re-wrote earlier posts. Examples:

- 3 near-identical "Tried the new ramen spot downtown!" reviews (rows 3, 43, 103)
- two 529-plan explainers (25, 39), two "Leo's first day" posts (27, 92), two "elite climate jets" rants (60, 140)
- paraphrased takes that share few exact words: "gentle parenting" ×3 (20, 30, 98), "iPads at restaurants" ×2 (14, 90), "stay-at-home wife" ×2 (34, 94)

**Why it matters:** if one copy is in train and another in test, the test question has effectively been seen before, so the test score goes up without the model getting any better.

**Fix:** `prepare_data.py` groups posts whose meaningful-word overlap (Jaccard) is ≥ 0.2. The paraphrases above are linked by hand with the `group_with` column in `audit.csv`. The split then keeps every group on one side.

## 2. Borderline labels (kept, noted in `audit.csv`)

| Row | Post | Kept as | Why it's borderline |
|---|---|---|---|
| 4 | lasagna cooked in a kitchen sink | ragebait | "disgust cooking" is outrage bait, but also low-attention spectacle |
| 59 | "ONLY 1% CAN DO THIS!" find-the-phone | brainrot | engagement bait without outrage |
| 75 | ASMR clip that turns into "the matrix is real" | brainrot | mixes conspiracy (ragebait) with ASMR |
| 125 | SQL task-manager template walkthrough | educational | partly self-promotion |

These are the posts most likely to show up in the confusion matrix. If real posts often look like these, tighten the label definitions in `doomscroll/schema.py` **and** in `data/real_test/README.md`.

## 3. Limits of the dataset

- **Small:** 160 rows gives a test split of about 23 posts. One flipped prediction moves accuracy by about 4 points, so read the bootstrap confidence intervals, not just the point score. Generating more data (`python generate_dataset.py --num-batches 35` gives about +350 rows) is the cheapest improvement.
- **Synthetic and stylised:** brainrot posts nearly all contain skibidi/gyatt/rizz/POV, and ragebait often starts with "Unpopular opinion" or "Unbelievable". A TF-IDF model can pick up these surface cues, which is why its score is high. Real posts are messier, which is what the [real test set](../data/real_test/README.md) is for.
- **Topic and label are linked:** each label covers similar topics (parenting, food, AI, climate, sports), which is good. But every batch was generated with a label mix, so no topic is exclusive to one label.

## Reproduce

```powershell
python prepare_data.py          # prints counts; writes data/processed/stats.json
```
