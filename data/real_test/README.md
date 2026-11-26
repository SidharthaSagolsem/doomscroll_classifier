# Real test set: labelling guide

The training data is synthetic (Gemini-written). This folder is for **200+ real posts**, labelled by hand, to measure how the model does on actual social media. **Never train on these posts.** They are for testing only.

## Collecting posts

- Aim for 200+ posts, at least 40 per label. Rough target: 50 ragebait, 50 brainrot, 60 neutral, 50 educational.
- Mix platforms: X/Twitter, Reddit, TikTok/Reels/Shorts captions or transcripts, YouTube comments, Instagram, Facebook, LinkedIn.
- Copy the text only. Remove usernames, @handles, names of private people, phone numbers and emails. Keep public figures' names.
- Skip posts that are mostly an image or video with no text, and non-English posts (for now).
- Don't cherry-pick only obvious examples. Grab posts as you scroll, then fill gaps for rare labels.
- Record the platform and the date you collected it, not the URL (URLs can identify people).

## File format

Copy `template.csv` to `labels.csv` (git-ignored, so the posts stay on your machine) and fill one row per post:

| column | meaning |
|---|---|
| `id` | r001, r002, ... |
| `platform` | twitter / reddit / tiktok / youtube / instagram / facebook / linkedin / other |
| `collected` | date, YYYY-MM-DD |
| `text` | the post text (anonymised) |
| `label_a` | your label |
| `label_b` | a second person's label, done **without seeing** `label_a` |
| `final_label` | the agreed label after discussing disagreements |
| `notes` | anything ambiguous |

Labels must be spelled exactly: `high-dopamine ragebait`, `brainrot/low-attention`, `neutral`, `educational`.

## Label rules

Ask in order and stop at the first "yes":

1. **Is the main purpose to teach, explain or inform** (even casually, even with slang)? → `educational`
2. **Is it built to make people angry or start a fight?** For example, sweeping insults about a group, "unpopular opinion" provocations, outrage at a strawman, conspiracy bait or moral panic. → `high-dopamine ragebait`
3. **Is it attention-grabbing noise with no real content?** For example, meme slang soup, "wait for the end", ASMR/satisfying bait, nonsense POV, "only 1% can see this". → `brainrot/low-attention`
4. Otherwise (life updates, questions, reviews, announcements, calm opinions) → `neutral`

Tricky cases:
- **A strong opinion isn't automatically ragebait.** "I think the ref got that call wrong" is neutral. "Refs are paid to rig games for Taylor Swift" is ragebait.
- **Brainrot slang inside a real explanation** is educational (rule 1 wins).
- **News headlines:** neutral if factual; ragebait if framed to enrage ("You won't BELIEVE what X just did").
- **Truly mixed posts:** pick the label for the post's *main* effect on a reader, and write a note.

## Workflow

1. Collect and anonymise posts into `labels.csv`, filling `label_a` as you go.
2. Hide `label_a` and have someone else fill `label_b` (or relabel it yourself a week later, without looking).
3. Check agreement:
   ```powershell
   python agreement.py data/real_test/labels.csv --disagreements data/real_test/disagreements.csv
   ```
   A Cohen's kappa above 0.6 is fine. Below that, the label rules above need sharpening.
4. Resolve the disagreements and fill `final_label`.
5. Convert to JSONL and evaluate:
   ```powershell
   python -c "import csv,json; rows=[r for r in csv.DictReader(open('data/real_test/labels.csv',encoding='utf-8')) if r['final_label']]; open('data/real_test/real_test.jsonl','w',encoding='utf-8').write(''.join(json.dumps({'id':r['id'],'input_text':r['text'],'label':r['final_label']})+'\n' for r in rows))"
   ```
   Then on Colab:
   ```
   python evaluate.py --model <user>/doomscroll-qwen2.5-1.5b-lora --test data/real_test/real_test.jsonl --out-dir results_real
   ```
