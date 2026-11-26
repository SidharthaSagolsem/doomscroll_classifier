"""Labels, prompt format and output parsing.

Training and inference MUST use the same prompt, so both import it from
here. The model is trained to answer with exactly one JSON object:

    {"label": "<one of LABELS>", "reasoning": "<one sentence>"}
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

LABELS: list[str] = [
    "high-dopamine ragebait",
    "brainrot/low-attention",
    "neutral",
    "educational",
]

# Used in metrics when the model's output can't be mapped to any label.
UNPARSED = "unparsed"

SYSTEM_PROMPT = """You classify social media posts into exactly one label.

Labels:
- "high-dopamine ragebait": engineered to provoke outrage, tribal anger or fights in the comments (inflammatory generalisations, strawmen, "unpopular opinion" hot takes, conspiracy bait).
- "brainrot/low-attention": hyper-stimulating meme soup with no information (absurdist slang like skibidi/gyatt/rizz, "wait for the end", ASMR/satisfying clips, POV nonsense).
- "neutral": ordinary low-stakes posts (life updates, questions, reviews, scores) that are not built to provoke or numb.
- "educational": the main purpose is to explain, inform or teach, even if written casually.

Reply with ONLY a JSON object, no markdown:
{"label": "<one of the four labels>", "reasoning": "<one short sentence>"}"""


def build_prompt_messages(text: str) -> list[dict[str, str]]:
    """Chat messages for one post (system + user), without the answer."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Post:\n{text.strip()}"},
    ]


def format_target(label: str, reasoning: str) -> str:
    """The exact assistant answer the model is trained to produce."""
    if label not in LABELS:
        raise ValueError(f"Unknown label: {label!r}")
    return json.dumps({"label": label, "reasoning": reasoning.strip()}, ensure_ascii=False)


def to_prompt_completion(row: dict) -> dict:
    """Convert a dataset row to TRL's conversational prompt-completion format.

    With this format TRL computes the loss ONLY on the completion (the JSON
    answer), not on the system prompt or the post. Training on the whole
    sequence is one likely cause of the v0 "Post Post Post..." output.
    """
    return {
        "prompt": build_prompt_messages(row["input_text"]),
        "completion": [
            {"role": "assistant", "content": format_target(row["label"], row["reasoning"])}
        ],
    }


@dataclass
class Prediction:
    label: str  # one of LABELS, or UNPARSED
    reasoning: str
    valid_json: bool  # True only if the raw output was a JSON object with a valid label


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def parse_output(raw: str) -> Prediction:
    """Turn raw model text into a Prediction. Never raises.

    1. Try every {...} block as JSON; accept the first with a valid label.
    2. Otherwise, if exactly one label name appears in the text, use it
       (valid_json=False, so the metrics can report format errors).
    3. Otherwise the label is UNPARSED.
    """
    text = _FENCE_RE.sub("", (raw or "").strip())

    for match in _OBJECT_RE.finditer(text):
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            label = str(obj.get("label", "")).strip().lower()
            if label in LABELS:
                return Prediction(label, str(obj.get("reasoning", "")).strip(), True)

    lowered = text.lower()
    found = [label for label in LABELS if label in lowered]
    if len(found) == 1:
        return Prediction(found[0], "", False)
    return Prediction(UNPARSED, "", False)
