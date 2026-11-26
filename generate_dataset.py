"""
Doomscroll & Ragebait Classifier — Synthetic SFT Dataset Generator (Gemini)
=============================================================================

Generates a labeled dataset of (social_media_text -> label + reasoning)
examples for fine-tuning an open-weight LLM, using the official Google
Gemini API (`google-genai` SDK) with `response_schema` to guarantee
schema-valid JSON — no credit card required, works on the Google AI
Studio free tier.

Usage:
    export GEMINI_API_KEY="AIza..."      # from aistudio.google.com/apikey
    pip install -r requirements.txt
    python generate_dataset.py --num-batches 50

Each batch request generates 10 diverse examples (one API call = 10 rows),
and every example is flushed to disk the moment it's validated, so a
crash at row 500 loses nothing before it.

FREE TIER RATE LIMITING
------------------------
The Google AI Studio free tier for gemini-2.5-flash currently allows
15 requests per minute (RPM). Free-tier limits are configuration, not a
fixed spec — Google can and does change them — so if you're on a paid
tier or the limit has changed, adjust `--rpm` accordingly; the script
always re-checks nothing on its own, so verify the current number at
https://ai.google.dev/gemini-api/docs/rate-limits before a big run.

Rather than relying on a plain asyncio.Semaphore (which caps *concurrency*
but not *rate*), this script uses a token-bucket style RateLimiter that
enforces "no more than N requests started in any rolling 60s window" —
which is what actually matters for RPM limits. tenacity's exponential
backoff is layered on top to absorb the occasional 429 that slips through
(e.g. from clock-edge effects or other traffic on the same key).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import List, Literal
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

MODEL = "gemini-3.5-flash"
EXAMPLES_PER_BATCH = 10
OUTPUT_PATH = Path("dataset.jsonl")
FREE_TIER_RPM = 15  # Google AI Studio free tier limit for gemini-2.5-flash

LABELS = [
    "high-dopamine ragebait",
    "brainrot/low-attention",
    "neutral",
    "educational",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("dataset_gen")


# --------------------------------------------------------------------------- #
# Schema (Pydantic — used for validation; a JSON Schema dict is derived
# below for response_schema, since Gemini's structured-output schema
# support is a constrained subset of full JSON Schema and doesn't like
# some Pydantic-emitted keywords like `$defs`/`additionalProperties`).
# --------------------------------------------------------------------------- #

class Example(BaseModel):
    input_text: str = Field(
        description=(
            "A realistic, synthetic social media post, short-form video "
            "transcript, or comment-section excerpt."
        )
    )
    label: Literal[
        "high-dopamine ragebait",
        "brainrot/low-attention",
        "neutral",
        "educational",
    ]
    reasoning: str = Field(
        description="1-2 sentence explanation of why the text fits the label."
    )


class ExampleBatch(BaseModel):
    examples: List[Example] = Field(
        description=f"Exactly {EXAMPLES_PER_BATCH} diverse, non-repetitive examples."
    )


# Hand-rolled schema (flat, no $defs) — the Gemini SDK accepts a Pydantic
# model directly for response_schema too, but spelling it out explicitly
# keeps the request payload transparent and avoids relying on SDK-version-
# specific Pydantic-to-schema conversion quirks.
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "examples": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "input_text": {"type": "STRING"},
                    "label": {
                        "type": "STRING",
                        "enum": LABELS,
                    },
                    "reasoning": {"type": "STRING"},
                },
                "required": ["input_text", "label", "reasoning"],
            },
        }
    },
    "required": ["examples"],
}


# --------------------------------------------------------------------------- #
# Prompting
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """You are a synthetic-data generator helping build a training \
set for a "Doomscroll & Ragebait Classifier" — a small open-weight model that \
will read a short social media post, comment, or video transcript and \
classify it.

Generate realistic, highly diverse EXAMPLE TEXT that authentically mimics \
real internet writing across platforms (X/Twitter, TikTok transcripts, \
Instagram Reels captions, YouTube Shorts, Reddit comments, Facebook rage \
posts). Where appropriate, use authentic Gen-Z/internet vernacular \
(lowercase everything, run-on sentences, slang like "ngl", "fr fr", \
"no cap", "it's giving...", "the way that...", "not me ___ing", excessive \
punctuation!!, ALL CAPS for emphasis, emoji, hashtags, ratio bait, etc.) — \
but only where it is natural for that label; do not force slang into \
`educational` or `neutral` content where it wouldn't realistically appear.

## Label definitions

1. "high-dopamine ragebait" — content engineered to provoke outrage, \
moral panic, tribal anger, or defensive engagement. Polarizing claims, \
strawmanning, "you won't believe what X just said", inflammatory \
generalizations about groups, engagement-farming hot takes, comment-bait \
questions designed to start fights.

2. "brainrot/low-attention" — low-substance, hyper-stimulating, meme-soup \
content optimized for zero-effort dopamine hits rather than outrage. \
Nonsensical trends, unrelated stitched clips, chaotic captions, \
"skibidi"/"gyatt"/"rizz"-style absurdist slang, pointless POV formats, \
oddly-specific niche humor with no informational value.

3. "neutral" — ordinary, low-stakes social content: life updates, \
mundane observations, product photos, weather chat, sports scores, \
wholesome-but-unremarkable posts. Not engineered to provoke or numb.

4. "educational" — content whose primary purpose is to inform, explain, \
or teach, even if casually written. Science facts, how-tos, history \
threads, financial literacy tips, coding explainers, etc.

## Diversity requirements for every batch of 10

- Vary the label distribution — don't put all 10 in one bucket.
- Vary platform/format (tweet, TikTok transcript, YouTube comment, \
Reddit post, Instagram caption, Facebook post, LinkedIn post gone wrong, etc.)
- Vary topic (politics, celebrity drama, tech, relationships, food, \
sports, finance, health, gaming, workplace, parenting, etc.)
- Vary length (a one-line tweet vs. a multi-sentence transcript excerpt).
- Avoid reusing phrasing, hooks, or subject matter from earlier examples \
in this same batch.
- Make ragebait and brainrot examples feel like things that could \
genuinely go viral — specific, punchy, textured — not generic placeholders.

Output ONLY the structured JSON — no preamble, no markdown fences."""


def build_user_prompt(batch_index: int) -> str:
    """A rotating theme nudge to reduce repetition across independent calls."""
    theme_pool = [
        "celebrity/pop-culture drama",
        "personal finance and money advice",
        "relationship and dating discourse",
        "politics and current events",
        "food, cooking, and restaurant reviews",
        "fitness, health, and wellness",
        "gaming and esports",
        "workplace and career content",
        "true crime and conspiracy-adjacent content",
        "AI, tech, and internet culture",
        "parenting and family life",
        "sports commentary",
        "climate and environment",
        "self-improvement and productivity",
        "school/college life",
    ]
    theme = random.choice(theme_pool)
    return (
        f"Generate a fresh batch of {EXAMPLES_PER_BATCH} diverse training "
        f"examples for the classifier. Lean (but don't limit yourself "
        f"entirely) toward the theme area of '{theme}' for variety across "
        f"batches, while still mixing labels, platforms, and tones within "
        f"the batch. Batch reference id: {batch_index}."
    )


# --------------------------------------------------------------------------- #
# Rate limiter — enforces N requests / rolling 60s window (RPM), which is
# what the free tier actually measures. A bare asyncio.Semaphore only caps
# concurrency, not throughput over time, so it can't hold you under RPM by
# itself once each call is fast — this is layered on top of one.
# --------------------------------------------------------------------------- #

class RollingWindowRateLimiter:
    """Blocks callers so that no more than `max_calls` acquisitions happen
    in any rolling `period` seconds. Safe under asyncio concurrency."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                # Drop timestamps outside the rolling window.
                while self._timestamps and now - self._timestamps[0] >= self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return

                # Window is full — figure out how long until the oldest
                # timestamp falls out of the window.
                sleep_for = self.period - (now - self._timestamps[0]) + 0.05

            log.debug("Rate limit reached, sleeping %.2fs", sleep_for)
            await asyncio.sleep(sleep_for)


# --------------------------------------------------------------------------- #
# API call with retries
# --------------------------------------------------------------------------- #

# ClientError covers 4xx including 429 (rate limit exceeded); ServerError
# covers transient 5xx. Both are worth retrying with backoff — a 429 means
# we should back off even though our own limiter should mostly prevent it
# (e.g. clock skew, or another process/tab sharing the same API key).
RETRYABLE_EXCEPTIONS = (ClientError, ServerError, APIError)


@retry(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=2, min=4, max=90),
    stop=stop_after_attempt(6),
    reraise=True,
)
async def _call_model(
    client: genai.Client, limiter: RollingWindowRateLimiter, batch_index: int
) -> ExampleBatch:
    await limiter.acquire()  # throttle to stay under free-tier RPM

    response = await client.aio.models.generate_content(
        model=MODEL,
        contents=build_user_prompt(batch_index),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=1.0,
        ),
    )

    if not response.text:
        raise ValueError(
            f"Batch {batch_index}: empty response "
            f"(finish_reason={getattr(response.candidates[0], 'finish_reason', '?') if response.candidates else '?'})"
        )

    payload = json.loads(response.text)
    return ExampleBatch.model_validate(payload)


# --------------------------------------------------------------------------- #
# Incremental writer (safe under concurrency)
# --------------------------------------------------------------------------- #

class JsonlWriter:
    """Append-only JSONL writer guarded by an asyncio lock so concurrent
    coroutines never interleave partial writes."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()
        self._written = 0

    async def write_examples(self, examples: List[Example]) -> None:
        async with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                for ex in examples:
                    f.write(json.dumps(ex.model_dump(), ensure_ascii=False) + "\n")
                    self._written += 1
            log.info(
                "Wrote %d examples (total so far: %d)", len(examples), self._written
            )

    @property
    def written(self) -> int:
        return self._written


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #

async def generate_one_batch(
    client: genai.Client,
    limiter: RollingWindowRateLimiter,
    writer: JsonlWriter,
    batch_index: int,
) -> None:
    try:
        # Force the script to pause for 5 seconds to avoid 429 rate limits
        await asyncio.sleep(5)
        batch = await _call_model(client, limiter, batch_index)
    except Exception as exc:  # noqa: BLE001 — log and move on, don't kill the run
        log.error("Batch %d failed permanently: %s", batch_index, exc)
        return

    if len(batch.examples) != EXAMPLES_PER_BATCH:
        log.warning(
            "Batch %d returned %d examples (expected %d) — saving anyway.",
            batch_index,
            len(batch.examples),
            EXAMPLES_PER_BATCH,
        )

    await writer.write_examples(batch.examples)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

async def run(num_batches: int, rpm: int, output_path: Path) -> None:
    client = genai.Client()  # reads GEMINI_API_KEY (or GOOGLE_API_KEY) from env
    limiter = RollingWindowRateLimiter(max_calls=rpm, period=60.0)
    writer = JsonlWriter(output_path)

    log.info(
        "Starting generation: %d batches x %d examples/batch = %d target rows "
        "(rate-limited to %d req/min) -> %s",
        num_batches,
        EXAMPLES_PER_BATCH,
        num_batches * EXAMPLES_PER_BATCH,
        rpm,
        output_path,
    )

    # We still launch all tasks concurrently — the RollingWindowRateLimiter
    # (not a concurrency cap) is what actually throttles them to `rpm`
    # requests/minute, so a handful of tasks can be "in flight" waiting on
    # the limiter without over-driving the API.
    tasks = [
        generate_one_batch(client, limiter, writer, i) for i in range(num_batches)
    ]
    await asyncio.gather(*tasks)

    log.info("Done. %d total examples written to %s", writer.written, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help=f"Number of batches to generate ({EXAMPLES_PER_BATCH} examples each). "
        f"Default: 50 (=500 examples).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=FREE_TIER_RPM,
        help=f"Max requests per rolling 60s window. Default: {FREE_TIER_RPM} "
        f"(Google AI Studio free tier for {MODEL} at the time of writing — "
        f"verify at https://ai.google.dev/gemini-api/docs/rate-limits).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output JSONL path. Default: {OUTPUT_PATH}",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(run(args.num_batches, args.rpm, args.output))
    except KeyboardInterrupt:
        log.warning("Interrupted by user — partial dataset.jsonl has been kept.")