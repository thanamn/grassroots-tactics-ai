"""Call DeepSeek with a tactical-explainer prompt, return parsed explanation.

Uses the openai SDK pointed at DeepSeek's OpenAI-compatible API endpoint.
JSON output is requested via response_format={"type": "json_object"} and
validated against the required keys after parsing.

Usage (CLI):
    python -m src.explainer --input data/cache/sample_metrics.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import openai

from src.config import CACHE_DIR, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from prompts.tactical_explainer import PROMPT_VERSION, build_messages

Lang = Literal["en", "th"]

# DeepSeek free/paid tier can return 429 (rate limit) or 503 (overloaded)
# under load. Back off exponentially; re-raise on permanent errors.
RETRY_DELAYS = (2.0, 6.0, 18.0)   # ~26 s total worst-case
RETRY_STATUS_HINTS = ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                      "DEADLINE_EXCEEDED", "INTERNAL")


def _is_transient(exc: Exception) -> bool:
    """True if the exception looks like a transient API hiccup worth retrying."""
    # openai SDK raises APIStatusError with a status_code attribute
    if hasattr(exc, "status_code") and exc.status_code in (429, 502, 503):
        return True
    s = repr(exc)
    return any(hint in s for hint in RETRY_STATUS_HINTS)


def explain(metrics: dict, phase_context: str = "general open play",
            lang: Lang = "en", max_tokens: int = 600) -> dict:
    """Send metrics → DeepSeek → parsed explanation dict."""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "DEEPSEEK_API_KEY not set. Copy .env.example to .env and add your key."
        )

    system, user_msg = build_messages(metrics, phase_context=phase_context, lang=lang)
    # DeepSeek requires the word "json" in the prompt when response_format=json_object.
    user_msg = user_msg + "\n\nRespond with a JSON object only."

    client = openai.OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url="https://api.deepseek.com",
    )

    # Retry on transient 5xx / rate-limit errors. Permanent errors
    # (auth, quota) propagate immediately.
    attempts = len(RETRY_DELAYS) + 1
    last_exc: Exception | None = None
    response = None
    for i in range(attempts):
        try:
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user_msg},
                ],
                response_format={"type": "json_object"},
                max_tokens=max_tokens,
                temperature=0.6,
            )
            break
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if i == attempts - 1 or not _is_transient(e):
                raise
            time.sleep(RETRY_DELAYS[i])
    else:  # pragma: no cover
        raise last_exc  # type: ignore[misc]

    raw = (response.choices[0].message.content or "") if response else ""
    parsed = json.loads(raw)

    required = ("headline", "implication", "coaching_cue")
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"DeepSeek response missing keys: {missing}. Raw: {raw[:200]}")

    return {
        "clip_id": metrics.get("clip_id", ""),
        "phase_context": phase_context,
        "language": lang,
        "model": DEEPSEEK_MODEL,
        "prompt_version": PROMPT_VERSION,
        "headline":     parsed["headline"],
        "implication":  parsed["implication"],
        "coaching_cue": parsed["coaching_cue"],
        "raw_response": raw,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to metrics JSON")
    parser.add_argument("--phase", default="general open play")
    parser.add_argument("--lang", default="en", choices=["en", "th"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    metrics = json.loads(Path(args.input).read_text())
    result = explain(metrics, phase_context=args.phase, lang=args.lang)

    out_path = (Path(args.output) if args.output
                else CACHE_DIR / f"{metrics['clip_id']}_explanation_{args.lang}.json")
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("Headline:    ", result["headline"])
    print("Implication: ", result["implication"])
    print("Coaching cue:", result["coaching_cue"])


if __name__ == "__main__":
    main()
