"""Call Gemini with a tactical-explainer prompt, return parsed explanation.

Uses the modern google-genai SDK (NOT the deprecated google-generativeai).
Gemini's response_schema feature forces strictly-structured JSON output,
so we don't need fragile regex parsing of free-form replies.

Usage (CLI):
    python -m src.explainer --input data/cache/sample_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

from src.config import CACHE_DIR, GEMINI_API_KEY, GEMINI_MODEL
from prompts.tactical_explainer import PROMPT_VERSION, build_messages

Lang = Literal["en", "th"]


def explain(metrics: dict, phase_context: str = "general open play",
            lang: Lang = "en", max_tokens: int = 600) -> dict:
    """Send metrics → Gemini → parsed explanation dict."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key. "
            "Get a free key at https://aistudio.google.com/apikey"
        )

    # Lazy import so the rest of the project still imports without google-genai installed
    from google import genai
    from google.genai import types

    # Build schema using types.Schema — plain dict doesn't enforce structured output reliably
    schema = types.Schema(
        type=types.Type.OBJECT,
        properties={
            "headline":     types.Schema(type=types.Type.STRING),
            "implication":  types.Schema(type=types.Type.STRING),
            "coaching_cue": types.Schema(type=types.Type.STRING),
        },
        required=["headline", "implication", "coaching_cue"],
    )

    system, user_msg = build_messages(metrics, phase_context=phase_context, lang=lang)
    client = genai.Client(api_key=GEMINI_API_KEY)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            # Gemini 2.5 Flash uses thinking tokens by default; disable them so
            # the full token budget goes to the actual JSON output.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            max_output_tokens=max_tokens,
            temperature=0.6,
        ),
    )
    raw = response.text or ""
    parsed = json.loads(raw)   # guaranteed valid JSON because of response_schema

    return {
        "clip_id": metrics["clip_id"],
        "phase_context": phase_context,
        "language": lang,
        "model": GEMINI_MODEL,
        "prompt_version": PROMPT_VERSION,
        "headline": parsed["headline"],
        "implication": parsed["implication"],
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

    out_path = Path(args.output) if args.output else CACHE_DIR / f"{metrics['clip_id']}_explanation_{args.lang}.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("Headline:    ", result["headline"])
    print("Implication: ", result["implication"])
    print("Coaching cue:", result["coaching_cue"])


if __name__ == "__main__":
    main()
