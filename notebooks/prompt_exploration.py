"""Prompt exploration: try the tactical explainer with mock metrics.

Run this BEFORE you have real tracking data, to lock down prompt quality.
You can iterate on prompts/tactical_explainer.py and re-run this script
in seconds without touching YOLO or videos at all.

Usage:
    # Set GEMINI_API_KEY in .env first (free key: https://aistudio.google.com/apikey)
    python notebooks/prompt_exploration.py
    python notebooks/prompt_exploration.py --lang th
    python notebooks/prompt_exploration.py --scenario stretched_attack

What to look for in the output
------------------------------
1. Does the headline name the *most important* event, not just any event?
2. Does the implication tie a metric to something visible on the pitch?
3. Is the coaching cue something a real coach would shout (≤ 12 words)?
4. Did the model sneak in jargon ("xG", "PPDA", "half-spaces")? It shouldn't.
5. JSON parses without error every time? (Gemini's response_schema enforces
   this server-side, so failures here mean a schema or quota issue.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python notebooks/prompt_exploration.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prompts.tactical_explainer import build_messages, make_mock_metrics
from src.explainer import explain


SCENARIOS: dict[str, dict] = {
    # The default mock — Team A defending, with one big stretch event
    "defending_phase": {
        "metrics": make_mock_metrics(),
        "phase": "Team A defending in own half",
    },

    # Team A holds shape well throughout — testing that Claude
    # doesn't invent drama when nothing happens
    "stable_block": {
        "metrics": {
            "clip_id": "mock_stable_block",
            "fps": 25.0, "duration_s": 12.0,
            "summary": {
                "team_A": {"hull_area": {"mean": 9800, "min": 9100, "max": 10500, "std": 320, "n": 300},
                           "spread_std": {"mean": 65, "min": 58, "max": 72, "std": 4, "n": 300}},
                "team_B": {"hull_area": {"mean": 14200, "min": 13800, "max": 14800, "std": 250, "n": 300},
                           "spread_std": {"mean": 92, "min": 88, "max": 96, "std": 3, "n": 300}},
                "centroid_distance": {"mean": 110, "min": 95, "max": 128, "std": 8, "n": 300},
            },
            "events": [],
        },
        "phase": "Team A defending mid-block, no transition",
    },

    # Team A loses shape attacking — mirror scenario
    "stretched_attack": {
        "metrics": {
            "clip_id": "mock_stretched_attack",
            "fps": 25.0, "duration_s": 14.0,
            "summary": {
                "team_A": {"hull_area": {"mean": 19500, "min": 14000, "max": 24200, "std": 2900, "n": 350},
                           "spread_std": {"mean": 118, "min": 92, "max": 145, "std": 16, "n": 350}},
                "team_B": {"hull_area": {"mean": 10200, "min": 8800, "max": 11400, "std": 720, "n": 350},
                           "spread_std": {"mean": 70, "min": 62, "max": 78, "std": 5, "n": 350}},
                "centroid_distance": {"mean": 168, "min": 110, "max": 215, "std": 32, "n": 350},
            },
            "events": [
                {"t": 3.5, "team": "team_A", "type": "stretch",
                 "delta_pct": 38.0, "hull_before": 15200, "hull_after": 20970},
                {"t": 9.8, "team": "team_A", "type": "stretch",
                 "delta_pct": 31.0, "hull_before": 18400, "hull_after": 24100},
            ],
        },
        "phase": "Team A attacking, Team B in compact mid-block",
    },
}


def render_only(scenario: str, lang: str) -> None:
    """Just print the prompt — no API call. Free to run, useful for debugging."""
    s = SCENARIOS[scenario]
    system, user = build_messages(s["metrics"], phase_context=s["phase"], lang=lang)
    print("─" * 60); print(f"SCENARIO: {scenario}  |  LANG: {lang}"); print("─" * 60)
    print("\n[SYSTEM]\n"); print(system)
    print("\n[USER]\n");   print(user)


def run_explanation(scenario: str, lang: str) -> None:
    s = SCENARIOS[scenario]
    print("─" * 60); print(f"SCENARIO: {scenario}  |  LANG: {lang}"); print("─" * 60)
    result = explain(s["metrics"], phase_context=s["phase"], lang=lang)
    print(f"\nHeadline:     {result['headline']}")
    print(f"Implication:  {result['implication']}")
    print(f"Coaching cue: {result['coaching_cue']}")
    print(f"\n(model: {result['model']}, prompt: {result['prompt_version']})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="defending_phase",
                        choices=list(SCENARIOS.keys()) + ["all"])
    parser.add_argument("--lang", default="en", choices=["en", "th"])
    parser.add_argument("--render-only", action="store_true",
                        help="Print prompt without calling the API")
    args = parser.parse_args()

    scenarios = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for sc in scenarios:
        if args.render_only:
            render_only(sc, args.lang)
        else:
            run_explanation(sc, args.lang)
        print()


if __name__ == "__main__":
    main()
