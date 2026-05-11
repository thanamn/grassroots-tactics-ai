"""Append a manually collected LLM answer to a benchmark run.

Use this when a model is only available through a web UI or a provider
playground. Generate manual packets with run_llm_benchmark.py, copy the model
answer into a text file, then import it here so score_llm_benchmark.py can score
it alongside automated API results.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LLM_RUNS_DIR = ROOT / "data" / "eval" / "llm_runs"
MODEL_PRESETS_PATH = ROOT / "study" / "llm_model_presets.json"
PROMPT_VARIANTS_PATH = ROOT / "study" / "llm_prompt_variants.json"
CASES_PATH = ROOT / "study" / "llm_eval_cases.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model-preset", required=True)
    parser.add_argument("--prompt", default="current_v2")
    parser.add_argument("--case", required=True)
    parser.add_argument("--response-file", required=True)
    parser.add_argument("--repeat-index", type=int, default=0)
    args = parser.parse_args()

    presets = _load_json(MODEL_PRESETS_PATH)
    prompts = _load_json(PROMPT_VARIANTS_PATH)
    cases = _load_json(CASES_PATH)
    if args.model_preset not in presets:
        raise ValueError(f"Unknown model preset: {args.model_preset}")
    if args.prompt not in prompts:
        raise ValueError(f"Unknown prompt variant: {args.prompt}")
    if args.case not in cases:
        raise ValueError(f"Unknown case: {args.case}")

    response_text = Path(args.response_file).read_text(encoding="utf-8").strip()
    preset = presets[args.model_preset]
    case = cases[args.case]
    row = {
        "run_id": args.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_preset": args.model_preset,
        "provider": preset.get("provider"),
        "model": preset.get("model"),
        "prompt_variant": args.prompt,
        "case_id": args.case,
        "task": case.get("task"),
        "lang": case.get("lang"),
        "repeat_index": args.repeat_index,
        "status": "ok",
        "manual_packet": "",
        "response_text": response_text,
        "latency_s": None,
        "usage": {},
        "error": "",
    }
    out_path = LLM_RUNS_DIR / args.run_id / "responses.jsonl"
    _append_jsonl(out_path, row)
    print(f"[manual] appended response -> {out_path}")


if __name__ == "__main__":
    main()
