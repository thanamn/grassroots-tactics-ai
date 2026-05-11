"""Score LLM benchmark outputs with reproducible heuristics.

These automated checks are not a substitute for coach/player judgement. They
catch the things that matter mechanically: schema validity, grounding in given
metrics, scope control, Thai/English language fit, jargon avoidance, concision,
and latency/token cost.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLM_RUNS_DIR = ROOT / "data" / "eval" / "llm_runs"
EVAL_REPORTS_DIR = ROOT / "data" / "eval" / "reports"
CASES_PATH = ROOT / "study" / "llm_eval_cases.json"

REQUIRED_KEYS = ("headline", "implication", "coaching_cue")
BANNED_TERMS = (
    "expected goals",
    "ppda",
    "voronoi",
    "expected threat",
    "xg",
    "xt",
    "half-space",
    "half spaces",
    "line-breaking",
    "progressive pass",
)
TIMESTAMP_RE = re.compile(r"(\d+(?:\.\d+)?\s*(?:s|sec|second|seconds)|วินาที|นาที)")
NUMBER_RE = re.compile(r"\d")
THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
DRAMATIC_INVENTION_TERMS = (
    "huge problem",
    "dangerous gap",
    "collapsed",
    "panic",
    "big mistake",
    "เสียหายหนัก",
    "พัง",
)
ACTION_TERMS = (
    "tell",
    "ask",
    "cue",
    "next time",
    "training",
    "shout",
    "close",
    "hold",
    "drop",
    "scan",
    "สั่ง",
    "บอก",
    "ซ้อม",
    "ครั้งหน้า",
    "บีบ",
    "ยืน",
)
REDIRECT_TERMS = (
    "the data shows",
    "from these metrics",
    "spacing",
    "compactness",
    "possession",
    "passing",
    "ข้อมูล",
    "ระยะ",
    "การยืน",
    "ครองบอล",
    "ส่งบอล",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round(value: float | int | None, digits: int = 4) -> float | str:
    if value is None:
        return ""
    return round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None and not np.isnan(v)]
    return None if not clean else float(sum(clean) / len(clean))


def _rate(values: list[bool]) -> float | None:
    return None if not values else float(sum(1 for v in values if v) / len(values))


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json|JSON|text)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def _parse_json_response(text: str) -> tuple[dict[str, Any] | None, str]:
    stripped = _strip_code_fence(text)
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed, ""
    except json.JSONDecodeError as exc:
        error = str(exc)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(stripped[start:end + 1])
            if isinstance(parsed, dict):
                return parsed, ""
        except json.JSONDecodeError as exc:
            error = str(exc)
    return None, error if "error" in locals() else "not a JSON object"


def _text_for_scoring(row: dict[str, Any], parsed: dict[str, Any] | None) -> str:
    if parsed:
        return "\n".join(str(parsed.get(key, "")) for key in REQUIRED_KEYS)
    return row.get("response_text", "")


def _contains_any(text: str, needles: list[str] | tuple[str, ...]) -> bool:
    lower = text.lower()
    return any(needle.lower() in lower for needle in needles)


def _count_terms(text: str, terms: list[str] | tuple[str, ...]) -> int:
    lower = text.lower()
    return sum(lower.count(term.lower()) for term in terms)


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


def _score_row(row: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    expectations = case.get("expectations", {})
    status_ok = row.get("status") == "ok"
    text_raw = row.get("response_text", "") or ""

    parsed = None
    parse_error = ""
    if expectations.get("json_required"):
        parsed, parse_error = _parse_json_response(text_raw)

    text = _text_for_scoring(row, parsed)
    text_lower = text.lower()
    forbidden = expectations.get("should_not_mention", [])
    should_mention = expectations.get("should_mention_any", [])

    required_keys_ok = True
    if expectations.get("json_required"):
        required_keys_ok = bool(parsed) and all(str(parsed.get(key, "")).strip() for key in expectations.get("must_have_keys", REQUIRED_KEYS))

    timestamp_ok = True
    if expectations.get("needs_timestamps"):
        timestamp_ok = bool(TIMESTAMP_RE.search(text))

    thai_ok = True
    if expectations.get("needs_thai"):
        thai_chars = len(THAI_RE.findall(text))
        thai_ok = thai_chars >= 8

    mention_ok = True
    if should_mention:
        mention_ok = _contains_any(text, should_mention)

    forbidden_count = _count_terms(text, forbidden)
    forbidden_ok = forbidden_count == 0
    banned_count = _count_terms(text, BANNED_TERMS)
    jargon_ok = banned_count == 0

    no_ball_invention_ok = True
    if expectations.get("ball_available") is False:
        no_ball_invention_ok = not _contains_any(text, ("possession", "pass count", "passing accuracy", "ครองบอล", "ความแม่นยำส่งบอล"))

    stable_ok = True
    if expectations.get("stable_case"):
        stable_ok = not _contains_any(text, DRAMATIC_INVENTION_TERMS)

    scope_ok = True
    if expectations.get("out_of_scope"):
        scope_ok = _contains_any(text, REDIRECT_TERMS) and forbidden_ok

    concise_ok = False
    if case.get("task") == "explainer" and parsed:
        field_word_counts = [_word_count(str(parsed.get(key, ""))) for key in REQUIRED_KEYS]
        concise_ok = all(count <= 40 for count in field_word_counts) and sum(field_word_counts) <= 95
    elif case.get("task") == "chat":
        concise_ok = _word_count(text) <= 100

    action_ok = True
    if case.get("task") == "explainer" and parsed:
        cue = str(parsed.get("coaching_cue", ""))
        action_ok = _contains_any(cue, ACTION_TERMS)
    elif case.get("task") == "chat" and not expectations.get("out_of_scope"):
        action_ok = _contains_any(text, ACTION_TERMS)

    has_number_ok = bool(NUMBER_RE.search(text))
    if expectations.get("stable_case"):
        has_number_ok = True

    task_success = 0
    task_success += 10 if status_ok else 0
    task_success += 15 if required_keys_ok else 0

    grounding = 0
    grounding += 8 if mention_ok else 0
    grounding += 7 if timestamp_ok else 0
    grounding += 5 if has_number_ok else 0
    grounding += 5 if no_ball_invention_ok else 0

    scope = 0
    scope += 8 if forbidden_ok else 0
    scope += 6 if stable_ok else 0
    scope += 6 if scope_ok else 0

    style = 0
    style += 6 if jargon_ok else 0
    style += 5 if concise_ok else 0
    style += 4 if thai_ok else 0

    actionability = 15 if action_ok else 0
    auto_score = task_success + grounding + scope + style + actionability

    return {
        "run_id": row.get("run_id", ""),
        "model_preset": row.get("model_preset", ""),
        "provider": row.get("provider", ""),
        "model": row.get("model", ""),
        "prompt_variant": row.get("prompt_variant", ""),
        "case_id": row.get("case_id", ""),
        "task": row.get("task", ""),
        "lang": row.get("lang", ""),
        "repeat_index": row.get("repeat_index", ""),
        "status": row.get("status", ""),
        "auto_score": auto_score,
        "task_success_score": task_success,
        "grounding_score": grounding,
        "scope_score": scope,
        "style_score": style,
        "actionability_score": actionability,
        "json_parse_ok": bool(parsed) if expectations.get("json_required") else "",
        "required_keys_ok": required_keys_ok,
        "timestamp_ok": timestamp_ok,
        "thai_ok": thai_ok,
        "mention_ok": mention_ok,
        "forbidden_ok": forbidden_ok,
        "jargon_ok": jargon_ok,
        "no_ball_invention_ok": no_ball_invention_ok,
        "stable_ok": stable_ok,
        "scope_ok": scope_ok,
        "concise_ok": concise_ok,
        "action_ok": action_ok,
        "banned_count": banned_count,
        "forbidden_count": forbidden_count,
        "word_count": _word_count(text),
        "latency_s": row.get("latency_s"),
        "prompt_tokens": (row.get("usage") or {}).get("prompt_tokens"),
        "completion_tokens": (row.get("usage") or {}).get("completion_tokens"),
        "total_tokens": (row.get("usage") or {}).get("total_tokens"),
        "parse_error": parse_error,
        "error": row.get("error", ""),
    }


def _paired_bootstrap_delta(
    baseline: dict[str, float],
    candidate: dict[str, float],
    iterations: int,
    seed: int,
) -> tuple[float | None, float | None, float | None, int]:
    keys = sorted(baseline.keys() & candidate.keys())
    if len(keys) < 2:
        return None, None, None, len(keys)
    base = np.array([baseline[k] for k in keys], dtype=float)
    cand = np.array([candidate[k] for k in keys], dtype=float)
    delta = cand - base
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), size=(iterations, len(delta)))
    means = delta[idx].mean(axis=1)
    return float(delta.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(keys)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-model", default="deepseek_v4_pro")
    parser.add_argument("--baseline-prompt", default="current_v2")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--practical-score-delta", type=float, default=5.0)
    args = parser.parse_args()

    cases = _load_json(CASES_PATH)
    responses_path = LLM_RUNS_DIR / args.run_id / "responses.jsonl"
    rows = _read_jsonl(responses_path)

    detail_rows = []
    for row in rows:
        case = cases.get(row.get("case_id"), {})
        if not case:
            continue
        if row.get("status") in {"manual_packet", "dry_run"}:
            continue
        detail_rows.append(_score_row(row, case))

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(row["model_preset"], row["prompt_variant"])].append(row)

    summary_rows = []
    for (model_preset, prompt_variant), group in sorted(grouped.items()):
        ok_rows = [row for row in group if row["status"] == "ok"]
        summary_rows.append({
            "run_id": args.run_id,
            "model_preset": model_preset,
            "provider": group[0].get("provider", ""),
            "model": group[0].get("model", ""),
            "prompt_variant": prompt_variant,
            "n_rows": len(group),
            "n_ok": len(ok_rows),
            "ok_rate": _round(_rate([row["status"] == "ok" for row in group])),
            "mean_auto_score": _round(_mean([float(row["auto_score"]) for row in ok_rows]), 2),
            "schema_success_rate": _round(_rate([bool(row["required_keys_ok"]) for row in ok_rows if row["json_parse_ok"] != ""])),
            "timestamp_success_rate": _round(_rate([bool(row["timestamp_ok"]) for row in ok_rows])),
            "scope_success_rate": _round(_rate([bool(row["scope_ok"]) for row in ok_rows])),
            "jargon_success_rate": _round(_rate([bool(row["jargon_ok"]) for row in ok_rows])),
            "mean_latency_s": _round(_mean([float(row["latency_s"]) for row in ok_rows if row["latency_s"] not in ("", None)]), 3),
            "mean_total_tokens": _round(_mean([float(row["total_tokens"]) for row in ok_rows if row["total_tokens"] not in ("", None)]), 1),
        })

    score_by_pair: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in detail_rows:
        if row["status"] != "ok":
            continue
        key = f"{row['case_id']}::{row['repeat_index']}"
        score_by_pair[(row["model_preset"], row["prompt_variant"])][key] = float(row["auto_score"])

    baseline_key = (args.baseline_model, args.baseline_prompt)
    comparison_rows = []
    if baseline_key in score_by_pair:
        baseline_scores = score_by_pair[baseline_key]
        for pair, scores in sorted(score_by_pair.items()):
            if pair == baseline_key:
                continue
            delta, low, high, n = _paired_bootstrap_delta(
                baseline_scores,
                scores,
                args.bootstrap_iterations,
                args.seed,
            )
            separated = bool(low is not None and high is not None and (low > 0 or high < 0))
            meaningful = bool(delta is not None and abs(delta) >= args.practical_score_delta)
            comparison_rows.append({
                "run_id": args.run_id,
                "baseline_model": args.baseline_model,
                "baseline_prompt": args.baseline_prompt,
                "model_preset": pair[0],
                "prompt_variant": pair[1],
                "paired_units": n,
                "delta_auto_score": _round(delta, 2),
                "delta_ci_low": _round(low, 2),
                "delta_ci_high": _round(high, 2),
                "bootstrap_separated_from_zero": separated,
                "practically_meaningful": meaningful,
                "comparison_flag": separated and meaningful,
                "practical_threshold": args.practical_score_delta,
            })

    reports_base = EVAL_REPORTS_DIR / f"{args.run_id}_llm"
    detail_fields = [
        "run_id", "model_preset", "provider", "model", "prompt_variant", "case_id",
        "task", "lang", "repeat_index", "status", "auto_score",
        "task_success_score", "grounding_score", "scope_score", "style_score",
        "actionability_score", "json_parse_ok", "required_keys_ok",
        "timestamp_ok", "thai_ok", "mention_ok", "forbidden_ok", "jargon_ok",
        "no_ball_invention_ok", "stable_ok", "scope_ok", "concise_ok",
        "action_ok", "banned_count", "forbidden_count", "word_count",
        "latency_s", "prompt_tokens", "completion_tokens", "total_tokens",
        "parse_error", "error",
    ]
    summary_fields = [
        "run_id", "model_preset", "provider", "model", "prompt_variant",
        "n_rows", "n_ok", "ok_rate", "mean_auto_score", "schema_success_rate",
        "timestamp_success_rate", "scope_success_rate", "jargon_success_rate",
        "mean_latency_s", "mean_total_tokens",
    ]
    comparison_fields = [
        "run_id", "baseline_model", "baseline_prompt", "model_preset",
        "prompt_variant", "paired_units", "delta_auto_score", "delta_ci_low",
        "delta_ci_high", "bootstrap_separated_from_zero",
        "practically_meaningful", "comparison_flag", "practical_threshold",
    ]
    _write_csv(reports_base.with_name(reports_base.name + "_details.csv"), detail_rows, detail_fields)
    _write_csv(reports_base.with_name(reports_base.name + "_summary.csv"), summary_rows, summary_fields)
    _write_csv(reports_base.with_name(reports_base.name + "_comparisons.csv"), comparison_rows, comparison_fields)

    json_report = {
        "run_id": args.run_id,
        "baseline_model": args.baseline_model,
        "baseline_prompt": args.baseline_prompt,
        "summary": summary_rows,
        "comparisons": comparison_rows,
        "notes": {
            "auto_score": "0-100 heuristic score: task success 25, grounding 25, scope 20, style 15, actionability 15.",
            "unit_of_comparison": "case_id x repeat_index paired across models/prompts.",
            "significance_rule": "comparison_flag requires bootstrap CI excluding 0 and absolute mean delta >= practical-score-delta.",
            "caveat": "Automated scores screen for validity and grounding; final claims should use human ratings for tactical usefulness.",
        },
    }
    reports_base.with_name(reports_base.name + "_summary.json").write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[score] wrote LLM reports under {EVAL_REPORTS_DIR}")


if __name__ == "__main__":
    main()
