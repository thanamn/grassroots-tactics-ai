"""Run model and prompt evaluations for the LLM parts of the app.

The app uses LLMs in two places:
1. one-shot tactical explanations that must return JSON;
2. coach chat answers grounded in spacing/compactness/possession/pass metrics.

This runner keeps the same case set for every model/prompt pair and writes
JSONL outputs that can be scored by scripts/score_llm_benchmark.py.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prompts.tactical_explainer import build_messages  # noqa: E402
from src.coach_chat import SYSTEM_EN as CHAT_SYSTEM_EN  # noqa: E402
from src.coach_chat import SYSTEM_TH as CHAT_SYSTEM_TH  # noqa: E402
from src.coach_chat import _summary_block  # noqa: E402

EVAL_DIR = ROOT / "data" / "eval"
LLM_RUNS_DIR = EVAL_DIR / "llm_runs"
MODEL_PRESETS_PATH = ROOT / "study" / "llm_model_presets.json"
PROMPT_VARIANTS_PATH = ROOT / "study" / "llm_prompt_variants.json"
CASES_PATH = ROOT / "study" / "llm_eval_cases.json"


def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _selected_items(
    items: dict[str, Any],
    selected: list[str],
    *,
    default_key: str | None = None,
    all_items: bool = False,
) -> dict[str, Any]:
    if all_items:
        return items
    if selected:
        missing = [name for name in selected if name not in items]
        if missing:
            raise ValueError(f"Unknown selection(s): {', '.join(missing)}")
        return {name: items[name] for name in selected}
    if default_key == "default_enabled":
        return {name: cfg for name, cfg in items.items() if cfg.get("default_enabled")}
    if default_key:
        return {default_key: items[default_key]}
    return items


def _preset_api_key_available(preset: dict[str, Any]) -> bool:
    """Return true when an OpenAI-compatible preset has enough auth to run."""
    if preset.get("endpoint_type") != "openai_compatible":
        return True
    if preset.get("provider") == "Ollama":
        return True
    api_key_env = preset.get("api_key_env") or ""
    return bool(api_key_env and os.getenv(api_key_env, ""))


def _render_explainer_messages(case: dict[str, Any], variant: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    metrics = case["metrics"]
    lang = case.get("lang", "en")
    phase = case.get("phase_context", "general open play")
    mode = variant.get("mode", "current")

    if mode == "minimal_schema":
        if lang == "th":
            system = (
                "คุณเป็นผู้ช่วยโค้ชฟุตบอล ตอบเป็น JSON เท่านั้น มี key: "
                "headline, implication, coaching_cue ใช้ภาษาง่ายและอิงจากตัวเลขที่ให้"
            )
            user = (
                f"บริบท: {phase}\nข้อมูล:\n"
                f"{json.dumps(metrics, ensure_ascii=False, indent=2)}\n"
                "ตอบเป็น JSON object เท่านั้น"
            )
        else:
            system = (
                "You are a football coaching assistant. Return JSON only with keys: "
                "headline, implication, coaching_cue. Use plain language and ground the answer in the metrics."
            )
            user = (
                f"Context: {phase}\nMetrics:\n"
                f"{json.dumps(metrics, ensure_ascii=False, indent=2)}\n"
                "Respond with a JSON object only."
            )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}], True

    system, user = build_messages(metrics, phase_context=phase, lang=lang)
    if mode == "no_jargon_guard":
        if lang == "th":
            system = (
                "คุณเป็นนักวิเคราะห์ฟุตบอลสำหรับโค้ชสมัครเล่น อธิบาย pattern สำคัญจากข้อมูล "
                "แล้วตอบเป็น JSON fields: headline, implication, coaching_cue"
            )
        else:
            system = (
                "You are a football analyst for amateur coaches. Explain the key pattern from the data "
                "and return JSON fields: headline, implication, coaching_cue."
            )
    elif mode == "metric_first":
        system += (
            "\n\nBefore writing the answer, choose the single largest metric or event change. "
            "The final answer should make that change visibly drive the coaching cue."
        )
    elif mode == "few_shot_coach":
        if lang == "th":
            system += (
                "\n\nตัวอย่างสั้น:\n"
                "{\"headline\":\"ทีม A หลวมขึ้นช่วงท้ายจังหวะ\",\"implication\":\"ราววินาทีที่ 8.0 ระยะห่างเปิดขึ้น ทำให้ช่องกลางรับยากขึ้น\","
                "\"coaching_cue\":\"ตอนเสียบอล ให้มิดฟิลด์ตัวใกล้สุดบีบเข้าหากันก่อนหันไปไล่บอล\"}"
            )
        else:
            system += (
                "\n\nShort example:\n"
                "{\"headline\":\"Team A stretched after the first pass.\","
                "\"implication\":\"Around 8.0 s the gap opened, so the middle became harder to defend.\","
                "\"coaching_cue\":\"On the next turnover, the nearest two midfielders should close together before chasing the ball.\"}"
            )

    user += "\n\nRespond with a JSON object only."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], bool(variant.get("use_json_mode", True))


def _render_chat_messages(case: dict[str, Any], variant: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    lang = case.get("lang", "en")
    base_system = CHAT_SYSTEM_TH if lang == "th" else CHAT_SYSTEM_EN
    system = base_system + "\n\n" + _summary_block(case["metrics"], lang)
    mode = variant.get("mode", "current")

    if mode == "minimal_schema":
        system = (
            ("ตอบสั้น 2-3 ประโยค อิงจากข้อมูลเท่านั้น ถ้าคำถามนอกเรื่องให้พากลับมาที่ข้อมูล\n\n")
            if lang == "th"
            else ("Answer in 2-3 sentences. Ground the answer in the metrics only. Redirect out-of-scope questions.\n\n")
        ) + _summary_block(case["metrics"], lang)
    elif mode == "metric_first":
        system += "\n\nStart from the strongest metric/event in the data before giving advice."
    elif mode == "few_shot_coach":
        system += "\n\nExample style: 'At about 8.7 s, Team A stretched. First cue: close the nearest two players before chasing.'"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": case["question"]},
    ], False


def _render_messages(case: dict[str, Any], variant: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    if case["task"] == "explainer":
        return _render_explainer_messages(case, variant)
    if case["task"] == "chat":
        return _render_chat_messages(case, variant)
    raise ValueError(f"Unknown task: {case['task']}")


def _manual_packet_text(
    *,
    run_id: str,
    preset_name: str,
    preset: dict[str, Any],
    prompt_name: str,
    case_id: str,
    case: dict[str, Any],
    messages: list[dict[str, str]],
) -> str:
    system_text = "\n\n".join(message["content"] for message in messages if message["role"] == "system").strip()
    user_text = "\n\n".join(message["content"] for message in messages if message["role"] == "user").strip()
    combined_prompt = (
        "Please follow the instruction block and answer the user task below.\n\n"
        "=== INSTRUCTION / SYSTEM ===\n"
        f"{system_text}\n\n"
        "=== USER TASK ===\n"
        f"{user_text}\n"
    )
    blocks = [
        f"# Manual LLM Packet: {case_id}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Model preset: `{preset_name}`",
        f"- Provider/model: `{preset.get('provider')}` / `{preset.get('model')}`",
        f"- Prompt variant: `{prompt_name}`",
        f"- Task: `{case.get('task')}`",
        f"- Language: `{case.get('lang')}`",
        "",
        "## Instructions",
        "",
        "1. Open the provider's chat/API playground.",
        "2. Paste the SYSTEM message into the system/developer instruction field if available.",
        "3. Paste the USER message as the user prompt.",
        "4. Use temperature 0.6 unless the UI does not expose it.",
        "5. Copy the exact model answer into the benchmark result later.",
        "6. If the site only gives one chat box, paste the COMBINED SINGLE-BOX PROMPT instead of sending separate messages.",
        "",
        "## COMBINED SINGLE-BOX PROMPT",
        "",
        "Use this for ChatGPT, Claude.ai, Gemini app, Le Chat, DeepSeek Chat, Qwen Chat, HuggingChat, Grok, or any site without a dedicated system/developer field.",
        "",
        "```text",
        combined_prompt,
        "```",
        "",
    ]
    for message in messages:
        blocks.extend([
            f"## {message['role'].upper()}",
            "",
            "```text",
            message["content"],
            "```",
            "",
        ])
    return "\n".join(blocks)


def _call_openai_compatible(
    *,
    preset: dict[str, Any],
    messages: list[dict[str, str]],
    use_json_mode: bool,
    temperature: float,
    max_tokens: int,
) -> tuple[str, dict[str, Any]]:
    import openai

    api_key_env = preset.get("api_key_env") or ""
    api_key = os.getenv(api_key_env, "")
    if preset.get("provider") == "Ollama" and not api_key:
        api_key = "ollama"
    if not api_key:
        raise RuntimeError(f"Missing API key env var: {api_key_env}")

    headers = None
    if preset.get("provider") == "OpenRouter":
        headers = {
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Grassroots Tactics AI LLM Evaluation",
        }

    client = openai.OpenAI(
        api_key=api_key,
        base_url=preset["base_url"],
        default_headers=headers,
    )
    kwargs: dict[str, Any] = {
        "model": preset["model"],
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if use_json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    text = (response.choices[0].message.content or "").strip()
    usage = getattr(response, "usage", None)
    usage_dict = {}
    if usage is not None:
        usage_dict = {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
    return text, usage_dict


def main() -> None:
    load_dotenv(ROOT / ".env")

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--preset", dest="presets", action="append", default=[])
    parser.add_argument("--provider", dest="providers", action="append", default=[])
    parser.add_argument("--all-presets", action="store_true")
    parser.add_argument("--prompt", dest="prompts", action="append", default=[])
    parser.add_argument("--all-prompts", action="store_true")
    parser.add_argument("--case", dest="cases", action="append", default=[])
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manual-packets-only", action="store_true")
    parser.add_argument("--write-manual-packets", action="store_true")
    parser.add_argument("--skip-missing-keys", action="store_true")
    parser.add_argument("--no-json-mode", action="store_true")
    parser.add_argument("--list-presets", action="store_true")
    args = parser.parse_args()

    presets_all = _load_json(MODEL_PRESETS_PATH)
    variants_all = _load_json(PROMPT_VARIANTS_PATH)
    cases_all = _load_json(CASES_PATH)

    if args.list_presets:
        for name, cfg in presets_all.items():
            print(f"{name}: {cfg.get('provider')} / {cfg.get('model')} [{cfg.get('endpoint_type')}]")
        return

    presets = _selected_items(
        presets_all,
        args.presets,
        default_key="default_enabled",
        all_items=args.all_presets,
    )
    if args.providers:
        allowed = {p.lower() for p in args.providers}
        presets = {
            name: cfg for name, cfg in presets.items()
            if str(cfg.get("provider", "")).lower() in allowed
        }
    if not presets:
        raise ValueError("No model presets selected.")

    skipped_for_missing_key: list[str] = []
    if args.skip_missing_keys and not args.manual_packets_only:
        runnable_presets = {}
        for name, preset in presets.items():
            if _preset_api_key_available(preset):
                runnable_presets[name] = preset
            else:
                skipped_for_missing_key.append(name)
        presets = runnable_presets
        if skipped_for_missing_key:
            print(f"[llm] skip missing API keys: {', '.join(skipped_for_missing_key)}")
        if not presets:
            raise ValueError(
                "No runnable model presets remain after --skip-missing-keys. "
                "Add API keys to .env, choose Ollama/local presets, or use --manual-packets-only."
            )

    prompt_default = "current_v2"
    variants = _selected_items(
        variants_all,
        args.prompts,
        default_key=None if args.all_prompts else prompt_default,
        all_items=args.all_prompts,
    )
    cases = _selected_items(
        cases_all,
        args.cases,
        default_key=None,
        all_items=args.all_cases or not args.cases,
    )

    run_id = args.run_id or f"llm_{_now_stamp()}"
    run_dir = LLM_RUNS_DIR / run_id
    out_path = run_dir / "responses.jsonl"
    manual_dir = run_dir / "manual_packets"
    rows: list[dict[str, Any]] = []

    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "repeat": args.repeat,
        "presets": list(presets),
        "prompt_variants": list(variants),
        "cases": list(cases),
        "skipped_for_missing_key": skipped_for_missing_key,
        "notes": "Use current_v2 only for primary model comparison; use all prompt variants for prompt ablation.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    for preset_name, preset in presets.items():
        for prompt_name, variant in variants.items():
            for case_id, case in cases.items():
                messages, use_json_mode = _render_messages(case, variant)
                if args.no_json_mode:
                    use_json_mode = False

                packet_path: Path | None = None
                if args.write_manual_packets or args.manual_packets_only or preset.get("endpoint_type") != "openai_compatible":
                    packet_path = manual_dir / f"{_safe_filename(preset_name)}__{_safe_filename(prompt_name)}__{_safe_filename(case_id)}.md"
                    packet_path.parent.mkdir(parents=True, exist_ok=True)
                    packet_path.write_text(
                        _manual_packet_text(
                            run_id=run_id,
                            preset_name=preset_name,
                            preset=preset,
                            prompt_name=prompt_name,
                            case_id=case_id,
                            case=case,
                            messages=messages,
                        ),
                        encoding="utf-8",
                    )

                if args.dry_run or args.manual_packets_only or preset.get("endpoint_type") != "openai_compatible":
                    rows.append({
                        "run_id": run_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model_preset": preset_name,
                        "provider": preset.get("provider"),
                        "model": preset.get("model"),
                        "prompt_variant": prompt_name,
                        "case_id": case_id,
                        "task": case.get("task"),
                        "lang": case.get("lang"),
                        "repeat_index": 0,
                        "status": "manual_packet" if not args.dry_run else "dry_run",
                        "manual_packet": str(packet_path) if packet_path and packet_path.exists() else "",
                        "response_text": "",
                        "latency_s": None,
                        "usage": {},
                        "error": "",
                    })
                    continue

                for repeat_index in range(args.repeat):
                    print(f"[llm] {preset_name} / {prompt_name} / {case_id} / repeat {repeat_index + 1}")
                    t0 = time.perf_counter()
                    try:
                        response_text, usage = _call_openai_compatible(
                            preset=preset,
                            messages=messages,
                            use_json_mode=use_json_mode,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        status = "ok"
                        error = ""
                    except Exception as exc:  # noqa: BLE001
                        response_text = ""
                        usage = {}
                        status = "error"
                        error = repr(exc)
                    latency_s = time.perf_counter() - t0
                    rows.append({
                        "run_id": run_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "model_preset": preset_name,
                        "provider": preset.get("provider"),
                        "model": preset.get("model"),
                        "prompt_variant": prompt_name,
                        "case_id": case_id,
                        "task": case.get("task"),
                        "lang": case.get("lang"),
                        "repeat_index": repeat_index,
                        "status": status,
                        "manual_packet": "",
                        "response_text": response_text,
                        "latency_s": round(latency_s, 3),
                        "usage": usage,
                        "error": error,
                    })
                    _write_jsonl(out_path, [rows[-1]])

    skipped_rows = [row for row in rows if row["status"] in {"dry_run", "manual_packet"}]
    if skipped_rows:
        _write_jsonl(out_path, skipped_rows)

    print(f"[llm] wrote responses -> {out_path}")
    print(f"[llm] wrote run config -> {run_dir / 'run_config.json'}")
    if manual_dir.exists():
        print(f"[llm] wrote manual packets -> {manual_dir}")


if __name__ == "__main__":
    main()
