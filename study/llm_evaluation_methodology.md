# LLM Evaluation Methodology

Last updated: 2026-05-10

This document covers the LLM parts of Grassroots Tactics AI. It is separate
from the detector/tracking benchmark because the LLM is evaluated on language,
grounding, usefulness, and scope control rather than pixel localization.

## What The Current Software Uses LLMs For

The current software uses an LLM in two product-facing places.

1. Tactical explainer (`src/explainer.py`)

- Input: compact metrics JSON from the finished clip.
- Current provider route: OpenAI SDK pointed at DeepSeek's OpenAI-compatible API.
- Current default model: `DEEPSEEK_MODEL=deepseek-v4-pro`.
- Current prompt: `prompts/tactical_explainer.py`, `PROMPT_VERSION = "v2"`.
- Output: strict JSON with `headline`, `implication`, and `coaching_cue`.
- Scope: spacing, compactness, possession, pass count, and passing accuracy.
- Ball/pass data: included only when `ball_metrics` exists, so the model should not invent possession or passing claims when ball labels are absent.

2. Coach chat (`src/coach_chat.py`)

- Input: a compact metrics summary plus the user's question.
- Current provider route: same DeepSeek OpenAI-compatible API.
- Output: short free-form answer, usually 2-3 sentences.
- Scope guard: the assistant should answer only spacing, compactness, possession, and passing questions.
- Redirect behavior: if the user asks about formations, pressing, transitions, set pieces, individual skill, or fitness, the model should redirect to what the current metrics can support.

The LLM is not used for player detection, ball detection, team assignment, or
metric computation. Those are evaluated by the tracking/ball benchmark.

## Evaluation Questions

RQ1. Model quality: Which LLM gives the best grounded, coach-useful explanation for the same tactical metrics?

RQ2. Model reliability: Which LLM most often returns valid JSON, follows scope limits, avoids jargon, and does not invent unavailable metrics?

RQ3. Cost and speed: Which LLM gives acceptable quality with low latency and token/cost overhead?

RQ4. Prompt design: Does the production prompt meaningfully improve output compared with simpler or alternative prompts?

RQ5. User value: Do casual football players understand, trust, and prefer the explanations?

## Model Comparison Design

The main comparison should hold the prompt constant.

- Primary prompt: `current_v2`.
- Cases: all cases in `study/llm_eval_cases.json`.
- Repeats: 2-3 repeats per model if budget allows. Use the same temperature for every model.
- Recommended temperature: `0.6` to match production behavior; use `0.2` for a lower-variance robustness pass.
- Unit of comparison: one case x repeat.
- Default baseline: `deepseek_v4_pro`, because that is closest to production.

Recommended first automated set:

- `deepseek_v4_pro`
- `deepseek_v4_flash`
- `openai_gpt_5_4_mini`
- `openai_gpt_5_1`
- `google_gemini_2_5_flash`
- `google_gemini_2_5_flash_lite`
- `mistral_small_latest`
- `groq_llama_3_1_8b`
- `groq_llama_3_3_70b`
- `groq_qwen3_32b`
- `together_llama_3_3_70b_turbo`
- `ollama_qwen3_8b` if Ollama is installed locally

Recommended expanded set:

- Frontier closed: OpenAI GPT-5.5/GPT-5.4/GPT-5.1, Claude Opus/Sonnet/Haiku, Gemini 3.1/Gemini 2.5, Grok 4.3.
- Fast closed: DeepSeek V4 Flash, Gemini Flash/Lite, Grok fast, Mistral Small.
- Open hosted: Llama 3.3 70B, Qwen 3 32B, GPT-OSS 120B/20B, Kimi K2.5.
- Local open: Llama 3.1 8B, Qwen 3 8B/14B, Gemma 3 4B, DeepSeek-R1 8B, GPT-OSS 20B.

The executable preset list is in `study/llm_model_presets.json`. Models marked
`endpoint_type = "openai_compatible"` can be run by the benchmark script when
the relevant API key is available. Models marked `manual_or_sdk` need either a
small SDK adapter or manual packet workflow.

## Prompt Comparison Design

Yes, the prompt should be tested, but not as a giant free-for-all. The clean
design is:

1. Use `current_v2` for the main model comparison.
2. Pick 3 representative models for prompt ablation.
3. Run all prompt variants on those 3 models.
4. Report whether prompt choice changes schema success, grounding, readability, and human preference.

Recommended prompt-ablation models:

- Production/cheap: `deepseek_v4_pro` or `deepseek_v4_flash`.
- Frontier: one of OpenAI/Gemini/Claude.
- Small/open: one Groq/Together/Ollama open model.

Current prompt variants:

- `current_v2`: production prompt with grassroots persona, banned jargon, JSON schema, and timestamp/metric grounding.
- `minimal_schema`: minimal JSON-only prompt. Tests whether model quality alone is enough.
- `no_jargon_guard`: weakens the style guard. Tests whether explicit readability/jargon constraints matter.
- `metric_first`: forces the answer to start from the strongest metric/event.
- `few_shot_coach`: adds one example of a good coach-facing answer.

Prompt techniques to document in the paper:

- Role/persona: "tactical analyst for grassroots coaches."
- Scope restriction: only spacing, compactness, possession, and passing.
- Grounding instruction: tie every claim to visible pitch behavior, timestamps, or numbers.
- Jargon blacklist: blocks elite-analytics terms that casual players may not know.
- Structured output: JSON fields so the UI can render consistently.
- Language-specific prompting: separate English and Thai templates.
- Metric summarization: pass compact clip summaries, not raw per-frame tracking.
- Optional ball/pass block: include only when ball annotations exist.
- Out-of-scope redirect for coach chat.

## Automated Metrics

Automated scoring is a screen, not the final truth. It catches mechanical failures.

- API success: the model returned a response without error.
- JSON parse success: explainer outputs parse as JSON.
- Required keys: `headline`, `implication`, and `coaching_cue` are present and non-empty.
- Grounding: output mentions at least one expected timestamp, event, metric, possession value, or pass count.
- Timestamp use: output includes a timestamp when the case expects one.
- No unavailable-metric invention: if ball labels are absent, the model should not claim possession or pass-count findings.
- Scope control: coach chat should redirect formation/pressing questions to supported metrics.
- Jargon avoidance: output does not use banned terms such as xG, PPDA, xT, Voronoi, half-space, or expected threat.
- Language fit: Thai cases contain Thai text.
- Concision: explanations stay short enough for the UI.
- Actionability: the output includes one concrete cue a coach could use.
- Latency: wall-clock time per response.
- Token use: prompt, completion, and total tokens when the provider returns usage.

The scorer creates a 0-100 automated score:

- Task success: 25 points.
- Grounding: 25 points.
- Scope/safety: 20 points.
- Style/readability: 15 points.
- Actionability: 15 points.

## Human Rating Metrics

For the paper, use automated scores plus human ratings. Human judgement is the
stronger evidence for coach usefulness.

Recommended blind rating form for each output:

- Tactical correctness: 1-5. Does the explanation match the metrics/video?
- Grounding: 1-5. Does it clearly point to a timestamp, number, or visible pitch pattern?
- Usefulness: 1-5. Could a coach/player act on this in training?
- Readability: 1-5. Is it easy for a casual football player to understand?
- Trust: 1-5. Would you trust this explanation as a first-pass assistant?
- Scope discipline: 1-5. Does it avoid unsupported claims?
- Preference rank: choose the best output among 3-5 anonymized model outputs for the same case.
- Free-text: "What made this answer better/worse?"

For casual football players, prioritize readability, usefulness, and trust.
For a coach, also ask tactical correctness and actionability.

## Significance And Practical Difference

Because the dataset is small, do not overclaim classical statistical
significance. Use paired comparisons and practical thresholds.

Automated model comparison:

- Pair by exact `case_id` and `repeat_index`.
- Use paired bootstrap on the auto-score delta.
- Report 95 percent bootstrap confidence intervals.
- Mark a difference as meaningful only if the interval excludes 0 and the mean delta is at least 5 auto-score points.

Human Likert ratings:

- Pair by participant and case whenever possible.
- Use paired bootstrap or Wilcoxon signed-rank for model A vs B.
- Treat a mean delta of 0.5 on a 5-point Likert scale as practically meaningful.
- Report effect size and confidence interval, not only a p-value.

Preference rankings:

- Report win rate by model/prompt.
- Use a sign test or bootstrap confidence interval over cases/participants.
- Do not claim "best model overall" if the preference difference is small or unstable.

Latency:

- Report median and mean latency.
- A practical latency difference is meaningful if it changes interaction feel, e.g. more than 2 seconds for chat or more than 5 seconds for post-clip explanation.

## Commands

List available model presets:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py --list-presets
```

Smoke-test the prompt/case rendering without calling APIs:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py --dry-run --run-id llm_dry_run_v1

# Or use the one-command plan runner:
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode smoke -RunId llm_smoke_v1
```

Run the default production-like comparison:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py `
  --run-id llm_default_v1 `
  --repeat 2

.venv\Scripts\python.exe scripts/score_llm_benchmark.py `
  --run-id llm_default_v1 `
  --baseline-model deepseek_v4_pro `
  --baseline-prompt current_v2

# Or:
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode default -RunId llm_default_v1 -Repeat 2
```

Run a 10+ model automated comparison:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py `
  --run-id llm_10plus_v1 `
  --prompt current_v2 `
  --repeat 2 `
  --preset deepseek_v4_pro `
  --preset deepseek_v4_flash `
  --preset openai_gpt_5_4_mini `
  --preset openai_gpt_5_1 `
  --preset google_gemini_2_5_flash `
  --preset google_gemini_2_5_flash_lite `
  --preset mistral_small_latest `
  --preset groq_llama_3_1_8b `
  --preset groq_llama_3_3_70b `
  --preset groq_qwen3_32b `
  --preset together_llama_3_3_70b_turbo

.venv\Scripts\python.exe scripts/score_llm_benchmark.py --run-id llm_10plus_v1

# Or:
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode 10plus -RunId llm_10plus_v1 -Repeat 2
```

Run prompt ablation on three representative models:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py `
  --run-id llm_prompt_ablation_v1 `
  --all-prompts `
  --repeat 2 `
  --preset deepseek_v4_pro `
  --preset google_gemini_2_5_flash `
  --preset groq_qwen3_32b

.venv\Scripts\python.exe scripts/score_llm_benchmark.py `
  --run-id llm_prompt_ablation_v1 `
  --baseline-model deepseek_v4_pro `
  --baseline-prompt current_v2

# Or:
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode prompt-ablation -RunId llm_prompt_ablation_v1 -Repeat 2
```

Create manual packets for models without simple API automation:

```powershell
.venv\Scripts\python.exe scripts/run_llm_benchmark.py `
  --run-id llm_manual_packets_v1 `
  --manual-packets-only `
  --all-presets `
  --prompt current_v2

# Or:
powershell -ExecutionPolicy Bypass -File scripts\run_llm_eval_plan.ps1 -Mode manual -RunId llm_manual_packets_v1
```

The packets are written to:

```text
data/eval/llm_runs/<run_id>/manual_packets/
```

Use `study/manual_llm_access_list.md` for links to ChatGPT, Claude, Gemini,
Google AI Studio, DeepSeek Chat, Groq, OpenRouter, HuggingChat, Qwen Chat,
Together, Mistral, and Grok, plus guidance on which sites support separate
system/user fields.

After copying a website/playground answer into a text file, append it to the
same benchmark run:

```powershell
.venv\Scripts\python.exe scripts/import_manual_llm_result.py `
  --run-id llm_manual_packets_v1 `
  --model-preset anthropic_claude_sonnet_4_6 `
  --prompt current_v2 `
  --case explainer_defensive_stretch_en `
  --response-file path\to\copied_answer.txt
```

## How To Report In The Paper

Recommended reporting structure:

- System role: The LLM translates computed tactical metrics into coach-facing explanations and chat answers.
- Model set: List models by family and size tier, not only by provider.
- Prompt set: State that `current_v2` is the production prompt, with ablations for schema-only, no-jargon guard, metric-first, and few-shot prompting.
- Automated results: schema success, grounding score, scope success, jargon avoidance, latency, token usage, and practical-difference flags.
- Human results: Likert means, confidence intervals, preference win rates, and representative comments.
- Main claim style: "Model X was more reliable/useful on our cases" rather than "Model X understands football better."

## Caveats

- Synthetic LLM cases are useful for repeatability, but final claims should include real outputs from the PSG/Milan evaluation clips once their metric JSON files exist.
- If a provider changes model aliases, update `study/llm_model_presets.json` before the final benchmark.
- Manual web UI runs are lower quality evidence than API runs because temperature, system-message handling, and hidden product prompts may differ.
- Automated scores can reward wording patterns; human blind review is needed for actual tactical usefulness.
