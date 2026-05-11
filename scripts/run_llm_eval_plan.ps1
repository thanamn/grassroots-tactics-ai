param(
  [ValidateSet("smoke", "default", "10plus", "prompt-ablation", "manual")]
  [string]$Mode = "smoke",

  [string]$RunId = "",

  [int]$Repeat = 2,

  [string]$Python = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = Split-Path -Parent $ScriptDir

if (-not $RunId) {
  $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
  $RunId = "llm_${Mode}_${stamp}"
}

if (-not $Python) {
  $venvPython = Join-Path $Root ".venv\Scripts\python.exe"
  if (Test-Path -LiteralPath $venvPython) {
    $Python = $venvPython
  } else {
    $Python = "python"
  }
}

if (-not (Test-Path -LiteralPath (Join-Path $Root ".env"))) {
  Write-Warning "No .env file found. Copy .env.example to .env and add API keys before running non-smoke modes."
}

function Invoke-ProjectPython {
  param([string[]]$ArgsList)
  Push-Location $Root
  try {
    & $Python @ArgsList
  } finally {
    Pop-Location
  }
}

Write-Host "[llm-plan] mode=$Mode run_id=$RunId repeat=$Repeat python=$Python"

switch ($Mode) {
  "smoke" {
    Invoke-ProjectPython @(
      "scripts/run_llm_benchmark.py",
      "--dry-run",
      "--run-id", $RunId
    )
    Invoke-ProjectPython @(
      "scripts/score_llm_benchmark.py",
      "--run-id", $RunId
    )
  }

  "default" {
    Invoke-ProjectPython @(
      "scripts/run_llm_benchmark.py",
      "--run-id", $RunId,
      "--repeat", "$Repeat",
      "--skip-missing-keys"
    )
    Invoke-ProjectPython @(
      "scripts/score_llm_benchmark.py",
      "--run-id", $RunId
    )
  }

  "10plus" {
    Invoke-ProjectPython @(
      "scripts/run_llm_benchmark.py",
      "--run-id", $RunId,
      "--prompt", "current_v2",
      "--repeat", "$Repeat",
      "--skip-missing-keys",
      "--preset", "deepseek_v4_pro",
      "--preset", "deepseek_v4_flash",
      "--preset", "openai_gpt_5_4_mini",
      "--preset", "openai_gpt_5_1",
      "--preset", "google_gemini_3_1_flash_lite",
      "--preset", "google_gemini_2_5_flash",
      "--preset", "google_gemini_2_5_flash_lite",
      "--preset", "mistral_small_latest",
      "--preset", "groq_llama_3_1_8b",
      "--preset", "groq_llama_3_3_70b",
      "--preset", "groq_qwen3_32b",
      "--preset", "together_llama_3_3_70b_turbo",
      "--preset", "xai_grok_4_20_non_reasoning"
    )
    Invoke-ProjectPython @(
      "scripts/score_llm_benchmark.py",
      "--run-id", $RunId
    )
  }

  "prompt-ablation" {
    Invoke-ProjectPython @(
      "scripts/run_llm_benchmark.py",
      "--run-id", $RunId,
      "--all-prompts",
      "--repeat", "$Repeat",
      "--skip-missing-keys",
      "--preset", "deepseek_v4_pro",
      "--preset", "google_gemini_2_5_flash",
      "--preset", "groq_qwen3_32b"
    )
    Invoke-ProjectPython @(
      "scripts/score_llm_benchmark.py",
      "--run-id", $RunId,
      "--baseline-model", "deepseek_v4_pro",
      "--baseline-prompt", "current_v2"
    )
  }

  "manual" {
    Invoke-ProjectPython @(
      "scripts/run_llm_benchmark.py",
      "--run-id", $RunId,
      "--manual-packets-only",
      "--all-presets",
      "--prompt", "current_v2"
    )
  }
}

Write-Host "[llm-plan] finished. Outputs are under data\eval\llm_runs\$RunId and data\eval\reports."
