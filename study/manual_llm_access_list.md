# Manual LLM Access List

Last updated: 2026-05-10

Use this list when a model is easier to test through a website than through an
API key. For every run, record the exact site, visible model/mode name, date,
and whether the prompt was pasted into a separate system field or into one chat
box.

## How To Paste Prompts

If the site has a separate System, Developer, or Instructions field:

1. Paste the packet's `SYSTEM` block into that field.
2. Paste the packet's `USER` block into the user/chat field.
3. Set temperature near 0.6 if the UI allows it.
4. Disable web search, browsing, code execution, tools, memory, and file search if possible.

If the site only has one chat box:

1. Use the packet's `COMBINED SINGLE-BOX PROMPT`.
2. Paste it as one message in a fresh chat.
3. Do not send the system block as one message and the user block as a second message unless the site gives no other option.

Reason: in a normal chat UI, the first message is not a real system prompt. It
is just another user message and may be interpreted differently by each product.
The combined prompt is more reproducible.

## Best Manual Test Set

Use these first if you want a credible 8-12 model comparison without collecting
many one-time API keys.

| Priority | Site | Link | Try These Models/Modes | System/User Field? | Cost/Access Notes | How To Record It |
|---|---|---|---|---|---|---|
| 1 | DeepSeek Chat | https://chat.deepseek.com | Instant / Expert if visible | No | Usually easiest production-family web check | `DeepSeek Chat, <mode>, web, date` |
| 1 | ChatGPT | https://chatgpt.com | GPT-5.x, GPT-5 mini/fast if selectable | No | Account required; available models depend on plan | `ChatGPT, visible model, web, date` |
| 1 | Claude | https://claude.ai | Sonnet, Opus, Haiku if selectable | No | Account required; available models depend on plan | `Claude.ai, visible model, web, date` |
| 1 | Google AI Studio | https://aistudio.google.com/prompts/new_chat | Gemini 3.1 Pro Preview, Gemini 3.1 Flash Lite, Gemini 2.5 Pro/Flash | Yes | Good free/low-friction option with real system instructions | `Google AI Studio, exact model, system field, date` |
| 1 | Gemini App | https://gemini.google.com | Gemini 3 / 2.5 visible default | No | Consumer app; less controlled than AI Studio | `Gemini app, visible model/mode, web, date` |
| 2 | Groq Console | https://console.groq.com/playground | Llama 3.3 70B, Llama 3.1 8B Instant, Qwen 3 32B, GPT-OSS 120B/20B if available | Usually yes | Free account may be enough for small manual runs | `Groq, exact model, playground, date` |
| 2 | OpenRouter Chat / Free Models | https://openrouter.ai/chat, https://openrouter.ai/collections/free-models | `openrouter/free`, free Qwen/Gemma/Llama variants, Claude/Gemini routes if available | Varies | Free models change often and can rate-limit | `OpenRouter, exact model or openrouter/free routed model, date` |
| 2 | HuggingChat | https://huggingface.co/chat | Omni router, Qwen, Llama, Gemma, Mistral/Mixtral models if listed | No | Free/open-model friendly; model list changes | `HuggingChat, exact selected model, date` |
| 2 | Qwen Chat | https://chat.qwen.ai | Qwen Max/Plus/Coder/Thinking modes if visible | No | Strong free/open-family option; exact model may vary | `Qwen Chat, visible model/mode, date` |
| 2 | Together Playground | https://api.together.ai/playground/chat | Llama 3.3 70B, Qwen, Kimi, DeepSeek, GPT-OSS if available | Usually yes | May require signup/free credits | `Together, exact model, playground, date` |
| 3 | Mistral Le Chat | https://chat.mistral.ai/chat | Default, Thinking, agent/model selector if visible | No | Easy, but exact model may be hidden in consumer chat | `Le Chat, visible mode/model or default, date` |
| 3 | Mistral AI Studio | https://console.mistral.ai | Mistral Medium/Large/Small, Magistral, Ministral if available | Usually yes | Better controlled than Le Chat; may require account | `Mistral Studio, exact model, date` |
| 3 | Grok | https://grok.com | Grok 4.3 / fast modes if visible | No | Access depends on xAI/X account/plan | `Grok, visible model/mode, date` |
| 4 | Perplexity | https://www.perplexity.ai | Sonar / model selector if visible | No | Not ideal for core LLM eval because search/RAG is product behavior | `Perplexity, model/mode, search on/off, date` |
| 4 | Chatbot Arena | https://lmarena.ai | Blind side-by-side only | No | Useful for preference intuition, not clean per-model scoring | `Arena, revealed model names, date` |

## Open / Open-Weight Models That Are Often Easy To Try

These are useful because they represent small, open, or hosted-open alternatives
to frontier closed models. Availability changes by site, so use the exact model
name shown in the UI.

| Model Family | Good Targets | Where To Try First | Why It Matters |
|---|---|---|---|
| Llama | Llama 3.3 70B, Llama 3.1 8B Instant | Groq, Together, HuggingChat, OpenRouter, Ollama | Strong open baseline; 8B is a speed/cheapness lower bound |
| Qwen | Qwen 3 32B, Qwen Max/Plus, Qwen Coder | Qwen Chat, Groq, Together, HuggingChat, OpenRouter, Ollama | Good multilingual/Thai candidate and strong open family |
| Gemma | Gemma 3/4 small and medium variants | HuggingChat, OpenRouter, Ollama | Small open Google-family comparison |
| Mistral/Mixtral | Mistral Small, Ministral, Mixtral, Magistral | Mistral Studio, HuggingChat, OpenRouter, Together | European/open-weight family; useful diversity |
| GPT-OSS | GPT-OSS 20B, GPT-OSS 120B | Groq, Together, Ollama, OpenRouter | Open-weight reasoning-style comparison |
| DeepSeek open family | DeepSeek-R1 style models, V3/V4 hosted modes | DeepSeek Chat, Together, OpenRouter, Ollama | Production-related and strong reasoning comparison |
| Kimi | Kimi K2/K2.5 if visible | Together, OpenRouter | Large non-Llama architecture diversity |

## Recommended Manual Evaluation Order

1. Run `Mode manual` to generate packets.
2. Test `current_v2` on 8-12 models/sites.
3. Score/import all answers.
4. Pick 3 representative models and repeat with prompt variants.
5. Do human blind rating only on the top 3-5 model outputs to avoid exhausting participants.

Suggested first 10:

- DeepSeek Chat Expert or DeepSeek V4 Pro API if available.
- DeepSeek Chat Instant or V4 Flash.
- ChatGPT best available GPT-5.x.
- ChatGPT smaller/fast model if selectable.
- Claude Sonnet.
- Claude Haiku or Opus, whichever is available.
- Gemini 3.1 Pro Preview in AI Studio.
- Gemini 2.5 Flash or 3.1 Flash Lite in AI Studio.
- Groq Llama 3.3 70B or Qwen 3 32B.
- HuggingChat/Qwen Chat open model.

## Manual Result Import

After copying a response into a `.txt` file:

```powershell
python scripts\import_manual_llm_result.py `
  --run-id llm_manual_v1 `
  --model-preset anthropic_claude_sonnet_4_6 `
  --prompt current_v2 `
  --case explainer_defensive_stretch_en `
  --response-file path\to\answer.txt
```

If the site/model is not in `study/llm_model_presets.json`, either add a new
manual preset or reuse the closest preset and note the exact model in a separate
run log.

## Reporting Caveat For The Paper

Phrase manual web runs like this:

"For models without convenient API access, we used the provider's public web or
playground interface with the same prompt packet. These results represent
consumer/playground behavior rather than pure API behavior, because the websites
may include hidden system prompts, routing, safety wrappers, memory, search, or
tool settings."
