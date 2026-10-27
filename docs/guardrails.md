# Guardrails

Defence in depth around the agent, in the order a request meets the layers:

```
user message ─> input_guard node ──blocked──> refusal, turn ends (model never called)
                  │ normalize (NFKC, zero-width chars)
                  │ prompt-injection patterns (EN + VI)
                  │ PII / secrets: redact (default) | block | allow
                  │ optional LLM policy rail (scope, jailbreak, harmful, privacy)
                  v
               chat node ──tool calls──> tool_call node
                  │                         │ egress allowlist + no secrets in arguments
                  │                         │ high-risk tools pause for human approval
                  │                         │ instruction-like text in tool output neutralized
                  v                         v
               output rail: PII / secrets redacted in the stored answer and in the stream
```

Deterministic rails are **on by default** and need no model or API key. The LLM
rail is off by default because every message then costs one extra model call.

## What each rail catches

Measured with the D24 Eval-Guard fixtures (`tests/fixtures/guardrails/`), see
`tests/test_guardrails.py`:

| Set | Deterministic rails (`GUARD_INPUT_PII_ACTION=block`) | + LLM rail |
|---|---|---|
| 20 attacks | 14 blocked: all PII (4), jailbreak (5) and prompt-injection (5) attacks | 20 (with a scripted classifier) |
| 8 clean controls | 8 allowed, 0 false positives | 8 allowed |
| 4 model answers | CCCD and password redacted; clean answers untouched (support domain allowlisted) | – |

The 6 attacks regex cannot catch are semantic: off-topic requests (poems, recipes,
crypto prices, maths, films) and asking for someone else's ID number. That is the
LLM rail's job, configured with a one-sentence scope in `GUARD_LLM_RAIL_POLICY`.

Detected data: email, Vietnamese phone numbers (`0…`, `+84…`, with separators),
CCCD (12 digits, province code checked), CMND (9 digits, only next to an ID keyword
so amounts are not redacted), credit cards (Luhn-checked), API keys (OpenAI-style,
AWS, GitHub, Slack, JWT, Bearer), private keys, password values (only values with a
digit or symbol, so "mật khẩu mặc định là gì?" is left alone), and your own
`GUARD_DENYLIST` terms, which also match when spaced out or punctuated.

## Human approval for risky tools

List tool names in `GUARD_APPROVAL_TOOLS`. When the model calls one, the graph pauses
with an `interrupt` and the API returns the approval question as the assistant
message. The user's next message resumes the graph. Only a reply that is entirely a yes
approves (`yes`, `ok`, `đồng ý`, `có`, `duyệt nhé`, `Xác nhận.`, ...); anything else, including
`có lẽ không` or `ok nhưng đừng`, cancels and the model is told the user declined.
All approvals are collected before any tool runs, so a resumed turn never repeats a
side effect.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `GUARD_ENABLED` | `true` | Master switch |
| `GUARD_BLOCK_INJECTION` | `true` | Refuse messages matching prompt-injection patterns |
| `GUARD_INPUT_PII_ACTION` | `redact` | `redact`: model and history get `[REDACTED:<kind>]`; `block`: refuse; `allow` |
| `GUARD_OUTPUT_REDACTION` | `true` | Redact PII/secrets in answers, including streamed ones |
| `GUARD_PII_ALLOWLIST` | – | Comma list of values or `@domain` suffixes never redacted (public support contacts) |
| `GUARD_DENYLIST` | – | Comma list of application secrets that must never be shown (internal hosts, demo passwords) |
| `GUARD_APPROVAL_TOOLS` | – | Comma list of tool names that need human approval |
| `GUARD_EGRESS_ALLOWLIST` | – | When set, URLs in tool arguments must be https on these hosts or their subdomains. Secrets in tool arguments are blocked whether or not it is set |
| `GUARD_SCREEN_TOOL_OUTPUT` | `true` | Neutralize instruction-like text in tool results |
| `GUARD_LLM_RAIL_ENABLED` | `false` | Add the LLM policy rail |
| `GUARD_LLM_RAIL_POLICY` | general assistant | What the assistant is for, e.g. "Answers employees' questions about HR policy." |
| `GUARD_LLM_RAIL_MODEL` | default model | A cheaper model name from the LLM registry |
| `GUARD_LLM_RAIL_FAIL_MODE` | `open` | On classifier errors: `open` lets the message through, `closed` refuses |
| `GUARD_REFUSAL_MESSAGE` | English sentence | Reply sent when a message is blocked |

Every intervention is logged (`guard_*` events) and counted in the Prometheus metric
`guardrail_events_total{rail, action}`.

## Limits

- Patterns are a signal, not a proof. Paraphrased attacks get through the deterministic
  rails; enable the LLM rail for anything public-facing and red-team it with
  [promptfoo](https://github.com/promptfoo/promptfoo) or [garak](https://github.com/NVIDIA/garak).
- Tool-output screening also rewrites legitimate documents that *quote* attacks
  (security training material). Turn it off for such corpora.
- Names and addresses are not detected; that needs NER. [Presidio](https://github.com/data-privacy-stack/presidio)
  provides it for English, but its spaCy models do not cover Vietnamese, which is why
  the kit uses validated patterns (the D24 lab also used Presidio only through custom
  pattern recognizers for CCCD/CMND/phone).
- Streaming holds back the last 96 characters until more text arrives, so the final
  words of a streamed answer arrive together at the end.

## Code

| File | Responsibility |
|---|---|
| `app/services/guardrails/detectors.py` | Normalization, injection patterns, PII/secret rules, `redact` |
| `app/services/guardrails/llm_rail.py` | `LLMPolicyRail` with structured `PolicyVerdict` |
| `app/services/guardrails/policy.py` | `Guardrails` facade, tool policy, `StreamingRedactor` |
| `app/core/langgraph/graph.py` | `input_guard` node, tool authorization, output redaction, stream redaction |

## Provenance

| Part | Source |
|---|---|
| Normalization, injection and PII patterns, egress allowlist, HITL for high-risk actions, untrusted content as data | Author's solution to the AI20K D11 Guardrails-HITL lab (`K4-L3-DAY11-…-Guardrails-HITL-Responsible-AI`, branch `solution`: `src/guardrails/*.py`, `src/agents/security_boundary.py`, `src/assignment/pipeline.py`) |
| Vietnamese attack styles, LLM self-check split, fixtures | AI20K D24 Eval-Guard lab (`fixtures/`, `doc/ASSIGNMENT.md` Phase C) |
| Layer design | NeMo Guardrails input/output/execution rails; OWASP LLM Top 10 (LLM01 prompt injection, LLM02 sensitive information disclosure, LLM06 excessive agency) |
