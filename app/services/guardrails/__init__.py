"""Guardrails: input, tool and output rails for the agent."""

from typing import (
    Literal,
    Optional,
    cast,
)

from langgraph.constants import TAG_NOSTREAM

from app.core.config import settings
from app.services.llm import llm_service
from app.services.guardrails.detectors import (
    Finding,
    find_injections,
    find_sensitive,
    normalize_for_security,
    redact,
)
from app.services.guardrails.llm_rail import (
    LLMPolicyRail,
    PolicyVerdict,
)
from app.services.guardrails.policy import (
    Guardrails,
    InputDecision,
    OutputDecision,
    PIIAction,
    StreamingRedactor,
    ToolDecision,
    is_affirmative,
)

_guardrails: Optional[Guardrails] = None


def build_guardrails_from_settings() -> Guardrails:
    """Assemble the rails from the GUARD_* settings."""
    pii_action = settings.GUARD_INPUT_PII_ACTION
    if pii_action not in ("block", "redact", "allow"):
        raise ValueError(f"GUARD_INPUT_PII_ACTION must be block, redact or allow, got {pii_action!r}")
    fail_mode = settings.GUARD_LLM_RAIL_FAIL_MODE
    if fail_mode not in ("open", "closed"):
        raise ValueError(f"GUARD_LLM_RAIL_FAIL_MODE must be open or closed, got {fail_mode!r}")

    llm_rail = None
    if settings.GUARD_LLM_RAIL_ENABLED:
        model = settings.GUARD_LLM_RAIL_MODEL or None

        async def classify(messages: list[dict], schema: type[PolicyVerdict]) -> PolicyVerdict:
            # nostream: the verdict must never reach a client reading the graph's message stream.
            return await llm_service.call(
                messages, model_name=model, response_format=schema, config={"tags": [TAG_NOSTREAM]}
            )

        llm_rail = LLMPolicyRail(
            settings.GUARD_LLM_RAIL_POLICY, classify, fail_mode=cast(Literal["open", "closed"], fail_mode)
        )

    return Guardrails(
        enabled=settings.GUARD_ENABLED,
        block_injection=settings.GUARD_BLOCK_INJECTION,
        input_pii_action=cast(PIIAction, pii_action),
        output_redaction=settings.GUARD_OUTPUT_REDACTION,
        pii_allowlist=tuple(settings.GUARD_PII_ALLOWLIST),
        denylist=tuple(settings.GUARD_DENYLIST),
        approval_tools=frozenset(settings.GUARD_APPROVAL_TOOLS),
        egress_allowlist=tuple(settings.GUARD_EGRESS_ALLOWLIST),
        screen_tool_output=settings.GUARD_SCREEN_TOOL_OUTPUT,
        llm_rail=llm_rail,
        refusal_message=settings.GUARD_REFUSAL_MESSAGE,
    )


def get_guardrails() -> Guardrails:
    """Return the process-wide rails, building them on first use."""
    global _guardrails
    if _guardrails is None:
        _guardrails = build_guardrails_from_settings()
    return _guardrails


def set_guardrails(guardrails: Optional[Guardrails]) -> None:
    """Replace the process-wide rails (used by tests)."""
    global _guardrails
    _guardrails = guardrails


__all__ = [
    "Finding",
    "Guardrails",
    "InputDecision",
    "LLMPolicyRail",
    "OutputDecision",
    "PolicyVerdict",
    "StreamingRedactor",
    "ToolDecision",
    "build_guardrails_from_settings",
    "find_injections",
    "find_sensitive",
    "get_guardrails",
    "is_affirmative",
    "normalize_for_security",
    "redact",
    "set_guardrails",
]
