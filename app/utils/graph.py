"""This file contains the graph utilities for the application."""

from typing import (
    Optional,
    cast,
)

import tiktoken
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
)
from langchain_core.messages import trim_messages as _trim_messages

from app.core.config import settings
from app.core.logging import logger
from app.schemas import Message

# Cache tiktoken encoding at module level — thread-safe and reusable
try:
    _TIKTOKEN_ENCODING = tiktoken.encoding_for_model(settings.DEFAULT_LLM_MODEL)
except KeyError:
    _TIKTOKEN_ENCODING = tiktoken.get_encoding("cl100k_base")


def _count_tokens_tiktoken(messages: list) -> int:
    """Count tokens locally using tiktoken — no API call needed."""
    num_tokens = 0
    for message in messages:
        # Every message has overhead tokens for role/name
        num_tokens += 4
        if isinstance(message, dict):
            for _, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(_TIKTOKEN_ENCODING.encode(value))
        elif isinstance(message, BaseMessage):
            content = message.content
            if isinstance(content, str):
                num_tokens += len(_TIKTOKEN_ENCODING.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, str):
                        num_tokens += len(_TIKTOKEN_ENCODING.encode(block))
                    elif isinstance(block, dict) and "text" in block:
                        num_tokens += len(_TIKTOKEN_ENCODING.encode(block["text"]))
    num_tokens += 2  # every reply is primed with assistant
    return num_tokens


def dump_messages(messages: list[Message]) -> list[dict]:
    """Dump the messages to a list of dictionaries.

    Args:
        messages (list[Message]): The messages to dump.

    Returns:
        list[dict]: The dumped messages.
    """
    return [message.model_dump() for message in messages]


def extract_text_content(content: str | list) -> str:
    """Extract plain text from an LLM content value.

    Handles both the simple string format and the structured block list returned
    by GPT-5 / Responses API models:
        [{'type': 'reasoning', ...}, {'type': 'text', 'text': '...'}]

    Args:
        content: Raw content from a LangChain BaseMessage.

    Returns:
        Plain text string (empty string when nothing extractable is present).
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif block.get("type") == "reasoning":
                logger.debug(
                    "reasoning_block_received",
                    reasoning_id=block.get("id"),
                    has_summary=bool(block.get("summary")),
                )
    return "".join(parts)


def latest_user_text(messages: list[BaseMessage]) -> str:
    """Return the text of the most recent user message in a LangGraph message list.

    Walks the list in reverse so the assistant reply, tool calls and tool
    messages after it are skipped. Returns an empty string when there is no
    user message.
    """
    last_user = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    return extract_text_content(last_user.content) if last_user else ""


def process_llm_response(response: BaseMessage) -> BaseMessage:
    """Normalise a raw LLM response so that ``response.content`` is always a plain string, regardless of the provider's content format.

    Args:
        response: The raw response from the LLM.

    Returns:
        The same BaseMessage instance with ``content`` set to a plain string.
    """
    if isinstance(response.content, list):
        response.content = extract_text_content(response.content)
        logger.debug(
            "processed_structured_content",
            content_block_count=len(response.content),
            extracted_length=len(response.content),
        )
    return response


def _current_turn(messages: list) -> list:
    """The messages from the latest user message to the end (the question and its tool round trips)."""
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            return messages[index:]
    return messages


def prepare_messages(messages: list[Message], system_prompt: str, max_tokens: Optional[int] = None) -> list[Message]:
    """Prepare the messages for the LLM: the system prompt, then as much recent history as fits.

    History is cut at user messages, oldest turn first, so a tool result never loses the
    call that produced it. The current turn is always kept whole, even when it alone is
    over the budget (a large RAG result): dropping it would send the model no question.

    Args:
        messages (list[Message]): The messages to prepare.
        system_prompt (str): The system prompt to use.
        max_tokens (Optional[int]): History budget in tokens; defaults to ``HISTORY_MAX_TOKENS``.

    Returns:
        list[Message]: The prepared messages.
    """
    budget = settings.HISTORY_MAX_TOKENS if max_tokens is None else max_tokens
    try:
        trimmed_messages = _trim_messages(
            dump_messages(messages),
            strategy="last",
            token_counter=_count_tokens_tiktoken,
            max_tokens=budget,
            start_on="human",
            include_system=False,
            allow_partial=False,
        )
    except ValueError as e:
        # Handle unrecognized content blocks (e.g., reasoning blocks from GPT-5)
        if "Unrecognized content block type" in str(e):
            logger.warning(
                "token_counting_failed_skipping_trim",
                error=str(e),
                message_count=len(messages),
            )
            # Skip trimming and return all messages
            trimmed_messages = messages
        else:
            raise
    if messages and not trimmed_messages:
        trimmed_messages = _current_turn(messages)
        logger.warning(
            "history_current_turn_over_budget",
            max_tokens=budget,
            turn_tokens=_count_tokens_tiktoken(trimmed_messages),
        )

    # langchain-core >= 1.6 types trim_messages() as returning BaseMessage;
    # cast back to the declared return type to keep pyright passing.
    return cast(list[Message], [Message(role="system", content=system_prompt)] + trimmed_messages)
