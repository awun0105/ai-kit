"""Shared test doubles."""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.utils.function_calling import convert_to_openai_tool


class ScriptedModel(BaseChatModel):
    """Tool-calling chat model fake: replies in order, records what it saw and which tools were bound.

    A reply may be an ``AIMessage``, an exception to raise, or a callable taking the
    messages and returning an ``AIMessage``.
    """

    replies: list = []
    seen: list = []
    bound: list = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound.append(sorted(convert_to_openai_tool(t)["function"]["name"] for t in tools))
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages)
        return ChatResult(generations=[ChatGeneration(message=reply)])
