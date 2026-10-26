"""A LangChain chat model backed by the resilient model ladder.

``ResilientChatModel`` makes the kit's reliability layer (fallback ladder,
breakers, retry budget, single retry layer) available wherever LangChain
expects a ``BaseChatModel``: ``deepagents.create_deep_agent``,
``langchain.agents.create_agent``, structured output, evaluation libraries.
"""

from typing import (
    Any,
    Optional,
    Sequence,
    override,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import (
    ChatGeneration,
    ChatResult,
)
from langchain_core.runnables import Runnable
from pydantic import (
    ConfigDict,
    Field,
)


class ResilientChatModel(BaseChatModel):
    """Chat model that routes every call through ``LLMService.ainvoke_with_tools``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    service: Any = Field(description="An LLMService (or anything with ainvoke_with_tools)")
    start_model: Optional[str] = Field(default=None, description="Registry model the ladder starts at")
    tools: list[Any] = Field(default_factory=list)
    tool_kwargs: dict[str, Any] = Field(default_factory=dict)

    @property
    @override
    def _llm_type(self) -> str:
        return "resilient-ladder"

    @property
    @override
    def _identifying_params(self) -> dict[str, Any]:
        return {"start_model": self.start_model, "tools": [getattr(t, "name", str(t)) for t in self.tools]}

    @override
    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Optional[Any] = None,
        **kwargs: Any,
    ) -> Runnable:
        """Return a copy whose calls bind ``tools`` on every rung of the ladder."""
        tool_kwargs = dict(kwargs)
        if tool_choice is not None:
            tool_kwargs["tool_choice"] = tool_choice
        return self.model_copy(update={"tools": list(tools), "tool_kwargs": tool_kwargs})

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("ResilientChatModel is async-only; use ainvoke / astream")

    @override
    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # The provider call is not given the run manager's callbacks: this outer run already
        # records the returned message and its token usage, and a nested LLM run would count it twice.
        message = await self.service.ainvoke_with_tools(
            messages, tools=self.tools, tool_kwargs=self.tool_kwargs, model_name=self.start_model
        )
        return ChatResult(generations=[ChatGeneration(message=message)])
