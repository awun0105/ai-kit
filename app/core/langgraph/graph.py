"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

import asyncio
from typing import (
    AsyncGenerator,
    Optional,
)
from urllib.parse import quote_plus

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.errors import (
    GraphBubbleUp,
    GraphInterrupt,
)
from langgraph.graph import (
    END,
    StateGraph,
)
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph.state import (
    Command,
    CompiledStateGraph,
)
from langgraph.types import (
    StateSnapshot,
    interrupt,
)
from psycopg import (
    AsyncConnection,
    sql,
)
from psycopg.rows import (
    DictRow,
    dict_row,
)
from psycopg_pool import AsyncConnectionPool

from app.core.config import settings
from app.core.kit import (
    get_answer_cache,
    get_cost_guard,
    get_guardrails,
)
from app.core.langgraph.tools import tools
from app.core.logging import logger
from app.core.metrics import (
    answer_cache_events_total,
    rag_citation_issues_total,
)
from app.core.observability import langfuse_callback_handler
from app.core.prompts import load_system_prompt
from app.schemas import (
    GraphState,
    Message,
)
from app.services.guardrails import is_affirmative
from app.services.llm import llm_service
from app.services.cost_guard import TurnUsage
from app.services.memory import memory_service
from app.services.rag.citations import (
    check_answer,
    cited_labels,
)
from app.services.rag.pipeline import INSUFFICIENT_EVIDENCE
from app.utils import (
    dump_messages,
    extract_text_content,
    prepare_messages,
    process_llm_response,
)
from app.utils.tasks import create_background_task

PostgresConnPool = AsyncConnectionPool[AsyncConnection[DictRow]]

NO_MEMORY = "No relevant memory found."

# Tools that call interrupt() to ask the human; run before any other tool of the same turn.
INTERRUPTING_TOOLS = frozenset({"ask_human"})


class LangGraphAgent:
    """Manages the LangGraph Agent/workflow and interactions with the LLM.

    This class handles the creation and management of the LangGraph workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use the LLM service with tools bound
        self.llm_service = llm_service
        self.llm_service.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[PostgresConnPool] = None
        self._graph: Optional[CompiledStateGraph] = None
        logger.info(
            "langgraph_agent_initialized",
            model=settings.DEFAULT_LLM_MODEL,
            environment=settings.ENVIRONMENT.value,
        )

    def register_tools(self, extra: list[BaseTool]) -> None:
        """Add tools discovered at runtime (e.g. from MCP servers) and rebind the models.

        The compiled graph looks tools up by name on every call, so it does not
        need rebuilding. Built-in tools win on a name clash.
        """
        new = [t for t in extra if t.name not in self.tools_by_name]
        clashes = [t.name for t in extra if t.name in self.tools_by_name]
        if clashes:
            logger.warning("tools_name_clash_skipped", tools=clashes)
        self.tools_by_name.update({t.name: t for t in new})
        self.llm_service.bind_tools(list(self.tools_by_name.values()))
        logger.info("tools_registered", added=[t.name for t in new], total=len(self.tools_by_name))

    async def _get_connection_pool(self) -> PostgresConnPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: The open connection pool.

        Raises:
            Exception: If the pool cannot be created, in every environment.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                connection_url = (
                    "postgresql://"
                    f"{quote_plus(settings.POSTGRES_USER)}:{quote_plus(settings.POSTGRES_PASSWORD)}"
                    f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
                )

                self._connection_pool = AsyncConnectionPool(
                    connection_url,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                        "row_factory": dict_row,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.exception(
                    "connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value
                )
                # Never degrade silently: the checkpointer is the only store for
                # conversation history and HITL resume state. Serving without it
                # loses data rather than surfacing an outage.
                raise e
        return self._connection_pool

    @staticmethod
    def _check_citations(messages: list, answer: str) -> None:
        """Flag answers citing a source number that knowledge_search never returned (observability only)."""
        source_count = 0
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, ToolMessage) and message.name == "knowledge_search":
                labels = cited_labels(extract_text_content(message.content))
                source_count = max(source_count, max(labels, default=0))
        if not source_count:
            return
        result = check_answer(answer, source_count)
        if result.out_of_range:
            rag_citation_issues_total.labels(kind="out_of_range").inc()
            logger.warning("rag_citation_out_of_range", labels=result.out_of_range, sources=source_count)
        elif not result.cited and INSUFFICIENT_EVIDENCE not in answer:
            rag_citation_issues_total.labels(kind="uncited").inc()
            logger.info("rag_answer_without_citations", sources=source_count)

    async def _input_guard(self, state: GraphState) -> Command:
        """Run the input rails on the newest user message before the model sees it.

        A blocked message gets the refusal as the reply and ends the turn. PII is
        replaced in the stored message too, so it never reaches the checkpointer
        in clear text when the rails redact or block it.
        """
        last = state.messages[-1] if state.messages else None
        if not isinstance(last, HumanMessage):
            return Command(goto="chat")

        guard = get_guardrails()
        original = extract_text_content(last.content)
        decision = await guard.check_input(original)
        updates: list[BaseMessage] = []
        pii = [f for f in decision.findings if f.kind != "prompt_injection"]
        if pii:
            stored = decision.text if decision.allowed else guard.redact_for_storage(original)
            updates.append(HumanMessage(content=stored, id=last.id))
        if not decision.allowed:
            updates.append(AIMessage(content=guard.refusal_message))
            return Command(update={"messages": updates}, goto=END)
        return Command(update={"messages": updates} if updates else {}, goto="chat")

    async def _chat(self, state: GraphState, config: RunnableConfig) -> Command:
        """Process the chat state and generate a response.

        Args:
            state (GraphState): The current state of the conversation.
            config (RunnableConfig): The runnable configuration for this invocation.

        Returns:
            Command: Command object with updated state and next node to execute.
        """
        metadata = config.get("metadata", {})
        # An answer bound for the shared answer cache is served to every user: build it from a
        # prompt that carries no one's name or long-term memory.
        neutral = bool(metadata.get("neutral_prompt"))
        username = None if neutral else metadata.get("username")
        thread_id = config.get("configurable", {}).get("thread_id")
        SYSTEM_PROMPT = load_system_prompt(
            username=username, long_term_memory=NO_MEMORY if neutral else state.long_term_memory
        )

        # Prepare messages with system prompt
        messages = prepare_messages(state.messages, SYSTEM_PROMPT)

        try:
            # One resilient call: retries, breakers and fallback across models happen inside,
            # and the invoker records the latency of the model that answered.
            response_message = await self.llm_service.call(dump_messages(messages))

            # Process response to handle structured content blocks
            response_message = process_llm_response(response_message)

            logger.info(
                "llm_response_generated",
                session_id=thread_id,
                model=(response_message.response_metadata or {}).get("model_name"),
                environment=settings.ENVIRONMENT.value,
            )

            # Determine next node based on whether there are tool calls
            if isinstance(response_message, AIMessage) and response_message.tool_calls:
                goto = "tool_call"
            else:
                goto = END
                # Output rail: the stored and returned answer never carries PII/secrets.
                checked = get_guardrails().check_output(extract_text_content(response_message.content))
                if checked.redacted:
                    response_message.content = checked.text
                self._check_citations(state.messages, checked.text)

            return Command(update={"messages": [response_message]}, goto=goto)
        except Exception:
            # Re-raise the original error: its class (AllProvidersFailedError, a client error) is
            # what callers and the stale-if-error path need, and the traceback stays intact.
            logger.exception(
                "llm_call_failed_all_models", session_id=thread_id, environment=settings.ENVIRONMENT.value
            )
            raise

    # Define our tool node
    async def _tool_call(self, state: GraphState) -> Command:
        """Process tool calls from the last message under the tool rails.

        Every call is authorized first: calls the policy forbids get an error
        result, and high-risk tools pause the graph for human approval. All
        approvals are collected before any tool runs, because a resumed node
        re-executes from the start and side effects must not repeat.

        Args:
            state: The current agent state containing messages and tool calls.

        Returns:
            Command: Command object with updated messages and routing back to chat.
        """
        guard = get_guardrails()
        tool_calls = state.messages[-1].tool_calls

        verdicts: list[Optional[str]] = []  # None = run it, otherwise the reason it was not run
        for tool_call in tool_calls:
            if tool_call["name"] not in self.tools_by_name:
                verdicts.append(f"unknown tool {tool_call['name']!r}")
                continue
            decision = guard.authorize_tool_call(tool_call["name"], tool_call["args"])
            if not decision.allowed:
                logger.info("guard_tool_call_blocked", tool=tool_call["name"], reason=decision.reason)
                verdicts.append(f"blocked by policy: {decision.reason}")
            elif decision.needs_approval:
                answer = interrupt(guard.approval_question(tool_call["name"], tool_call["args"]))
                approved = is_affirmative(answer)
                logger.info("guard_tool_call_reviewed", tool=tool_call["name"], approved=approved)
                verdicts.append(None if approved else "the user did not approve this action")
            else:
                verdicts.append(None)

        async def _execute_tool(tool_call: dict, refusal: Optional[str]) -> ToolMessage:
            artifact = None
            if refusal is not None:
                content = f"Tool call not executed: {refusal}."
            else:
                try:
                    # Invoking with the full ToolCall returns a ToolMessage, which keeps
                    # the artifact of tools declared with response_format="content_and_artifact".
                    result = await self.tools_by_name[tool_call["name"]].ainvoke({**tool_call, "type": "tool_call"})
                    raw = extract_text_content(result.content if isinstance(result, ToolMessage) else result)
                    artifact = result.artifact if isinstance(result, ToolMessage) else None
                    content = guard.screen_tool_result(tool_call["name"], raw)
                    if content != raw:
                        # The artifact (e.g. MCP structured content) holds the same untrusted
                        # text; drop it so nothing downstream renders the unscreened copy.
                        artifact = None
                except GraphBubbleUp:
                    raise  # interrupt() inside a tool (ask_human, MCP elicitation) must pause the graph
                except Exception as e:
                    # Report the error to the model instead of retrying the node: a node
                    # retry would re-run every tool in it, repeating side effects.
                    logger.warning("tool_call_failed", tool=tool_call["name"], error=str(e))
                    content = f"Tool error: {type(e).__name__}: {e}"
            return ToolMessage(
                content=content,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
                artifact=artifact,
            )

        # A tool that pauses for the human re-runs when the graph resumes, and so would every tool
        # executed next to it. Run those first, one by one; the rest only once they all returned.
        calls = list(zip(tool_calls, verdicts, strict=True))
        done: dict[str, ToolMessage] = {}
        for tool_call, refusal in calls:
            if tool_call["name"] in INTERRUPTING_TOOLS:
                done[tool_call["id"]] = await _execute_tool(tool_call, refusal)
        rest = [(tc, v) for tc, v in calls if tc["id"] not in done]
        for message in await asyncio.gather(*[_execute_tool(tc, v) for tc, v in rest]):
            done[message.tool_call_id] = message

        return Command(update={"messages": [done[tc["id"]] for tc in tool_calls]}, goto="chat")

    async def create_graph(self, checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
        """Create and configure the LangGraph workflow.

        Args:
            checkpointer: Optional checkpointer to use instead of the Postgres one.
                Tests pass an in-memory saver so the graph runs without a database.

        Returns:
            CompiledStateGraph: The configured LangGraph instance, always with a checkpointer.

        Raises:
            Exception: If the graph cannot be built, in every environment.
        """
        if self._graph is None:
            try:
                graph_builder = StateGraph(GraphState)
                graph_builder.add_node("input_guard", self._input_guard, destinations=("chat", END))
                graph_builder.add_node("chat", self._chat, destinations=("tool_call", END))
                graph_builder.add_node("tool_call", self._tool_call, destinations=("chat",))
                graph_builder.set_entry_point("input_guard")
                graph_builder.set_finish_point("chat")

                if checkpointer is None:
                    # Raises if the pool cannot be created — no checkpointer, no service.
                    connection_pool = await self._get_connection_pool()
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()

                self._graph = graph_builder.compile(
                    checkpointer=checkpointer, name=f"{settings.PROJECT_NAME} Agent ({settings.ENVIRONMENT.value})"
                )

                logger.info(
                    "graph_created",
                    graph_name=f"{settings.PROJECT_NAME} Agent",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.exception("graph_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                raise e

        return self._graph

    async def _get_graph(self) -> CompiledStateGraph:
        """Return the compiled graph, creating it on first access.

        Raises:
            Exception: Propagated from ``create_graph()`` when initialisation
                fails. Callers can rely on the return being non-``None``.
        """
        if self._graph is None:
            self._graph = await self.create_graph()
        return self._graph

    @staticmethod
    def _awaiting_human(state: StateSnapshot) -> bool:
        """True only when a task is paused on ``interrupt()``.

        ``state.next`` alone is not enough: a node that raised (all providers down)
        also leaves ``next`` set, and treating the user's next message as a resume
        value would silently drop their question.
        """
        return bool(state.next) and any(task.interrupts for task in state.tasks)

    async def _cache_question(self, state: StateSnapshot, messages: list[Message]) -> Optional[str]:
        """Return the question when this turn may use the answer cache, else None.

        Only the first question of a new conversation qualifies (later turns depend
        on context), and only if the deterministic input rails find nothing in it.
        """
        cache = get_answer_cache()
        if cache is None or state.next or (state.values or {}).get("messages"):
            return None
        if messages[-1].role != "user" or sum(m.role == "user" for m in messages) != 1:
            return None
        question = messages[-1].content
        decision = await get_guardrails().check_input(question, use_llm_rail=False)
        if not decision.allowed or decision.findings or not cache.is_cacheable(question):
            return None
        return question

    async def _answer_from_cache(
        self, graph: CompiledStateGraph, config: RunnableConfig, question: str, allow_stale: bool
    ) -> Optional[str]:
        """Serve a cached answer and record the turn in the conversation history."""
        cache = get_answer_cache()
        hit = await cache.lookup(question, allow_stale=allow_stale) if cache else None
        if hit is None:
            answer_cache_events_total.labels(outcome="miss").inc()
            return None
        current = await graph.aget_state(config)
        existing = (current.values or {}).get("messages") or []
        already_has_question = bool(existing) and isinstance(existing[-1], HumanMessage)
        new_messages: list[BaseMessage] = [] if already_has_question else [HumanMessage(content=question)]
        new_messages.append(AIMessage(content=hit.answer))
        await graph.aupdate_state(config, {"messages": new_messages}, as_node="chat")
        outcome = "hit_fresh" if hit.fresh else "hit_stale"
        answer_cache_events_total.labels(outcome=outcome).inc()
        logger.info("answer_cache_served", outcome=outcome, similarity=round(hit.similarity, 4), age_s=int(hit.age_s))
        return hit.answer

    async def _store_in_cache(self, question: str, final_messages: list[BaseMessage]) -> None:
        """Cache the final answer if it was built from knowledge_search results."""
        cache = get_answer_cache()
        last = final_messages[-1] if final_messages else None
        if cache is None or not isinstance(last, AIMessage) or last.tool_calls:
            return
        sources: list[str] = []
        for message in reversed(final_messages):
            if isinstance(message, HumanMessage):
                break
            if isinstance(message, ToolMessage) and message.name == "knowledge_search":
                sources.extend(message.artifact or [])
        if sources and await cache.store_answer(question, extract_text_content(last.content), sources):
            answer_cache_events_total.labels(outcome="stored").inc()

    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> list[Message]:
        """Get a response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.

        Returns:
            list[Message]: The response from the LLM.
        """
        graph = await self._get_graph()
        usage = TurnUsage()  # every model call of this turn, nested ones included, for the cost guard
        callbacks: list[BaseCallbackHandler] = [usage]
        if settings.LANGFUSE_TRACING_ENABLED:
            callbacks.append(langfuse_callback_handler)
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }

        try:
            # Run state check and memory search concurrently to save 200-500ms
            state, relevant_memory = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search(user_id, get_guardrails().redact_for_storage(messages[-1].content)),
            )

            await get_cost_guard().check(user_id or "anonymous")
            question = await self._cache_question(state, messages)
            if question:
                config["metadata"]["neutral_prompt"] = True  # this answer may be cached for everyone
                cached = await self._answer_from_cache(graph, config, question, allow_stale=False)
                if cached is not None:
                    return [Message(role="assistant", content=cached)]

            try:
                if self._awaiting_human(state):
                    logger.info("resuming_interrupted_graph", session_id=session_id, next_nodes=state.next)
                    response = await graph.ainvoke(
                        Command(resume=messages[-1].content),
                        config=config,
                    )
                else:
                    relevant_memory = relevant_memory or NO_MEMORY
                    response = await graph.ainvoke(
                        input={"messages": dump_messages(messages), "long_term_memory": relevant_memory},
                        config=config,
                    )
            except GraphInterrupt:
                raise
            except Exception:
                # stale-if-error: a slightly old answer to a knowledge question beats an error.
                stale = await self._answer_from_cache(graph, config, question, allow_stale=True) if question else None
                if stale is None:
                    raise
                return [Message(role="assistant", content=stale)]

            # Check if the graph was interrupted during this invocation
            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
                return [Message(role="assistant", content=str(interrupt_value))]

            create_background_task(memory_service.add(user_id, response["messages"]))
            create_background_task(get_cost_guard().record(user_id or "anonymous", usage))
            if question:
                create_background_task(self._store_in_cache(question, response["messages"]))
            return self.__process_messages(response["messages"])
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted", session_id=session_id, interrupt_value=str(interrupt_value))
            return [Message(role="assistant", content=str(interrupt_value))]
        except Exception as e:
            logger.exception("get_response_failed", error=str(e), session_id=session_id)
            raise

    async def get_stream_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
        username: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Get a stream response from the LLM.

        Args:
            messages (list[Message]): The messages to send to the LLM.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.
            username (Optional[str]): The display name of the user.

        Yields:
            str: Tokens of the LLM response.
        """
        usage = TurnUsage()  # every model call of this turn, nested ones included, for the cost guard
        callbacks: list[BaseCallbackHandler] = [usage]
        if settings.LANGFUSE_TRACING_ENABLED:
            callbacks.append(langfuse_callback_handler)
        config: RunnableConfig = {
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks,
            "metadata": {
                "user_id": user_id,
                "username": username,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": settings.DEBUG,
            },
        }
        graph = await self._get_graph()

        try:
            # Run state check and memory search concurrently to save 200-500ms
            state, relevant_memory = await asyncio.gather(
                graph.aget_state(config),
                memory_service.search(user_id, get_guardrails().redact_for_storage(messages[-1].content)),
            )

            await get_cost_guard().check(user_id or "anonymous")
            question = await self._cache_question(state, messages)
            if question:
                config["metadata"]["neutral_prompt"] = True  # this answer may be cached for everyone
                cached = await self._answer_from_cache(graph, config, question, allow_stale=False)
                if cached is not None:
                    yield cached
                    return

            if self._awaiting_human(state):
                logger.info("resuming_interrupted_graph_stream", session_id=session_id, next_nodes=state.next)
                graph_input = Command(resume=messages[-1].content)
            else:
                relevant_memory = relevant_memory or NO_MEMORY
                graph_input = {"messages": dump_messages(messages), "long_term_memory": relevant_memory}

            redactor = get_guardrails().stream_redactor()
            emitted = False
            try:
                async for token, _ in graph.astream(
                    graph_input,
                    config,
                    stream_mode="messages",
                ):
                    if not isinstance(token, (AIMessage, AIMessageChunk)):
                        continue

                    text = redactor.feed(extract_text_content(token.content))
                    if text:
                        emitted = True
                        yield text
            except GraphInterrupt:
                raise
            except Exception:
                # stale-if-error, but only before any text went out: never splice two answers.
                stale = (
                    await self._answer_from_cache(graph, config, question, allow_stale=True)
                    if question and not emitted
                    else None
                )
                if stale is None:
                    raise
                yield stale
                return
            tail = redactor.flush()
            if tail:
                yield tail

            # After streaming completes, check for interrupt or update memory
            state = await graph.aget_state(config)
            if state.next:
                interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
                logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
                yield str(interrupt_value)
            elif state.values and "messages" in state.values:
                create_background_task(memory_service.add(user_id, state.values["messages"]))
                create_background_task(get_cost_guard().record(user_id or "anonymous", usage))
                if question:
                    create_background_task(self._store_in_cache(question, state.values["messages"]))
        except GraphInterrupt:
            state = await graph.aget_state(config)
            interrupt_value = state.tasks[0].interrupts[0].value if state.tasks else "Waiting for input."
            logger.info("graph_interrupted_stream", session_id=session_id, interrupt_value=str(interrupt_value))
            yield str(interrupt_value)
        except Exception as stream_error:
            logger.exception("stream_processing_failed", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        graph = await self._get_graph()

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}
        state: StateSnapshot = await graph.aget_state(config=config)
        return self.__process_messages(state.values["messages"]) if state.values else []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        openai_style_messages = convert_to_openai_messages(messages)
        # keep just assistant and user messages
        return [
            Message(role=message["role"], content=str(message["content"]))
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID.

        Args:
            session_id: The ID of the session to clear history for.

        Raises:
            Exception: If there's an error clearing the chat history.
        """
        try:
            # Make sure the pool is initialized in the current event loop
            conn_pool = await self._get_connection_pool()
            if conn_pool is None:
                raise RuntimeError("connection pool unavailable; cannot clear chat history")

            # Batch all DELETEs in a single pipeline round-trip
            async with conn_pool.connection() as conn:
                async with conn.pipeline():
                    for table in settings.CHECKPOINT_TABLES:
                        await conn.execute(
                            sql.SQL("DELETE FROM {} WHERE thread_id = %s").format(sql.Identifier(table)),
                            (session_id,),
                        )
                logger.info(
                    "checkpoint_tables_cleared_for_session",
                    tables=settings.CHECKPOINT_TABLES,
                    session_id=session_id,
                )

        except Exception as e:
            logger.error(
                "clear_chat_history_operation_failed",
                session_id=session_id,
                error=str(e),
            )
            raise
