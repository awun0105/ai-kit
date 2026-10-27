"""Run the deep-research team on a topic from the command line.

Usage (needs `uv sync --extra deep` and a model API key):
    APP_ENV=development uv run python -m scripts.research "survey of world models" > report.md
"""

import argparse
import asyncio
import sys

from app.core.kit import get_guardrails
from app.core.langgraph.tools import tools as builtin_tools
from app.services.agents import ResilientChatModel
from app.services.guardrails import guard_tools
from app.services.agents.research import (
    build_research_agent,
    run_research,
)
from app.services.llm import llm_service


async def main() -> None:
    """Research the topic with the built-in tools and print the report."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topic")
    parser.add_argument("--max-model-calls", type=int, default=40)
    parser.add_argument("--max-tool-calls", type=int, default=80)
    args = parser.parse_args()

    sources = guard_tools([t for t in builtin_tools if t.name != "ask_human"], get_guardrails())
    agent = build_research_agent(
        ResilientChatModel(service=llm_service), sources, args.max_model_calls, args.max_tool_calls
    )
    result = await run_research(agent, args.topic)
    print(result.report)
    if not result.citations_ok:
        print("\nwarning: the citation check did not pass", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
