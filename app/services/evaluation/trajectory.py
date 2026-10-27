"""Did the agent call the tools it should have? Match the tool calls of a run against the expected ones.

Ported from Google ADK's ``TrajectoryEvaluator`` (google/adk-python, commit 2084efb,
``src/google/adk/evaluation/trajectory_evaluator.py``): the same three match modes,
on tool names only. Expected arguments (search queries) vary from run to run, so
matching them would fail runs that did the right thing.
"""

from collections import Counter
from typing import Literal

ToolMatch = Literal["exact", "in_order", "any_order"]


def match_trajectory(actual: list[str], expected: list[str], mode: ToolMatch = "in_order") -> bool:
    """Whether the tool names called in a run satisfy the expected ones.

    - ``exact``: the same calls in the same order, nothing extra.
    - ``in_order``: every expected call, in that order; other calls may come in between.
    - ``any_order``: every expected call (as many times as listed), in any order, extras allowed.
    """
    if mode == "exact":
        return actual == expected
    if mode == "in_order":
        remaining = iter(actual)
        return all(name in remaining for name in expected)  # ``in`` consumes the iterator up to the match
    needed, called = Counter(expected), Counter(actual)
    return all(called[name] >= count for name, count in needed.items())
