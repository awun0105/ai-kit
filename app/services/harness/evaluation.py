"""Eval for in-app tasks (spec section 8): did the agent reach the expected app state with the expected tools?

A case gives a goal, the tools the agent should call (matched like the eval
layer's trajectories) and a part of ``app.context`` the final state must
contain. Run each case several times for a ``pass_rate``. Imports the eval
layer's trajectory matcher, so this module is not loaded by the harness package.
"""

from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.services.evaluation.trajectory import (
    ToolMatch,
    match_trajectory,
)
from app.services.harness.harness import Harness
from app.services.harness.permissions import PermissionPolicy


class InAppCase(BaseModel):
    """One task to evaluate."""

    id: str
    goal: str
    scope: Any = None
    expected_tools: list[str] = Field(default_factory=list)
    tool_match: ToolMatch = "in_order"
    expect: dict[str, Any] = Field(default_factory=dict)  # must be contained in app.context after the task


class CaseRun(BaseModel):
    """What one run of a case did."""

    state: str
    tools_called: list[str] = Field(default_factory=list)
    context: Any = None
    error: Optional[str] = None


class CaseResult(BaseModel):
    """All runs of one case."""

    id: str
    runs: list[CaseRun]
    passed: list[bool]

    @property
    def pass_rate(self) -> float:
        """Share of runs that passed."""
        return sum(self.passed) / len(self.passed) if self.passed else 0.0


class InAppReport(BaseModel):
    """Results of a set of cases."""

    items: list[CaseResult]

    @property
    def pass_rate(self) -> float:
        """Share of all runs that passed."""
        runs = [ok for item in self.items for ok in item.passed]
        return sum(runs) / len(runs) if runs else 0.0


def contains(actual: Any, expected: Any) -> bool:
    """Whether ``expected`` is part of ``actual``: dict keys recursively, other values equal."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(k in actual and contains(actual[k], v) for k, v in expected.items())
    return actual == expected


def run_passed(case: InAppCase, run: CaseRun) -> bool:
    """A run passes when the task was accepted, the trajectory matches and the state has what was expected."""
    return (
        run.state == "accepted"
        and match_trajectory(run.tools_called, case.expected_tools, case.tool_match)
        and contains(run.context, case.expect)
    )


def harness_runner(
    harness: Harness, policy: PermissionPolicy = PermissionPolicy.allow_in_draft
) -> Callable[[InAppCase], Awaitable[CaseRun]]:
    """Run cases through ``harness`` with automatic review (no user in the loop)."""

    async def run(case: InAppCase) -> CaseRun:
        task_id = await harness.start_task(case.goal, case.scope, policy=policy, review="auto")
        state = await harness.wait(task_id)
        context = await harness.adapter.context()
        return CaseRun(
            state=state.value, tools_called=list(harness.task(task_id).tools_called), context=context.model_dump()
        )

    return run


async def run_in_app_eval(
    cases: list[InAppCase], run_case: Callable[[InAppCase], Awaitable[CaseRun]], runs: int = 1
) -> InAppReport:
    """Run every case ``runs`` times (one after another: the harness has one writer) and score each run."""
    items = []
    for case in cases:
        results = []
        for _ in range(runs):
            try:
                results.append(await run_case(case))
            except Exception as error:  # noqa: BLE001  (a failed run is a result, not a crash)
                results.append(CaseRun(state="error", error=type(error).__name__))
        items.append(CaseResult(id=case.id, runs=results, passed=[run_passed(case, r) for r in results]))
    return InAppReport(items=items)
