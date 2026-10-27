"""The task state machine (spec section 5.4), reduced from fframes Studio's edit transaction states."""

from enum import StrEnum
from typing import (
    Callable,
    Optional,
)


class TaskState(StrEnum):
    """Where a task is, from the request to its end."""

    context_ready = "context_ready"
    running = "running"
    waiting_permission = "waiting_permission"
    validating = "validating"
    repair_needed = "repair_needed"
    candidate_ready = "candidate_ready"
    accepted = "accepted"
    rejected = "rejected"
    conflict = "conflict"
    failed = "failed"
    cancelled = "cancelled"


S = TaskState
TERMINAL = frozenset({S.accepted, S.rejected, S.conflict, S.failed, S.cancelled})
ACTIVE = frozenset(set(S) - TERMINAL)

_STOP = frozenset({S.failed, S.cancelled})
TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    S.context_ready: frozenset({S.running}) | _STOP,
    S.running: frozenset({S.waiting_permission, S.validating}) | _STOP,
    S.waiting_permission: frozenset({S.running}) | _STOP,
    S.validating: frozenset({S.candidate_ready, S.repair_needed}) | _STOP,
    S.repair_needed: frozenset({S.running}) | _STOP,
    S.candidate_ready: frozenset({S.accepted, S.rejected, S.conflict}) | _STOP,
    **{state: frozenset() for state in TERMINAL},
}


class InvalidTransition(Exception):
    """A move the state machine does not allow."""


class TaskStateMachine:
    """Holds a task's state and refuses moves the spec does not allow."""

    def __init__(self, on_change: Optional[Callable[[TaskState, TaskState], None]] = None):
        """Start in ``context_ready``; ``on_change(old, new)`` runs after every move."""
        self.state = TaskState.context_ready
        self._on_change = on_change

    @property
    def active(self) -> bool:
        """Whether the task has not ended yet."""
        return self.state in ACTIVE

    def to(self, new: TaskState) -> None:
        """Move to ``new`` or raise ``InvalidTransition``."""
        if new not in TRANSITIONS[self.state]:
            raise InvalidTransition(f"{self.state} -> {new}")
        old, self.state = self.state, new
        if self._on_change is not None:
            self._on_change(old, new)
