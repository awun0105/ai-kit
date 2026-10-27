"""The data that crosses the app/harness connection, as pydantic models with a published JSON schema.

``docs/harness/protocol.schema.json`` is generated from these models
(``scripts/harness_schema.py``); a port reads the schema, not this file.
"""

from typing import (
    Any,
    Literal,
    Optional,
)

from pydantic import (
    BaseModel,
    Field,
)

PROTOCOL_VERSION = "0.1"

OpLevel = Literal["read", "write", "dangerous"]
Support = Literal["undo", "materialize", "effects"]


class Operation(BaseModel):
    """Something the app can do: a tool for the agent."""

    name: str
    description: str
    params: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    level: OpLevel


class AppDescription(BaseModel):
    """What ``app.describe`` returns."""

    protocol_version: str
    operations: list[Operation]
    supports: list[Support] = Field(default_factory=list)


class AppContext(BaseModel):
    """What ``app.context`` returns: the current revision and a size-bounded view of the state."""

    revision: str
    state: Any = None
    selection: Any = None


class ValidationIssue(BaseModel):
    """One problem found in a draft."""

    code: str
    message: str
    path: Optional[str] = None


class ValidationReport(BaseModel):
    """What ``draft.validate`` and ``draft.sync`` return."""

    ok: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class Effect(BaseModel):
    """An approved ``dangerous`` operation, run with ``effect.run`` only after its revision is committed."""

    id: str
    op: str
    args: dict[str, Any] = Field(default_factory=dict)


class CommitResult(BaseModel):
    """What ``draft.commit`` returns: the new revision, or ``conflict`` when the base has moved."""

    revision: Optional[str] = None
    conflict: bool = False


class DraftDiff(BaseModel):
    """What ``draft.diff`` returns: a text summary plus any app-defined detail."""

    summary: str
    detail: Any = None


MODELS: list[type[BaseModel]] = [
    Operation,
    AppDescription,
    AppContext,
    ValidationIssue,
    ValidationReport,
    Effect,
    CommitResult,
    DraftDiff,
]


def protocol_schema() -> dict[str, Any]:
    """The JSON schema of every model above, keyed by name, plus the protocol version."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "In-app agent harness protocol",
        "protocol_version": PROTOCOL_VERSION,
        "definitions": {model.__name__: model.model_json_schema() for model in MODELS},
    }
