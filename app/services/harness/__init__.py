"""In-app agent harness: an agent works inside an app through drafts, permissions, validation and revisions.

The language-neutral contract is ``docs/harness.md`` and
``docs/harness/protocol.schema.json``; this package is the Python reference.
It imports no LangChain: the ACP driver (``acp_driver``, ``acp`` extra) and the
kit's agent driver (``app/harness_kit_driver.py``) plug in from outside.
"""

from app.services.harness.adapter import (
    AdapterClient,
    AppAdapter,
    adapter_handlers,
)
from app.services.harness.bridge import (
    BridgeInfo,
    ToolBridge,
    mcp_server_spec,
)
from app.services.harness.broker import (
    TaskRuntime,
    ToolBroker,
    ToolResult,
)
from app.services.harness.capability import (
    CapabilityError,
    CapabilityStore,
)
from app.services.harness.driver import (
    AgentDriver,
    DriverError,
    DriverFactory,
    TurnContext,
)
from app.services.harness.harness import Harness
from app.services.harness.permissions import (
    PermissionDesk,
    PermissionPolicy,
    decide,
)
from app.services.harness.protocol import (
    PROTOCOL_VERSION,
    AppContext,
    AppDescription,
    CommitResult,
    DraftDiff,
    Effect,
    Operation,
    ValidationIssue,
    ValidationReport,
)
from app.services.harness.rpc import (
    APP_ERROR,
    JsonRpcPeer,
    RpcError,
    spawn_peer,
)
from app.services.harness.sidecar import (
    harness_handlers,
    run_sidecar,
)
from app.services.harness.state import (
    TaskState,
    TaskStateMachine,
)
from app.services.harness.transcript import Transcript

__all__ = [
    "APP_ERROR",
    "PROTOCOL_VERSION",
    "AdapterClient",
    "AgentDriver",
    "AppAdapter",
    "AppContext",
    "AppDescription",
    "BridgeInfo",
    "CapabilityError",
    "CapabilityStore",
    "CommitResult",
    "DraftDiff",
    "DriverError",
    "DriverFactory",
    "Effect",
    "Harness",
    "JsonRpcPeer",
    "Operation",
    "PermissionDesk",
    "PermissionPolicy",
    "RpcError",
    "TaskRuntime",
    "TaskState",
    "TaskStateMachine",
    "ToolBridge",
    "ToolBroker",
    "ToolResult",
    "Transcript",
    "TurnContext",
    "ValidationIssue",
    "ValidationReport",
    "adapter_handlers",
    "decide",
    "harness_handlers",
    "mcp_server_spec",
    "run_sidecar",
    "spawn_peer",
]
