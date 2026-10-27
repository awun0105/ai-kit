"""The toolbox components import on their own: no settings, database, model registry or env vars.

That is what lets a project copy one of them (with its tests and doc) instead of the whole app.
The app's wiring lives in ``app/core/kit.py``.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TOOLBOX = [
    "app.services.rag",
    "app.services.guardrails",
    "app.services.reliability",
    "app.services.evaluation",
    "app.services.mcp",
    "app.services.agents",
    "app.services.cost_guard",
    "app.services.compaction",
    "app.services.harness",
]
APP_MODULES = ["app.core.config", "app.core.logging", "app.services.llm", "app.services.database", "app.core.kit"]


@pytest.mark.parametrize(
    "module", TOOLBOX + ["app.services.a2a.server", "app.services.a2a.client", "app.services.acp"]
)
def test_toolbox_module_imports_without_the_app(module):
    if module.startswith("app.services.a2a"):
        pytest.importorskip("a2a")
    if module.startswith("app.services.acp"):
        pytest.importorskip("acp")
    probe = f"import sys, {module}; print(','.join(m for m in {APP_MODULES!r} if m in sys.modules))"
    # No APP_ENV, no JWT secret, no API key, and no .env file to fall back on: run from an empty directory.
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT / "tests" / "fixtures", env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"{module} pulled in {result.stdout.strip()}"


def test_harness_core_loads_no_langchain():
    probe = "import sys, app.services.harness; print(','.join(m for m in sys.modules if m.split('.')[0] in ('langchain', 'langchain_core', 'langgraph')))"
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT / "tests" / "fixtures", env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""
