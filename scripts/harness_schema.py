"""Write the harness protocol's JSON schema to docs/harness/protocol.schema.json.

Usage:
    uv run python -m scripts.harness_schema
"""

import json
from pathlib import Path

from rich.console import Console

from app.services.harness.protocol import protocol_schema

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "docs" / "harness" / "protocol.schema.json"


def main() -> None:
    """Regenerate the schema file."""
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCHEMA_PATH.write_text(json.dumps(protocol_schema(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    Console().print(f"[green]wrote[/green] {SCHEMA_PATH}")


if __name__ == "__main__":
    main()
