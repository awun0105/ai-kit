"""One JSONL file of events per task, with every string redacted before it is written."""

import json
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
)


def redact_values(value: Any, redact: Callable[[str], str]) -> Any:
    """``value`` with ``redact`` applied to every string inside it."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_values(item, redact) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_values(item, redact) for item in value]
    return value


class Transcript:
    """Appends events to ``path``."""

    def __init__(self, path: Path, redact: Callable[[str], str] = lambda text: text):
        """``redact`` runs on every string (wire in ``guardrails.redact`` in the app)."""
        self.path = path
        self._redact = redact
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, data: dict) -> None:
        """Append one event."""
        line = {"ts": round(time.time(), 3), "event": event, "data": redact_values(data, self._redact)}
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
