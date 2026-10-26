"""Golden datasets: questions with expected answers and the documents that support them."""

import json
from pathlib import Path
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
)


class GoldenItem(BaseModel):
    """One evaluation question."""

    id: str
    question: str
    expected_answer: str = ""
    expected_sources: list[str] = Field(default_factory=list, description="Document ids that answer the question")
    difficulty: str = "unspecified"
    should_refuse: bool = Field(
        default=False, description="Adversarial or out-of-scope: the right answer is to decline or correct the premise"
    )


def load_golden(path: str | Path) -> list[GoldenItem]:
    """Load a golden set from JSON or JSONL.

    Accepts the kit's own format (a list of ``GoldenItem`` objects, or JSONL) and the
    AI20K D14 format (``{"qa_pairs": [{"id", "question", "expected_answer",
    "contexts": [{"source_doc", "text"}], "difficulty", "attack_type"}]}``).
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [GoldenItem.model_validate_json(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if isinstance(data, list):
        return [GoldenItem.model_validate(item) for item in data]
    if isinstance(data, dict) and "qa_pairs" in data:
        return [_from_d14(item) for item in data["qa_pairs"]]
    raise ValueError(f"{path}: expected a list of items, JSONL, or an object with 'qa_pairs'")


def _from_d14(item: dict) -> GoldenItem:
    sources: list[str] = []
    for context in item.get("contexts") or []:
        doc = context.get("source_doc")
        if doc and doc not in sources:
            sources.append(doc)
    difficulty = item.get("difficulty") or "unspecified"
    return GoldenItem(
        id=str(item["id"]),
        question=item["question"],
        expected_answer=item.get("expected_answer", ""),
        expected_sources=sources,
        difficulty=difficulty,
        should_refuse=difficulty == "adversarial" or bool(item.get("attack_type")),
    )


def by_difficulty(items: list[GoldenItem], difficulty: Optional[str]) -> list[GoldenItem]:
    """Filter by difficulty; ``None`` keeps everything."""
    return [i for i in items if difficulty is None or i.difficulty == difficulty]
