"""Deterministic citation checks for answers and reports.

The D16 lab showed the most convincing RAG failure: true sentences cited to the
wrong source. The D23 lab made the research agent run a citation validator until
it passed. These checks cover the mechanical part: every ``[n]`` must point at a
source that exists, and a report's source list must be fully cited. Whether the
cited source really supports the sentence still needs a judge (see the eval layer).
"""

import re

from pydantic import BaseModel

_LABEL_GROUP = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_SOURCES_HEADING = re.compile(r"^#{1,6}\s*(?:sources|references|nguồn|tài liệu tham khảo)\s*$", re.I | re.M)
_SOURCE_LINE = re.compile(r"^\s*(?:[-*]\s*)?\[?(\d+)[\].]\s+\S", re.M)


class CitationCheck(BaseModel):
    """Result of a citation check."""

    cited: list[int]
    out_of_range: list[int]
    uncited_sources: list[int]
    source_count: int

    @property
    def ok(self) -> bool:
        """True when every label resolves and every listed source is cited."""
        return not self.out_of_range and not self.uncited_sources

    def problems(self) -> list[str]:
        """Human-readable issues, for an agent to fix."""
        issues = []
        if self.out_of_range:
            issues.append(
                f"labels {self.out_of_range} have no matching source (sources are numbered 1..{self.source_count})"
            )
        if self.uncited_sources:
            issues.append(f"sources {self.uncited_sources} are listed but never cited; cite or remove them")
        return issues


def cited_labels(text: str) -> list[int]:
    """All citation numbers in order of first appearance; ``[1][2]`` and ``[1, 3]`` both count."""
    seen: list[int] = []
    for group in _LABEL_GROUP.findall(text):
        for part in re.split(r"[,;]", group):
            n = int(part)
            if n not in seen:
                seen.append(n)
    return seen


def check_answer(answer: str, source_count: int) -> CitationCheck:
    """Check an answer against ``source_count`` numbered sources it was given."""
    cited = cited_labels(answer)
    return CitationCheck(
        cited=cited,
        out_of_range=[n for n in cited if not 1 <= n <= source_count],
        uncited_sources=[],  # an answer need not use every retrieved chunk
        source_count=source_count,
    )


def check_report(markdown: str) -> CitationCheck:
    """Check a report whose last section lists its sources as ``[n] ...`` or ``n. ...`` lines."""
    headings = list(_SOURCES_HEADING.finditer(markdown))
    if not headings:
        cited = cited_labels(markdown)
        return CitationCheck(cited=cited, out_of_range=cited, uncited_sources=[], source_count=0)
    split = headings[-1].start()
    body, sources = markdown[:split], markdown[split:]
    listed = sorted({int(n) for n in _SOURCE_LINE.findall(sources)})
    cited = cited_labels(body)
    return CitationCheck(
        cited=cited,
        out_of_range=[n for n in cited if n not in listed],
        uncited_sources=[n for n in listed if n not in cited],
        source_count=len(listed),
    )
