"""Deterministic detectors: prompt injection, PII and secrets.

Ported from the author's D11 Guardrails lab solution (``input_guardrails.py``,
``output_guardrails.py``, ``security_boundary.py``) and extended with the
Vietnamese and English attack styles in the D24 Eval-Guard adversarial set.
Regex is one signal, not the whole boundary: semantic cases (off-topic, asking
for someone else's data) belong to the optional LLM rail.
"""

import re
import unicodedata
from typing import (
    Callable,
    Literal,
    Optional,
)

from pydantic import BaseModel

FindingKind = Literal[
    "prompt_injection",
    "email",
    "phone",
    "national_id",
    "credit_card",
    "api_key",
    "password",
    "private_key",
    "denylisted",
]

PII_KINDS: frozenset[FindingKind] = frozenset({"email", "phone", "national_id", "credit_card"})
SECRET_KINDS: frozenset[FindingKind] = frozenset({"api_key", "password", "private_key", "denylisted"})

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏⁠﻿­"))


class Finding(BaseModel):
    """A detected span. ``start``/``end`` index the text the detector was given."""

    kind: FindingKind
    start: int
    end: int
    value: str


def normalize_for_security(text: str) -> str:
    """NFKC-fold look-alike characters and drop zero-width ones before matching.

    ``Ig​nore`` and full-width ``Ｉｇｎｏｒｅ`` both become ``Ignore``.
    """
    return unicodedata.normalize("NFKC", text or "").translate(_ZERO_WIDTH)


# --------------------------------------------------------------------------- injection

_W = r"(?:\S+\s+)"  # one arbitrary word, for "ignore <up to N words> instructions"
_INJECTION_PATTERNS = [
    # English instruction overrides
    rf"\b(?:ignore|disregard|forget|override|bypass)\s+(?:all|any|every|your|previous|prior|above|earlier|preceding)\b\s+{_W}{{0,3}}?(?:instructions?|prompts?|rules|polic(?:y|ies)|guidelines|directives)\b",
    r"\b(?:ignore|disregard|forget)\s+(?:the\s+|your\s+)?(?:system|developer)\s+(?:prompt|message|instructions?)\b",
    r"\byou\s+are\s+now\s+(?:a|an|the|in|dan|my|no\s+longer|unrestricted|free|jailbroken)\b",
    rf"\b(?:pretend|act)\s+(?:that\s+)?(?:you\s+are|to\s+be|as)\s+{_W}{{0,3}}?(?:dan|unrestricted|unfiltered|jailbroken|evil)\b",
    r"\bdo\s+anything\s+now\b|\bdan\s+mode\b|\bjailbreak(?:ed|ing)?\b",
    r"\bunrestricted\s+(?:ai|assistant|mode|model|chatbot)\b",
    rf"\b(?:reveal|show|print|output|repeat|dump|leak|display)\s+(?:me\s+)?{_W}{{0,3}}?(?:system\s+(?:prompt|instructions?|message)|(?:your|the)\s+(?:instructions|prompt|rules)|hidden\s+prompt|initial\s+prompt)",
    r"\bsystem\s+override\b|\[\s*(?:admin|system|developer)\s+(?:command|override|mode)\b|\b(?:developer|god)\s+mode\b",
    r"\bdump\s+all\b",
    # Vietnamese instruction overrides
    rf"\bbỏ\s+qua\s+(?:tất\s+cả|mọi|toàn\s+bộ|hết)\s+{_W}{{0,2}}?(?:hướng\s+dẫn|chỉ\s+dẫn|chỉ\s+thị|quy\s+tắc|quy\s+định|lệnh|policy|chính\s+sách)",
    r"\bbỏ\s+qua\s+(?:các\s+|những\s+)?(?:hướng\s+dẫn|chỉ\s+dẫn|chỉ\s+thị|quy\s+tắc)\s+(?:trước|ở\s+trên|ban\s+đầu|hệ\s+thống)",
    r"\b(?:đóng|nhập)\s+vai\b(?!\s+trò)",  # "đóng vai trò" means "plays the role of" and is benign
    r"\b(?:bạn|mày)\s+(?:là|đang\s+là|giờ\s+là|bây\s+giờ\s+là)\s+(?:một\s+)?(?:ai|trợ\s+lý|chatbot)\s+(?:không|ko)\s+(?:có\s+)?(?:giới\s+hạn|bị\s+giới\s+hạn|kiểm\s+duyệt|quy\s+tắc)",
    r"\b(?:in|hiển\s+thị|tiết\s+lộ|cho\s+(?:tôi\s+)?xem|liệt\s+kê)\s+(?:ra\s+)?(?:toàn\s+bộ\s+|tất\s+cả\s+)?(?:system\s+(?:prompt|instructions?)|chỉ\s+dẫn\s+hệ\s+thống|lời\s+nhắc\s+hệ\s+thống|prompt\s+hệ\s+thống)",
    rf"\btiết\s+lộ\s+{_W}{{0,2}}?(?:mật\s*khẩu|api|khóa|key|thông\s+tin\s+nội\s+bộ|dữ\s+liệu\s+mật|bảng\s+lương|lương)",
]
_INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in _INJECTION_PATTERNS), re.IGNORECASE)


def find_injections(text: str) -> list[Finding]:
    """Return instruction-override spans in ``text`` (already normalized)."""
    return [
        Finding(kind="prompt_injection", start=m.start(), end=m.end(), value=m.group(0))
        for m in _INJECTION_RE.finditer(text)
    ]


# --------------------------------------------------------------------------- PII and secrets


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _cccd_ok(digits: str) -> bool:
    # Citizen ID: 3-digit province code (001-096) first.
    return 1 <= int(digits[:3]) <= 96


_ID_CONTEXT = re.compile(r"(?:cmnd|cccd|chứng\s+minh|căn\s+cước|id\s*card|national\s+id|passport|hộ\s+chiếu)", re.I)


class _Rule:
    def __init__(
        self,
        kind: FindingKind,
        pattern: str,
        flags: int = 0,
        group: int | str = 0,
        validate: Optional[Callable[[re.Match[str], str], bool]] = None,
    ):
        self.kind = kind
        self.regex = re.compile(pattern, flags)
        self.group = group
        self.validate = validate


def _digits(m: re.Match[str]) -> str:
    return re.sub(r"\D", "", m.group(0))


_SENSITIVE_RULES = [
    _Rule(
        "private_key",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
    ),
    _Rule(
        "api_key",
        r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{16,}"  # OpenAI-style
        r"|\bAKIA[0-9A-Z]{16}\b"  # AWS access key id
        r"|\bgh[pousr]_[A-Za-z0-9]{36,}\b"  # GitHub token
        r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"  # Slack token
        r"|\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"  # JWT
        r"|\bBearer\s+[A-Za-z0-9._~+/-]{20,}=*",
    ),
    _Rule(
        "password",
        # Only the value is redacted. It must contain a digit or symbol so that
        # "mật khẩu mặc định là gì?" or "password is required" are left alone.
        r"\b(?:password|passwd|pwd|mật\s*khẩu)\b(?:\s+[^\s:=]+){0,4}?\s*(?:is|was|=|:|là)\s*"
        r"(?P<value>(?=\S*[\d!@#$%^&*_+=])[^\s,;]{4,})",
        re.IGNORECASE,
        group="value",
    ),
    _Rule("email", r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
    _Rule(
        "credit_card",
        r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)",
        validate=lambda m, _t: 13 <= len(_digits(m)) <= 19 and _luhn_ok(_digits(m)),
    ),
    _Rule(
        "national_id",
        r"(?<!\d)\d{12}(?!\d)",
        validate=lambda m, _t: _cccd_ok(m.group(0)),
    ),
    _Rule(
        "national_id",
        # 9-digit legacy ID only next to an ID keyword: bare 9-digit numbers are often amounts.
        r"(?<!\d)\d{9}(?!\d)",
        validate=lambda m, t: bool(_ID_CONTEXT.search(t[max(0, m.start() - 30) : m.start()])),
    ),
    _Rule(
        "phone",
        r"(?<![\w+])(?:\+84[\s.-]?|0)(?:[35789](?:[\s.-]?\d){8}|2(?:[\s.-]?\d){9})(?!\d)",
    ),
]


def find_sensitive(
    text: str,
    kinds: frozenset[FindingKind] = PII_KINDS | SECRET_KINDS,
    denylist: tuple[str, ...] = (),
) -> list[Finding]:
    """Return non-overlapping PII/secret spans, earliest and longest first.

    ``denylist`` holds application secrets (e.g. an internal hostname) that must
    never appear; they also match with punctuation or spacing inserted.
    """
    found: list[Finding] = []
    for rule in _SENSITIVE_RULES:
        if rule.kind not in kinds:
            continue
        for m in rule.regex.finditer(text):
            if rule.validate and not rule.validate(m, text):
                continue
            start, end = m.span(rule.group)
            found.append(Finding(kind=rule.kind, start=start, end=end, value=m.group(rule.group)))

    if "denylisted" in kinds:
        for term in denylist:
            letters = [re.escape(ch) for ch in term if ch.isalnum()]
            if not letters:
                continue
            spaced = r"[\W_]{0,3}".join(letters)  # admin123 also matches "a-d-m-i-n 1 2 3"
            for m in re.finditer(spaced, text, re.IGNORECASE):
                found.append(Finding(kind="denylisted", start=m.start(), end=m.end(), value=m.group(0)))

    found.sort(key=lambda f: (f.start, -(f.end - f.start)))
    merged: list[Finding] = []
    for f in found:
        if merged and f.start < merged[-1].end:
            continue  # overlaps an earlier, longer finding
        merged.append(f)
    return merged


def redact(text: str, findings: list[Finding], placeholder: str = "[REDACTED:{kind}]") -> str:
    """Replace each finding span with a placeholder; spans must index ``text``."""
    out, last = [], 0
    for f in sorted(findings, key=lambda f: f.start):
        if f.start < last:
            continue
        out.append(text[last : f.start])
        out.append(placeholder.format(kind=f.kind))
        last = f.end
    out.append(text[last:])
    return "".join(out)
