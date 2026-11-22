"""Deterministic pre-flight prompt-injection blocker and PII redaction engine."""
from __future__ import annotations

from dataclasses import dataclass
import re

from routeguard.security import normalize_adversarial_text

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("instruction_override", re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)\b", re.I)),
    ("system_prompt_exfil", re.compile(r"\b(?:reveal|print|dump|repeat|output)\s+(?:your\s+)?(?:system\s+prompt|hidden\s+instructions|developer\s+message)\b", re.I)),
    ("role_hijack", re.compile(r"\byou\s+are\s+now\s+(?:DAN|unfiltered|in\s+developer\s+mode|bypassing\s+safety)\b", re.I)),
    ("delimiter_smuggling", re.compile(r"(?:<\s*/?\s*system\s*>|\[SYSTEM_OVERRIDE\]|###\s*NEW\s+SYSTEM\s+RULE)", re.I)),
    ("credential_exfil", re.compile(r"\b(?:print|return|leak)\s+(?:the\s+)?(?:api[_\s]?key|env(?:ironment)?\s+variables|secret\s+token)\b", re.I)),
)

_PII_PATTERNS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("IN_PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), "[REDACTED_PAN]"),
    ("IN_AADHAAR", re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[REDACTED_AADHAAR]"),
    ("US_SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED_CARD]"),
)


@dataclass
class GuardrailOutcome:
    blocked: bool
    block_reasons: list[str]
    sanitized_prompt: str
    pii_redacted_types: list[str]


def inspect_prompt(prompt: str) -> GuardrailOutcome:
    """Normalize NFKC/zero-width/Base64 smuggling, inspect injections, and mask PII."""
    text = prompt or ""
    normalized, decoded_blobs = normalize_adversarial_text(text)
    inspect_targets = [normalized] + decoded_blobs
    reasons: list[str] = []
    for candidate in inspect_targets:
        for name, rx in _INJECTION_PATTERNS:
            if rx.search(candidate) and name not in reasons:
                reasons.append(name)

    sanitized = text
    redacted_types: list[str] = []
    for label, rx, repl in _PII_PATTERNS:
        if rx.search(sanitized):
            sanitized = rx.sub(repl, sanitized)
            redacted_types.append(label)

    return GuardrailOutcome(
        blocked=bool(reasons),
        block_reasons=reasons,
        sanitized_prompt=sanitized,
        pii_redacted_types=redacted_types,
    )
