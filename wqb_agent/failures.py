"""Failure taxonomy.

Distinguishes research-level failures (the hypothesis/direction/expression is
wrong) from system-level failures (network, auth, rate-limit, API 5xx,
timeouts). Only research-level failures are allowed to enter research memory
(lessons / avoid / garbage); system failures must not pollute research
experience.
"""

import re
from functools import lru_cache


class FailureKind:
    RESEARCH = "RESEARCH"
    SYNTAX = "SYNTAX"
    DATA = "DATA"
    INFRA = "INFRA"
    AUTH = "AUTH"
    RATE_LIMIT = "RATE_LIMIT"
    TIMEOUT = "TIMEOUT"


# Kinds that are caused by the hypothesis / expression / field choice and may
# be learned from. Everything else is environmental and must stay out of the
# research memory.
RESEARCH_RELEVANT = {FailureKind.RESEARCH, FailureKind.SYNTAX, FailureKind.DATA}


def is_research_relevant(kind):
    return kind in RESEARCH_RELEVANT


_TIMEOUT_RE = re.compile(r"timeout|timed out", re.IGNORECASE)
_AUTH_RE = re.compile(r"auth|credential|401|login", re.IGNORECASE)
_RATE_RE = re.compile(r"429|rate.?limit|too many requests|retry-after", re.IGNORECASE)
_SYNTAX_RE = re.compile(r"rejected|422|400|syntax|parse|invalid expression",
                       re.IGNORECASE)
_NOT_FOUND_RE = re.compile(r"404|not found|missing", re.IGNORECASE)
_INFRA_RE = re.compile(r"5\d\d|connection|timeout|broken|unavailable|refused|"
                       r"network|temporary", re.IGNORECASE)


@lru_cache(maxsize=512)
def classify_error(error_text, status_code=None):
    """Map an error (message + optional status code) to a FailureKind.

    Error messages repeat heavily within a run (the same syntax / timeout text
    for every affected expression), so the regex scan is cached by text.
    """
    text = error_text or ""
    if status_code == 401:
        return FailureKind.AUTH
    if status_code == 429:
        return FailureKind.RATE_LIMIT
    if status_code in (400, 422):
        return FailureKind.SYNTAX
    if status_code in (403, 404):
        return FailureKind.DATA
    if status_code is not None and status_code >= 500:
        return FailureKind.INFRA

    if _TIMEOUT_RE.search(text):
        return FailureKind.TIMEOUT
    if _AUTH_RE.search(text):
        return FailureKind.AUTH
    if _RATE_RE.search(text):
        return FailureKind.RATE_LIMIT
    if _SYNTAX_RE.search(text):
        return FailureKind.SYNTAX
    if _NOT_FOUND_RE.search(text):
        return FailureKind.DATA
    if _INFRA_RE.search(text):
        return FailureKind.INFRA
    return FailureKind.RESEARCH
