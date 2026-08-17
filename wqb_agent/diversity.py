"""Alpha diversity checks: field/dataset overlap, expression structure similarity,
hypothesis similarity. Used to keep the submission pool non-redundant."""

import re

_FIELD_RE = re.compile(r"[a-z0-9_]+")


def expression_tokens(expr):
    """Tokenize an expression into operators, field names and numbers."""
    return [t for t in _FIELD_RE.findall(expr or "") if t]


def expression_similarity(a, b):
    """Jaccard similarity over expression tokens (operators/fields/windows)."""
    ta = set(expression_tokens(a))
    tb = set(expression_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def field_similarity(fields_a, fields_b):
    fa = set(fields_a or [])
    fb = set(fields_b or [])
    if not fa or not fb:
        return 0.0
    return len(fa & fb) / len(fa | fb)


def hypothesis_similarity(h1_tags, h2_tags):
    ta = set(h1_tags or [])
    tb = set(h2_tags or [])
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _record_fields(rec):
    if isinstance(rec, dict):
        return rec.get("fields_used", [])
    return getattr(rec, "fields_used", [])


def _record_expr(rec):
    if isinstance(rec, dict):
        return rec.get("expression", "")
    return getattr(rec, "expression", "")


def is_redundant(record, pool_records, expr_th=0.6, field_th=0.5):
    """True if record is too similar to an existing pool record."""
    for rec in pool_records:
        fs = field_similarity(_record_fields(record), _record_fields(rec))
        ts = expression_similarity(_record_expr(record), _record_expr(rec))
        if fs >= field_th and ts >= expr_th:
            return True, rec
    return False, None


def deduplicate(pool_records, expr_th=0.6, field_th=0.6):
    """Drop near-duplicates, keeping the best-scoring one of each group."""
    from .state import score_of

    kept = []
    dropped = []
    for rec in sorted(
        pool_records, key=lambda r: score_of(r.get("metrics")), reverse=True
    ):
        redundant, _ = is_redundant(rec, kept, expr_th=expr_th, field_th=field_th)
        if redundant:
            dropped.append(rec)
        else:
            kept.append(rec)
    return kept, dropped
