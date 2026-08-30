"""Alpha diversity: structural fingerprints and multi-dimensional redundancy.

The submission pool must not be an accident of token overlap. Two alphas are
highly redundant when they share a lineage root / hypothesis / dataset /
operator family even if their expression tokens differ; a different hypothesis
or dataset is evidence of novelty worth keeping at a lower score.

A minimal research fingerprint per alpha:

    hypothesis_id, dataset_family, fields, operator_family, lineage_root

(plus neutralization / horizon_bucket as cheap extras). All selection is
deterministic; no embeddings and no learned ranking.
"""

import re
from functools import lru_cache

from .state import score_of

_FIELD_RE = re.compile(r"[a-z0-9_]+")

_WORD_BOUNDARY_RE = re.compile(r"(?<![\w])")

_CALL_RE = re.compile(r"\b([a-z_]+)\(")

_TS_WINDOW_RE = re.compile(r"\bts_[a-z_]+\([^)]*,\s*(\d+)\)")

_TRAILING_DIGITS_RE = re.compile(r"\d+$")

_KNOWN_OPS = {
    "rank",
    "ts_rank",
    "ts_mean",
    "ts_std_dev",
    "ts_sum",
    "ts_delta",
    "ts_min",
    "ts_max",
    "ts_zscore",
    "ts_corr",
    "ts_decay_linear",
    "zscore",
    "scale",
    "group_neutralize",
    "signed_power",
    "abs",
}


@lru_cache(maxsize=1024)
def _field_boundary_pattern(field_id):
    """Compile the word-boundary match for one field id once; extract_fields
    reuses it across candidates instead of re-compiling per search."""
    return re.compile(
        _WORD_BOUNDARY_RE.pattern + re.escape(field_id) + r"(?![\w])"
    )


def extract_fields(expression, known_fields):
    """Return the subset of known_fields that actually appear in the expression.

    Matches longest ids first so that a field id which is a prefix of another
    (e.g. ``returns`` vs ``returns_5d``) does not cause a false positive.
    """
    expression = expression or ""
    found = []
    for fid in sorted(dict.fromkeys(known_fields or []), key=len, reverse=True):
        if not fid:
            continue
        if _field_boundary_pattern(fid).search(expression):
            found.append(fid)
    return found


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


# ---- record accessors (dict or object) ----

def _record_fields(rec):
    if isinstance(rec, dict):
        return rec.get("fields_used", [])
    return getattr(rec, "fields_used", [])


def _record_expr(rec):
    if isinstance(rec, dict):
        return rec.get("expression", "")
    return getattr(rec, "expression", "")


def _rec_datasets(rec):
    if isinstance(rec, dict):
        return rec.get("datasets") or []
    return getattr(rec, "datasets", []) or []


def _rec_hypothesis(rec):
    if isinstance(rec, dict):
        return rec.get("hypothesis_id")
    return getattr(rec, "hypothesis_id", None)


def _rec_lineage(rec):
    if isinstance(rec, dict):
        return rec.get("lineage") or []
    return getattr(rec, "lineage", []) or []


# ---- canonical dimensions ----

def dataset_family(dataset_id):
    """Canonical family of a dataset id: trailing digits stripped. ``pv1`` and
    ``pv13`` both belong to the ``pv`` family."""
    d = str(dataset_id or "")
    return _TRAILING_DIGITS_RE.sub("", d) or d


def operator_family(expression):
    """Canonical operator family: the sorted set of operator calls in the
    expression. ``rank(ts_mean(x, 20))`` and ``rank(ts_mean(y, 20))`` share
    the ``rank-ts_mean`` family even though their fields differ. The leading
    sign is intentionally ignored: ``-rank`` and ``rank`` are the same family."""
    calls = {m.group(1) for m in _CALL_RE.finditer(expression or "")}
    ops = sorted(c for c in calls if c in _KNOWN_OPS)
    return "-".join(ops) or "raw"


def horizon_bucket(expression):
    """Time-series windows appearing in the expression, sorted and deduped."""
    windows = sorted({int(m.group(1)) for m in _TS_WINDOW_RE.finditer(expression or "")})
    return tuple(windows)


def lineage_root(record):
    """The deepest ancestor of an alpha's lineage; an exploration alpha with an
    empty lineage is its own root."""
    lineage = _rec_lineage(record) or []
    if lineage:
        return lineage[-1]
    return _record_expr(record) or None


def fingerprint(record):
    """Minimal research identity for an alpha or candidate."""
    datasets = sorted({str(d) for d in _rec_datasets(record) if d})
    return {
        "hypothesis_id": _rec_hypothesis(record),
        "datasets": datasets,
        "dataset_family": sorted({dataset_family(d) for d in datasets}),
        "fields": sorted({f for f in _record_fields(record) if f}),
        "operator_family": operator_family(_record_expr(record)),
        "lineage_root": lineage_root(record),
        "neutralization": "group_neutralize(" in (_record_expr(record) or ""),
        "horizon_bucket": horizon_bucket(_record_expr(record)),
    }


# ---- redundancy ----

def _same_root(a, b):
    return bool(a["lineage_root"]) and a["lineage_root"] == b["lineage_root"]


def _different_hypotheses(a, b):
    ha, hb = a["hypothesis_id"], b["hypothesis_id"]
    return bool(ha) and bool(hb) and ha != hb


def _datasets_compatible(a, b):
    """Empty sets are neutral; non-empty dataset families with no overlap
    block redundancy because a different data source is novelty."""
    fa, fb = set(a["dataset_family"]), set(b["dataset_family"])
    if not fa or not fb:
        return True
    return bool(fa & fb)


def _same_direction(a, b):
    """Leading sign of the expression. ``rank(x)`` and ``-rank(x)`` are the
    same operator family but opposite research directions and must not be
    deduplicated against each other."""
    def sign(expr):
        return "-" if (expr or "").lstrip().startswith("-") else "+"
    return sign(a) == sign(b)


def _near_identical_expression(expr_a, expr_b, threshold=0.85):
    ta = set(expression_tokens(expr_a))
    tb = set(expression_tokens(expr_b))
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


def is_redundant(record, pool_records, expr_th=0.6, field_th=0.5,
                 _pool_fps=None):
    """Multi-dimensional redundancy against an existing set.

    Precedence:
    1. A different named hypothesis blocks dedup across hypotheses (a similar
       expression under another hypothesis is a new economic question), with
       the single exception of a near-identical expression.
    2. Same lineage root -> same research family -> redundant.
    3. Same hypothesis + same operator family + overlapping fields +
       compatible datasets -> redundant.
    4. Token-level near-duplicates (legacy behaviour).

    ``_pool_fps`` is an optional pre-computed fingerprint list parallel to
    ``pool_records``; batch callers (filter_candidates) pass it to avoid
    re-running the fingerprint regexes for every candidate.
    """
    fa = fingerprint(record)
    expr_a = _record_expr(record)
    if _pool_fps is None:
        _pool_fps = [fingerprint(rec) for rec in pool_records]
    for rec, fb in zip(pool_records, _pool_fps):
        if _different_hypotheses(fa, fb):
            if _near_identical_expression(expr_a, _record_expr(rec)):
                return True, rec
            continue
        if _same_root(fa, fb):
            return True, rec
        fs = field_similarity(fa["fields"], fb["fields"])
        if (
            _same_direction(expr_a, _record_expr(rec))
            and fa["operator_family"] == fb["operator_family"]
            and fs >= field_th
            and _datasets_compatible(fa, fb)
        ):
            return True, rec
        if fs >= field_th and (
            _same_direction(expr_a, _record_expr(rec))
            and expression_similarity(expr_a, _record_expr(rec)) >= expr_th
            and _datasets_compatible(fa, fb)
        ):
            return True, rec
    return False, None


def deduplicate(pool_records, expr_th=0.6, field_th=0.6):
    """Drop near-duplicates, keeping the best-scoring one of each group."""
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


# ---- pool selection ----

def _dim_count_over(counts, values, n, ratio):
    """True if any of ``values`` already holds more than ceil(n*ratio) slots."""
    cap = int(-(-n * ratio // 1)) if n > 0 else 0  # ceiling(n * ratio)
    for v in values:
        if cap > 0 and counts.get(v, 0) >= cap:
            return True
    return False


def select_diverse(
    pool_records,
    n,
    max_per_lineage=2,
    family_cap_ratio=0.6,
    hypothesis_cap_ratio=0.6,
):
    """Deterministic greedy selection: quality first, then structural novelty.

    Yields ``n`` diversified quality alphas instead of the ``n`` top scores.

    Slots are filled root-by-root in round-robin order (best root first) so a
    concentrated high-scoring family cannot crowd out a different lineage:
    first one slot per lineage root, then a second slot up to
    ``max_per_lineage``. Soft dataset-family / hypothesis caps bound how much
    of the pool any single source or question may hold. Any remaining slots
    are filled from the highest scores that respect the hard per-lineage cap,
    so high-quality alphas are never silently deleted just because a dimension
    is concentrated.
    """
    if len(pool_records) <= n:
        return list(pool_records)
    ordered = sorted(
        pool_records,
        key=lambda r: (score_of(r.get("metrics")), r.get("created_at", 0)),
        reverse=True,
    )

    by_root = {}
    for rec in ordered:
        by_root.setdefault(lineage_root(rec), []).append(rec)
    roots = sorted(
        by_root, key=lambda r: score_of(by_root[r][0].get("metrics")), reverse=True
    )

    selected = []
    root_counts = {}
    family_counts = {}
    hypothesis_counts = {}

    def over_soft_cap(f):
        return _dim_count_over(
            family_counts, f["dataset_family"], n, family_cap_ratio
        ) or _dim_count_over(
            hypothesis_counts, [f["hypothesis_id"]], n, hypothesis_cap_ratio
        )

    def take(rec, f):
        root = f["lineage_root"]
        selected.append(rec)
        if root:
            root_counts[root] = root_counts.get(root, 0) + 1
        for fam in f["dataset_family"]:
            family_counts[fam] = family_counts.get(fam, 0) + 1
        hypothesis_counts[f["hypothesis_id"]] = (
            hypothesis_counts.get(f["hypothesis_id"], 0) + 1
        )

    # Round-robin passes: one slot per root, then a second, up to the cap.
    for _ in range(max_per_lineage):
        progressed = False
        for root in roots:
            if len(selected) >= n:
                break
            idx = root_counts.get(root, 0)
            if idx >= len(by_root[root]):
                continue
            rec = by_root[root][idx]
            f = fingerprint(rec)
            if over_soft_cap(f):
                continue
            take(rec, f)
            progressed = True
        if not progressed or len(selected) >= n:
            break

    # Fill any remaining slots with the highest scores, honoring only the
    # hard per-lineage cap.
    if len(selected) < n:
        seen_ids = {id(rec) for rec in selected}
        for rec in ordered:
            if len(selected) >= n:
                break
            if id(rec) in seen_ids:
                continue
            f = fingerprint(rec)
            root = f["lineage_root"]
            if root and root_counts.get(root, 0) >= max_per_lineage:
                continue
            take(rec, f)
    return selected


def concentration(pool_records, key="lineage_root"):
    """Value -> count for one fingerprint dimension. ``key`` is one of
    ``hypothesis_id``, ``dataset_family``, ``operator_family``, ``lineage_root``."""
    counts = {}
    for rec in pool_records:
        value = fingerprint(rec)[key]
        if isinstance(value, (list, tuple, set)):
            for v in value or [None]:
                counts[v] = counts.get(v, 0) + 1
        else:
            counts[value] = counts.get(value, 0) + 1
    return counts


def pool_diversity_summary(pool_records):
    """Per-dimension concentration for logging and tests."""
    return {
        dim: concentration(pool_records, dim)
        for dim in ("hypothesis_id", "dataset_family", "operator_family", "lineage_root")
    }


# ---- pre-simulation candidate filter ----

def filter_candidates(candidates, simulated_exprs, pool_records, allow_research_mutation=True):
    """Reject candidates before simulation, so simulation budget is not wasted.

    A research mutation (has ``parent`` + ``mutation`` + ``research_question``,
    i.e. a single-variable robustness probe such as window-up / field-swap) is
    allowed to simulate — that is where information gain lives — but never as a
    duplicate of an already-simulated expression. Exploration candidates that
    are redundant with the pool or with an earlier kept candidate are rejected.
    Validation perturbations are built by the validator and never pass through
    this filter.

    Returns ``(kept, blocked)`` where ``blocked`` is a list of
    ``(candidate, reason)``.
    """
    simulated = set(simulated_exprs or [])
    pool = list(pool_records or [])
    # Pre-compute fingerprints once; is_redundant is called per candidate and
    # kept grows, so re-deriving them per call would be quadratic.
    pool_fps = [fingerprint(rec) for rec in pool]
    kept = []
    kept_fps = []
    blocked = []
    for cand in candidates:
        expr = cand.get("expression") or ""
        if expr in simulated:
            blocked.append((cand, "already-simulated"))
            continue
        if allow_research_mutation and _is_research_mutation(cand):
            kept.append(cand)
            kept_fps.append(fingerprint(cand))
            continue
        redundant, _ = is_redundant(
            cand, pool + kept, _pool_fps=pool_fps + kept_fps
        )
        if redundant:
            blocked.append((cand, "redundant"))
            continue
        kept.append(cand)
        kept_fps.append(fingerprint(cand))
    return kept, blocked


def _is_research_mutation(cand):
    return (
        bool(cand.get("mutation"))
        and cand.get("mutation") != "explore"
        and bool(cand.get("parent"))
        and bool(cand.get("research_question"))
    )
