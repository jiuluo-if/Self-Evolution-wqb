import re

from .diversity import extract_fields

WINDOW_STEPS = [5, 10, 20, 60]

_TS_OP_RE = re.compile(
    r"(ts_(?:rank|mean|std_dev|sum|delta|min|max|zscore|corr)\([^,]+,\s*)(\d+)"
)


def _window_change(expression, direction):
    """Step the last time-series window by one WINDOW_STEPS step.

    direction=+1 moves up, -1 moves down. Returns None if impossible.
    """
    found = None
    for m in _TS_OP_RE.finditer(expression):
        found = m
    if not found:
        return None
    current = int(found.group(2))
    steps = WINDOW_STEPS
    if direction > 0:
        target = next((w for w in steps if w > current), steps[-1])
    else:
        smaller = [w for w in steps if w < current]
        target = smaller[-1] if smaller else current
    if target == current:
        return None
    return (
        expression[: found.start()]
        + found.group(1)
        + str(target)
        + expression[found.end():]
    )


def _swap_field(expression, old_field, new_field):
    if new_field == old_field:
        return None
    pattern = re.compile(r"\b" + re.escape(old_field) + r"\b")
    swapped = pattern.sub(new_field, expression)
    if swapped == expression:
        return None
    return swapped


# ---- hypothesis-driven operator families ----
#
# A candidate is built from the hypothesis, never from a generic bag of
# operators. Each hypothesis type maps to a small family of operators that
# encode its economic content.

_REVERSAL_TEMPLATES = [
    "-rank({p})",
    "-ts_rank({p}, 20)",
    "-rank(ts_zscore({p}, 20))",
    "-rank(ts_mean({p}, 5))",
    "group_neutralize(-rank({p}), {g})",
    "-rank(ts_rank({p}, 5))",
]

_MOMENTUM_TEMPLATES = [
    "rank(ts_mean({p}, 20))",
    "rank(ts_rank({p}, 20))",
    "rank(ts_sum({p}, 20))",
    "rank(ts_mean(ts_rank({p}, 5), 20))",
    "group_neutralize(rank(ts_mean({p}, 20)), {g})",
    "rank(ts_rank({p}, 60))",
]

_REVISION_TEMPLATES = [
    "rank(ts_delta({p}, 5))",
    "-rank(ts_delta({p}, 5))",
    "rank(ts_mean(ts_delta({p}, 5), 5))",
    "group_neutralize(rank(ts_delta({p}, 5)), {g})",
    "rank(ts_delta({p}, 10))",
]

_CROSS_SECTIONAL_TEMPLATES = [
    "rank({p})",
    "zscore({p})",
    "group_neutralize(rank({p}), {g})",
    "rank(ts_rank({p}, 20))",
    "-rank({p})",
    "group_neutralize(zscore({p}), {g})",
]

_RELATIONSHIP_TEMPLATES = [
    "rank(ts_corr({p}, {s}, 20))",
    "rank(ts_corr({p}, {s}, 60))",
    "zscore(ts_delta({p}, 5)) - zscore(ts_delta({s}, 5))",
    "rank({p} - {s})",
    "group_neutralize(rank(ts_corr({p}, {s}, 20)), {g})",
]


def _family_for(hypothesis):
    tags = {t.lower() for t in (hypothesis.get("tags") or [])}
    direction = hypothesis.get("direction")
    if direction == "reversal":
        return "reversal"
    if tags & {"momentum", "trend", "continuation"}:
        return "momentum"
    if tags & {"revision", "estimate", "forecast", "revision-upward"}:
        return "revision"
    if tags & {"relationship", "corr", "correlation", "ratio", "spread",
               "cross", "pair"}:
        return "relationship"
    return "cross_sectional"


def _templates_for(hypothesis):
    family = _family_for(hypothesis)
    if family == "reversal":
        return _REVERSAL_TEMPLATES
    if family == "momentum":
        return _MOMENTUM_TEMPLATES
    if family == "revision":
        return _REVISION_TEMPLATES
    if family == "relationship":
        return _RELATIONSHIP_TEMPLATES
    return _CROSS_SECTIONAL_TEMPLATES


class CandidateBuilder:
    """Builds candidates in two pools:

    Exploration pool: novel hypothesis-driven structures on new fields.
    Deepening pool:   single-variable, single-parameter local optimizations
                      of existing alpha lineages.
    """

    def __init__(self, neutralization="SUBINDUSTRY", max_deepen_per_lineage=3):
        self.neutralization = neutralization.lower()
        self.max_deepen_per_lineage = max_deepen_per_lineage

    def build(self, hypothesis, fields, current_best, count=6):
        """Backward-compatible entry point: deepens current_best when present,
        otherwise explores from scratch."""
        active = [current_best] if current_best else []
        return self.build_pools(hypothesis, fields, active, total=count)

    def build_pools(self, hypothesis, fields, active_alphas, total=6, explore_ratio=0.5):
        deepenable = [
            a
            for a in active_alphas
            if (a.get("attempts", 0) < self.max_deepen_per_lineage)
            and a.get("expression")
        ]
        if not deepenable:
            return self.build_explore(hypothesis, fields, total)
        explore_count = max(1, int(round(total * explore_ratio)))
        explore = self.build_explore(hypothesis, fields, explore_count)
        deepen = self.build_deepen(fields, deepenable, total - len(explore))
        return explore + deepen

    # ---- exploration pool ----

    def build_explore(self, hypothesis, fields, count):
        primary = fields[0]["id"] if fields else None
        if not primary:
            return []
        secondary = fields[1]["id"] if len(fields) > 1 else None
        g = self.neutralization
        all_ids = [f["id"] if isinstance(f, dict) else f for f in fields]

        templates = _templates_for(hypothesis)
        candidates = []
        seen = set()
        for t in templates:
            filled = t.format(p=primary, s=secondary or primary, g=g)
            if filled in seen:
                continue
            seen.add(filled)
            candidates.append(
                {
                    "expression": filled,
                    "rationale": (
                        f"Exploration on {primary} via "
                        f"{_family_for(hypothesis)} family."
                    ),
                    "mutation": "explore",
                    "parent": None,
                    "lineage": [],
                    "fields_used": extract_fields(filled, all_ids),
                }
            )
            if len(candidates) >= count:
                break
        return candidates

    # ---- deepening pool ----

    def build_deepen(self, fields, active_alphas, count):
        candidates = []
        seen = set()
        for alpha in active_alphas:
            expr = alpha.get("expression") or ""
            if not expr:
                continue
            if alpha.get("attempts", 0) >= self.max_deepen_per_lineage:
                continue
            alpha_fields = list(
                dict.fromkeys(
                    list(alpha.get("fields_used") or []) + self._field_ids(fields)
                )
            )
            field_meta = {
                f["id"]: f for f in fields if isinstance(f, dict) and f.get("id")
            }
            for cand in self._mutations(
                expr, alpha.get("lineage") or [], alpha_fields, field_meta
            ):
                if cand["expression"] in seen:
                    continue
                seen.add(cand["expression"])
                candidates.append(cand)
                if len(candidates) >= count:
                    return candidates
        return candidates

    def _mutations(self, expr, lineage, field_ids, field_meta):
        """Single-variable, single-parameter mutations of an existing alpha.

        field swap is restricted to a semantically-close field (same dataset,
        then same category, then expression-internal secondary, then any known
        field) so unrelated fields are not swapped in.
        """
        primary = self._primary_field(expr, field_ids)
        g = self.neutralization
        mutations = []

        if primary:
            alt = self._semantic_alt_field(expr, primary, field_ids, field_meta)
            if alt:
                swapped = _swap_field(expr, primary, alt)
                if swapped:
                    mutations.append(
                        {
                            "expression": swapped,
                            "rationale": (
                                f"Single-variable field swap: {primary} -> {alt} "
                                f"(semantically related)."
                            ),
                            "mutation": "field-swap",
                            "parent": expr,
                            "lineage": [expr] + lineage,
                            "fields_used": extract_fields(swapped, field_ids),
                        }
                    )

        up = _window_change(expr, +1)
        if up:
            mutations.append(
                {
                    "expression": up,
                    "rationale": "Single window step up on the time-series operator.",
                    "mutation": "window-up",
                    "parent": expr,
                    "lineage": [expr] + lineage,
                    "fields_used": extract_fields(up, field_ids),
                }
            )

        down = _window_change(expr, -1)
        if down:
            mutations.append(
                {
                    "expression": down,
                    "rationale": "Single window step down on the time-series operator.",
                    "mutation": "window-down",
                    "parent": expr,
                    "lineage": [expr] + lineage,
                    "fields_used": extract_fields(down, field_ids),
                }
            )

        if "ts_mean(" not in expr:
            smooth = f"ts_mean({expr}, 5)"
            mutations.append(
                {
                    "expression": smooth,
                    "rationale": "Wrap in 5d ts_mean to reduce turnover.",
                    "mutation": "smooth-ts-mean-5",
                    "parent": expr,
                    "lineage": [expr] + lineage,
                    "fields_used": extract_fields(smooth, field_ids),
                }
            )

        if "group_neutralize" not in expr:
            neutralized = f"group_neutralize({expr}, {g})"
            mutations.append(
                {
                    "expression": neutralized,
                    "rationale": f"Add {g} neutralization layer.",
                    "mutation": f"neutralize-{g}",
                    "parent": expr,
                    "lineage": [expr] + lineage,
                    "fields_used": extract_fields(neutralized, field_ids),
                }
            )

        return mutations

    @staticmethod
    def _semantic_alt_field(expr, primary, field_ids, field_meta):
        """Pick a semantically-close replacement for the primary field."""
        meta = field_meta.get(primary)
        if meta:
            same_ds = [
                f
                for f in field_ids
                if f != primary
                and field_meta.get(f)
                and field_meta[f].get("dataset") == meta.get("dataset")
            ]
            if same_ds:
                return same_ds[0]
            same_cat = [
                f
                for f in field_ids
                if f != primary
                and field_meta.get(f)
                and field_meta[f].get("category") == meta.get("category")
            ]
            if same_cat:
                return same_cat[0]
        secondary = CandidateBuilder._secondary_field(expr, field_ids)
        if secondary:
            return secondary
        others = [f for f in field_ids if f != primary]
        return others[0] if others else None

    @staticmethod
    def _field_ids(fields):
        return [f["id"] if isinstance(f, dict) else f for f in fields]

    @staticmethod
    def _primary_field(expr, fields):
        for f in fields:
            fid = f["id"] if isinstance(f, dict) else f
            if re.search(r"\b" + re.escape(fid) + r"\b", expr):
                return fid
        return None

    @staticmethod
    def _secondary_field(expr, fields):
        primary = CandidateBuilder._primary_field(expr, fields)
        for f in fields:
            fid = f["id"] if isinstance(f, dict) else f
            if fid != primary:
                return fid
        return None
