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
        deepen = self.build_deepen(
            fields, deepenable, total - len(explore), hypothesis=hypothesis
        )
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
        family = _family_for(hypothesis)
        mechanism = hypothesis.get("expected_mechanism")
        expected_dir = hypothesis.get("expected_direction")
        sign = (expected_dir or {}).get("sign")
        horizon = hypothesis.get("expected_horizon_days")
        semantic = (hypothesis.get("field_semantics") or {}).get("primary")
        research_question = (
            hypothesis.get("research_question")
            or hypothesis.get("statement", "")
            or (
                f"Does {primary} in direction {sign or 'the hypothesis direction'} "
                "predict forward returns?"
            )
        )
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
                    "rationale": self._explore_rationale(
                        hypothesis, primary, filled, family, sign, horizon
                    ),
                    "research_question": research_question,
                    "hypothesis_id": hypothesis.get("id"),
                    "mechanism": mechanism,
                    "field_semantic": semantic,
                    "expected_direction": expected_dir,
                    "expected_horizon_days": horizon,
                    "falsification_variant": self._is_falsification(
                        family, sign
                    ),
                    "field_discovery_reason": self._field_discovery_reason(
                        fields[0]
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

    @staticmethod
    def _explore_rationale(hypothesis, primary, template, family, sign, horizon):
        """Why this field, this operator, this direction, this window."""
        parts = [f"Exploration on {primary} via {family} family"]
        mechanism = hypothesis.get("expected_mechanism")
        if mechanism:
            parts.append(f"testing mechanism: {mechanism}")
        if sign:
            parts.append(f"expected direction: {sign}")
        if horizon:
            parts.append(f"horizon prior: {horizon}d")
        return "; ".join(parts) + "."

    @staticmethod
    def _is_falsification(family, sign):
        """A template whose sign contradicts the hypothesis's expected sign is
        an explicit falsification probe, never a silent direction flip."""
        if family == "reversal":
            return sign is not None and sign != "negative"
        if family == "momentum":
            return sign is not None and sign != "positive"
        # Mixed-sign families (revision / cross-sectional / relationship)
        # cannot be judged template-by-template.
        return False

    # ---- deepening pool ----

    def build_deepen(self, fields, active_alphas, count, hypothesis=None):
        candidates = []
        seen = set()
        base_ctx = self._deepen_context(hypothesis)
        for alpha in active_alphas:
            expr = alpha.get("expression") or ""
            if not expr:
                continue
            if alpha.get("attempts", 0) >= self.max_deepen_per_lineage:
                continue
            # A deepening mutation inherits the hypothesis of the alpha it is
            # refining, not the hypothesis of the round that happens to run it.
            # Otherwise the same lineage root would masquerade as several
            # independent hypotheses in the diversity fingerprint.
            ctx = dict(base_ctx)
            if alpha.get("hypothesis_id"):
                ctx["hypothesis_id"] = alpha.get("hypothesis_id")
            alpha_fields = list(
                dict.fromkeys(
                    list(alpha.get("fields_used") or []) + self._field_ids(fields)
                )
            )
            field_meta = {
                f["id"]: f for f in fields if isinstance(f, dict) and f.get("id")
            }
            for cand in self._mutations(
                expr, alpha.get("lineage") or [], alpha_fields, field_meta, ctx
            ):
                if cand["expression"] in seen:
                    continue
                seen.add(cand["expression"])
                candidates.append(cand)
                if len(candidates) >= count:
                    return candidates
        return candidates

    @staticmethod
    def _deepen_context(hypothesis):
        mechanism = (hypothesis or {}).get("expected_mechanism")
        return {
            "mechanism": mechanism,
            "mechanism_clause": mechanism or "the hypothesized effect",
            "hypothesis_id": (hypothesis or {}).get("id"),
            "expected_direction": (hypothesis or {}).get("expected_direction"),
            "expected_horizon_days": (hypothesis or {}).get(
                "expected_horizon_days"
            ),
            "field_semantic": (
                ((hypothesis or {}).get("field_semantics") or {}).get("primary")
            ),
        }

    def _mutations(self, expr, lineage, field_ids, field_meta, ctx=None):
        """Single-variable, single-parameter mutations of an existing alpha.

        field swap is restricted to a semantically-close field (same dataset,
        then same category, then expression-internal secondary, then any known
        field) so unrelated fields are not swapped in. Each mutation changes
        exactly one thing and carries a research question that names the
        mechanism being probed.
        """
        ctx = ctx or self._deepen_context(None)
        primary = self._primary_field(expr, field_ids)
        g = self.neutralization
        mutations = []
        mechanism = ctx["mechanism_clause"]

        def stamp(cand, mutation):
            cand["hypothesis_id"] = ctx["hypothesis_id"]
            cand["mechanism"] = ctx["mechanism"]
            cand["field_semantic"] = ctx["field_semantic"]
            cand["expected_direction"] = ctx["expected_direction"]
            cand["expected_horizon_days"] = ctx["expected_horizon_days"]
            cand["falsification_variant"] = False
            cand["field_discovery_reason"] = self._field_discovery_reason(
                field_meta.get(primary)
            )
            cand["mutation"] = mutation
            return cand

        if primary:
            alt = self._semantic_alt_field(expr, primary, field_ids, field_meta)
            if alt:
                swapped = _swap_field(expr, primary, alt)
                if swapped:
                    mutations.append(
                        stamp(
                            {
                                "expression": swapped,
                                "rationale": (
                                    f"Single-variable field swap: {primary} -> {alt} "
                                    f"(semantically related)."
                                ),
                                "research_question": (
                                    f"Does substituting the primary field of {expr} "
                                    f"with {alt} preserve {mechanism}?"
                                ),
                                "parent": expr,
                                "lineage": [expr] + lineage,
                                "fields_used": extract_fields(swapped, field_ids),
                            },
                            "field-swap",
                        )
                    )

        up = _window_change(expr, +1)
        if up:
            target = self._last_window(up)
            mutations.append(
                stamp(
                    {
                        "expression": up,
                        "rationale": "Single window step up on the time-series operator.",
                        "research_question": (
                            f"Does {expr} survive a longer horizon neighbor "
                            f"({target}d) for {mechanism}?"
                        ),
                        "parent": expr,
                        "lineage": [expr] + lineage,
                        "fields_used": extract_fields(up, field_ids),
                    },
                    "window-up",
                )
            )

        down = _window_change(expr, -1)
        if down:
            target = self._last_window(down)
            mutations.append(
                stamp(
                    {
                        "expression": down,
                        "rationale": "Single window step down on the time-series operator.",
                        "research_question": (
                            f"Does {expr} survive a shorter horizon neighbor "
                            f"({target}d) for {mechanism}?"
                        ),
                        "parent": expr,
                        "lineage": [expr] + lineage,
                        "fields_used": extract_fields(down, field_ids),
                    },
                    "window-down",
                )
            )

        if "ts_mean(" not in expr:
            smooth = f"ts_mean({expr}, 5)"
            mutations.append(
                stamp(
                    {
                        "expression": smooth,
                        "rationale": "Wrap in 5d ts_mean to reduce turnover.",
                        "research_question": (
                            f"Does 5d smoothing of {expr} preserve {mechanism} "
                            "at lower turnover?"
                        ),
                        "parent": expr,
                        "lineage": [expr] + lineage,
                        "fields_used": extract_fields(smooth, field_ids),
                    },
                    "smooth-ts-mean-5",
                )
            )

        if "group_neutralize" not in expr:
            neutralized = f"group_neutralize({expr}, {g})"
            mutations.append(
                stamp(
                    {
                        "expression": neutralized,
                        "rationale": f"Add {g} neutralization layer.",
                        "research_question": (
                            f"Does {g} neutralization of {expr} preserve "
                            f"{mechanism}?"
                        ),
                        "parent": expr,
                        "lineage": [expr] + lineage,
                        "fields_used": extract_fields(neutralized, field_ids),
                    },
                    f"neutralize-{g}",
                )
            )

        return mutations

    @staticmethod
    def _last_window(expression):
        """The time-series window of a mutated expression (for questions)."""
        matches = list(_TS_OP_RE.finditer(expression))
        if not matches:
            return None
        return int(matches[-1].group(2))

    @staticmethod
    def _field_discovery_reason(field):
        """Compact, interpretable summary of why a real field was selected.
        Only a reference to the discovery evidence; the full field_match is
        not duplicated into the candidate."""
        if not isinstance(field, dict):
            return None
        fm = field.get("field_match")
        if not isinstance(fm, dict):
            return None
        return {
            "dataset": field.get("dataset"),
            "semantic_concept": fm.get("semantic_concept"),
            "matched_terms": list(fm.get("matched_terms") or []),
            "semantic_score": fm.get("semantic_score"),
            "match_score": fm.get("total_score"),
        }

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
