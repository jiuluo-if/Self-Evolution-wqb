import re

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
        reversal = hypothesis.get("direction") == "reversal"
        sign = "-" if reversal else ""
        g = self.neutralization

        templates = [
            f"{sign}rank({primary})",
            f"rank(ts_rank({primary}, 20))",
            f"{sign}rank(ts_mean({primary}, 5))",
            f"{sign}rank(ts_delta({primary}, 5))",
            f"group_neutralize({sign}rank({primary}), {g})",
            f"{sign}zscore({primary})",
            f"rank(ts_zscore({primary}, 20))",
            f"rank(ts_std_dev({primary}, 20))",
            f"-ts_delta(rank({primary}), 5)",
        ]
        if secondary:
            templates.append(f"{sign}rank(ts_corr({primary}, {secondary}, 20))")

        candidates = []
        seen = set()
        for t in templates:
            if t in seen:
                continue
            seen.add(t)
            candidates.append(
                {
                    "expression": t,
                    "rationale": f"Exploration on {primary}.",
                    "mutation": "explore",
                    "parent": None,
                    "lineage": [],
                    "fields_used": [f["id"] for f in fields],
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
            for cand in self._mutations(expr, alpha.get("lineage") or [], alpha_fields):
                if cand["expression"] in seen:
                    continue
                seen.add(cand["expression"])
                candidates.append(cand)
                if len(candidates) >= count:
                    return candidates
        return candidates

    def _mutations(self, expr, lineage, field_ids):
        """Single-variable, single-parameter mutations of an existing alpha."""
        primary = self._primary_field(expr, field_ids)
        secondary = self._secondary_field(expr, field_ids)
        g = self.neutralization
        mutations = []

        if primary and secondary:
            swapped = _swap_field(expr, primary, secondary)
            if swapped:
                mutations.append(
                    {
                        "expression": swapped,
                        "rationale": f"Single-variable field swap: {primary} -> {secondary}.",
                        "mutation": "field-swap",
                        "parent": expr,
                        "lineage": [expr] + lineage,
                        "fields_used": list(field_ids),
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
                    "fields_used": list(field_ids),
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
                    "fields_used": list(field_ids),
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
                    "fields_used": list(field_ids),
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
                    "fields_used": list(field_ids),
                }
            )

        return mutations

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
