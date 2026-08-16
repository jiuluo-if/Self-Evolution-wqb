import re

WINDOW_STEPS = [5, 10, 20, 60]


class CandidateBuilder:
    def __init__(self, neutralization="SUBINDUSTRY"):
        self.neutralization = neutralization.lower()

    def build(self, hypothesis, fields, current_best, count=6):
        if current_best and current_best.get("expression"):
            return self._mutate_best(hypothesis, fields, current_best, count)
        return self._from_scratch(hypothesis, fields, count)

    @staticmethod
    def _main_field(fields):
        return fields[0]["id"] if fields else None

    def _from_scratch(self, hypothesis, fields, count):
        primary = self._main_field(fields)
        if not primary:
            return []
        reversal = hypothesis.get("direction") == "reversal"
        group = self.neutralization
        base = f"rank({primary})"
        candidates = [
            {
                "expression": f"-{base}" if reversal else base,
                "rationale": "Baseline: raw cross-sectional rank of primary field.",
                "mutation": "baseline",
                "parent": None,
            },
            {
                "expression": base if reversal else f"-{base}",
                "rationale": "Opposite sign of baseline rank.",
                "mutation": "sign-flip",
                "parent": None,
            },
            {
                "expression": f"rank(ts_rank({primary}, 20))",
                "rationale": "Time-series rank over 20d to smooth cross-sectional noise.",
                "mutation": "ts-rank-20",
                "parent": None,
            },
            {
                "expression": f"zscore({primary})",
                "rationale": "Standardize field with cross-sectional zscore.",
                "mutation": "zscore",
                "parent": None,
            },
            {
                "expression": f"group_neutralize({base}, {group})",
                "rationale": f"Neutralize baseline rank within {group}.",
                "mutation": f"neutralize-{group}",
                "parent": None,
            },
            {
                "expression": f"rank(ts_mean({primary}, 5))",
                "rationale": "Short 5d mean of field to lower turnover.",
                "mutation": "ts-mean-5",
                "parent": None,
            },
        ]
        return candidates[:count]

    def _mutate_best(self, hypothesis, fields, current_best, count):
        best_expr = current_best["expression"]
        best_fields = current_best.get("fields_used") or []
        primary = best_fields[0] if best_fields else self._main_field(fields)
        candidate_fields = [f["id"] for f in fields]
        secondary = None
        for fid in candidate_fields:
            if fid != primary:
                secondary = fid
                break
        group = self.neutralization
        candidates = []
        seen_exprs = {best_expr}

        def add(expression, rationale, mutation):
            if expression in seen_exprs:
                return
            seen_exprs.add(expression)
            candidates.append(
                {
                    "expression": expression,
                    "rationale": rationale,
                    "mutation": mutation,
                    "parent": current_best.get("id"),
                }
            )

        if secondary:
            swapped = re.sub(re.escape(primary), secondary, best_expr)
            add(
                swapped,
                f"Single-variable field swap: {primary} -> {secondary}.",
                "field-swap",
            )

        add(
            f"ts_mean({best_expr}, 5)",
            "Wrap in 5d ts_mean to reduce turnover.",
            "smooth-ts-mean-5",
        )

        add(
            f"group_neutralize({best_expr}, {group})",
            f"Add {group} neutralization layer.",
            f"neutralize-{group}",
        )

        new_window = self._next_window(best_expr)
        if new_window:
            add(
                new_window,
                "Single-step window change on the time-series operator.",
                "window-step",
            )

        if best_expr.startswith("-"):
            add(best_expr[1:], "Flip sign from negative to positive.", "sign-flip")
        else:
            add(f"-{best_expr}", "Flip sign from positive to negative.", "sign-flip")

        if "ts_rank" not in best_expr:
            add(
                f"rank(ts_rank({primary}, 20))",
                "Replace primary field with 20d time-series rank.",
                "ts-rank-20",
            )

        if len(candidates) < count and "zscore(" not in best_expr:
            add(
                f"zscore({best_expr})",
                "Add a zscore normalization layer.",
                "zscore-layer",
            )

        return candidates[:count]

    @staticmethod
    def _next_window(expression):
        pattern = re.compile(
            r"(ts_(?:rank|mean|std_dev|sum|delta|min|max)\([^,]+,\s*)(\d+)"
        )
        found = None
        for m in pattern.finditer(expression):
            found = m
        if not found:
            return None
        start, end = found.start(), found.end()
        current = int(found.group(2))
        target = None
        for w in WINDOW_STEPS:
            if w > current:
                target = w
                break
        if target is None:
            target = WINDOW_STEPS[-1]
        new_expr = (
            expression[:start]
            + found.group(1)
            + str(target)
            + expression[end:]
        )
        if new_expr == expression:
            return None
        return new_expr
