"""Robustness validation of suspiciously high-signal alphas.

An alpha with abnormally high Sharpe/Fitness is not trusted at face value. We
perturb it (window step up/down, smoothing, semantically-nearby field swap) and
re-simulate. It is only validated when the majority of perturbations survive,
the median perturbed score stays above the minimum, and the signal retains a
meaningful fraction of the original performance. A single lucky perturbation is
never enough.
"""

import math

from .candidate import _swap_field, _window_change
from .simulator import Simulator
from .state import Experiment, score_of


def _pick_nearby_field(primary, alt_fields):
    """Choose a semantically-related replacement field: same dataset, then
    same category, then any other known field."""
    metas = [f for f in alt_fields if isinstance(f, dict)]
    ids = [f["id"] for f in metas] if metas else list(alt_fields or [])
    primary_meta = next((m for m in metas if m.get("id") == primary), None)
    if primary_meta:
        same_ds = [
            m["id"]
            for m in metas
            if m.get("dataset") == primary_meta.get("dataset")
            and m["id"] != primary
        ]
        if same_ds:
            return same_ds[0]
        same_cat = [
            m["id"]
            for m in metas
            if m.get("category") == primary_meta.get("category")
            and m["id"] != primary
        ]
        if same_cat:
            return same_cat[0]
    for fid in ids:
        if fid != primary:
            return fid
    return None


class HighSignalValidator:
    def __init__(
        self,
        client,
        settings,
        max_concurrent=3,
        poll_timeout_sec=900,
        min_valid_fitness=1.0,
        majority_ratio=0.6,
        min_pass=2,
        retention_ratio=0.5,
    ):
        self.settings = dict(settings)
        self.simulator = Simulator(
            client,
            max_concurrent=max_concurrent,
            poll_timeout_sec=poll_timeout_sec,
        )
        self.max_concurrent = max_concurrent
        self.poll_timeout_sec = poll_timeout_sec
        self.min_valid_fitness = min_valid_fitness
        self.majority_ratio = majority_ratio
        self.min_pass = min_pass
        self.retention_ratio = retention_ratio

    def perturbations(self, expression, fields_used, alt_fields=None, max_perturbs=4):
        perms = []
        seen = set()

        def add(new_expr, label):
            if new_expr and new_expr != expression and new_expr not in seen:
                seen.add(new_expr)
                perms.append((new_expr, label))

        add(_window_change(expression, +1), "window-up")
        add(_window_change(expression, -1), "window-down")
        if "ts_mean(" not in expression:
            add(f"ts_mean({expression}, 5)", "smooth-ts-mean-5")
        if alt_fields:
            primary = fields_used[0] if fields_used else None
            if primary:
                nearby = _pick_nearby_field(primary, alt_fields)
                if nearby:
                    add(_swap_field(expression, primary, nearby),
                        f"field-swap->{nearby}")
        return perms[:max_perturbs]

    def build_perturbation_jobs(self, record, alt_fields=None):
        """Return (jobs, perms) so the caller can run them through a shared
        scheduler that charges the unified simulation budget."""
        expression = record["expression"]
        fields_used = record.get("fields_used") or []
        perms = self.perturbations(expression, fields_used, alt_fields=alt_fields)
        jobs = [
            Experiment(
                record.get("round_no", 0),
                record.get("hypothesis_id"),
                new_expr,
                self.settings,
                fields_used,
                lineage=[expression],
                datasets=record.get("datasets") or [],
            )
            for new_expr, _ in perms
        ]
        for job, (_, label) in zip(jobs, perms):
            job.mutation = f"validation-{label}"
        return jobs, perms

    def decide(self, record, results):
        """Combined robustness verdict.

        results: list of {expression, score, sharpe, turnover, checks_passed,
        error}. Requires a majority of perturbations to pass, a surviving
        median, WQB checks to pass, and a minimum number of survivors.
        """
        if not results:
            return False, []
        original = score_of(record.get("metrics"))
        scores = [r["score"] for r in results]
        passed = [
            r
            for r in results
            if r["score"] >= self.min_valid_fitness and r.get("checks_passed")
        ]
        total = len(results)
        need = max(self.min_pass, int(math.ceil(total * self.majority_ratio)))
        if len(passed) < need:
            return False, results

        ordered = sorted(scores)
        median_score = ordered[len(ordered) // 2]
        # Retention can only be enforced when an original baseline exists; a
        # validation without the original alpha still requires majority +
        # median + checks.
        retention = (median_score / original) if original and original > 0 else 0.0
        retention_ok = True if (not original or original <= 0) else (
            retention >= self.retention_ratio
        )
        stable = (
            median_score >= self.min_valid_fitness
            and retention_ok
            and len(passed) >= self.min_pass
        )
        return stable, results

    def validate(self, record, alt_fields=None):
        """Backward-compatible entry point that runs its own simulations."""
        jobs, _ = self.build_perturbation_jobs(record, alt_fields=alt_fields)
        if not jobs:
            return False, []
        self.simulator.run(jobs)
        results = []
        for exp in jobs:
            passed = (exp.metrics or {}).get("passed") is True
            results.append(
                {
                    "expression": exp.expression,
                    "score": score_of(exp.metrics) if exp.metrics else -1.0,
                    "sharpe": (exp.metrics or {}).get("sharpe"),
                    "turnover": (exp.metrics or {}).get("turnover"),
                    "checks_passed": passed,
                    "error": exp.error,
                }
            )
        return self.decide(record, results)
