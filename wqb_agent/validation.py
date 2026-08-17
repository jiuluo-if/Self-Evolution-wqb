"""Validation of suspiciously high-signal alphas.

An alpha with abnormally high Sharpe/Fitness is not trusted at face value.
We perturb it slightly (window step, smoothing, field swap) and re-simulate;
if the signal survives, it is validated, otherwise archived as noise.
"""

from .candidate import _swap_field, _window_change
from .simulator import Simulator
from .state import Experiment, score_of


class HighSignalValidator:
    def __init__(
        self,
        client,
        settings,
        max_concurrent=3,
        poll_timeout_sec=900,
        min_valid_fitness=1.0,
    ):
        self.settings = dict(settings)
        self.simulator = Simulator(
            client,
            max_concurrent=max_concurrent,
            poll_timeout_sec=poll_timeout_sec,
        )
        self.min_valid_fitness = min_valid_fitness

    def perturbations(self, expression, fields_used, alt_fields=None, max_perturbs=3):
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
                for alt in alt_fields:
                    if alt == primary:
                        continue
                    add(_swap_field(expression, primary, alt), f"field-swap->{alt}")
                    break
        return perms[:max_perturbs]

    def validate(self, record, alt_fields=None):
        """Run perturbations and check the signal is stable.

        Returns (stable, details) where details list per-perturbation results.
        """
        expression = record["expression"]
        fields_used = record.get("fields_used") or []
        round_no = record.get("round_no", 0)
        hypothesis_id = record.get("hypothesis_id")
        perms = self.perturbations(expression, fields_used, alt_fields=alt_fields)
        if not perms:
            return False, []

        experiments = [
            Experiment(
                round_no,
                hypothesis_id,
                new_expr,
                self.settings,
                fields_used,
                lineage=[expression],
                datasets=record.get("datasets") or [],
            )
            for new_expr, _ in perms
        ]
        self.simulator.run(experiments)

        details = []
        best = -1.0
        for exp in experiments:
            s = score_of(exp.metrics) if exp.metrics else -1.0
            best = max(best, s)
            details.append(
                {
                    "expression": exp.expression,
                    "score": s,
                    "sharpe": (exp.metrics or {}).get("sharpe"),
                    "turnover": (exp.metrics or {}).get("turnover"),
                    "error": exp.error,
                }
            )
        stable = best >= self.min_valid_fitness
        return stable, details
