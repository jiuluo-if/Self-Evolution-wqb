"""Simulator: runs a single submit -> poll -> fetch-metrics cycle.

Concurrency, budget and checkpointing live in BacktestScheduler; this module
deliberately does not manage any shared mutable state.
"""

from .failures import FailureKind


class Simulator:
    def __init__(self, client, max_concurrent=3, poll_timeout_sec=900):
        self.client = client
        # Kept for backward compatibility; the scheduler drives concurrency.
        self.max_concurrent = max_concurrent
        self.poll_timeout_sec = poll_timeout_sec

    def simulate(self, experiment, poll_timeout_sec=None):
        """Execute exactly one simulation and update the experiment in place."""
        experiment.status = "RUNNING"
        try:
            progress_url = self.client.submit_simulation(
                experiment.expression, experiment.settings
            )
            alpha_id = self.client.poll_progress(
                progress_url,
                timeout_sec=poll_timeout_sec or self.poll_timeout_sec,
            )
            payload = self.client.get_alpha(alpha_id)
            experiment.alpha_id = alpha_id
            experiment.metrics = _extract_metrics(payload)
            experiment.status = "DONE"
        except Exception as exc:  # noqa: BLE001
            experiment.error = f"{type(exc).__name__}: {exc}"
            experiment.status = "FAILED"
        return experiment

    def run(self, experiments):
        """Backward-compatible convenience wrapper around the scheduler."""
        from .scheduler import BacktestScheduler

        scheduler = BacktestScheduler(
            self.client,
            self,
            max_concurrent=self.max_concurrent,
            poll_timeout_sec=self.poll_timeout_sec,
        )
        scheduler.add_jobs(experiments)
        scheduler.run()
        return experiments


def classify_experiment_failure(experiment):
    """Convenience helper: map an experiment's error to a FailureKind."""
    error = experiment.error or ""
    if "WQBAuthError" in error:
        return FailureKind.AUTH
    if "WQBRateLimitError" in error or "429" in error:
        return FailureKind.RATE_LIMIT
    if "WQBTimeoutError" in error or "timed out" in error.lower():
        return FailureKind.TIMEOUT
    if "WQBRejectedError" in error or "422" in error or "400" in error:
        return FailureKind.SYNTAX
    if "WQBNotFoundError" in error or "404" in error:
        return FailureKind.DATA
    return FailureKind.INFRA


def _extract_metrics(payload):
    is_block = payload.get("is") or {}
    checks = is_block.get("checks") or []
    # Empty checks list must NOT be treated as PASS: all([]) == True would
    # silently approve an unverified alpha. Use a tri-state: True / False /
    # None (UNKNOWN) when there are no checks at all.
    if not checks:
        passed = None
    else:
        passed = all(bool(c.get("pass")) for c in checks)
    return {
        "sharpe": _num(is_block.get("sharpe")),
        "fitness": _num(is_block.get("fitness")),
        "turnover": _num(is_block.get("turnover")),
        "margin": _num(is_block.get("margin")),
        "returns": _num(is_block.get("returns")),
        "checks": [
            {"name": c.get("name"), "pass": bool(c.get("pass"))}
            for c in checks
        ],
        "passed": passed,
    }


def _num(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
