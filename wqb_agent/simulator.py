from concurrent.futures import ThreadPoolExecutor, as_completed


class Simulator:
    def __init__(self, client, max_concurrent=3, poll_timeout_sec=900):
        self.client = client
        self.max_concurrent = max_concurrent
        self.poll_timeout_sec = poll_timeout_sec

    def run(self, experiments):
        done = []
        pending = list(experiments)
        while pending:
            batch = pending[: self.max_concurrent]
            pending = pending[self.max_concurrent :]
            done.extend(self._run_batch(batch))
        return done

    def _run_batch(self, experiments):
        with ThreadPoolExecutor(max_workers=len(experiments)) as pool:
            futures = {
                pool.submit(self._simulate_one, exp): exp for exp in experiments
            }
            for future in as_completed(futures):
                future.result()
        return experiments

    def _simulate_one(self, experiment):
        experiment.status = "RUNNING"
        try:
            progress_url = self.client.submit_simulation(
                experiment.expression, experiment.settings
            )
            alpha_id = self.client.poll_progress(
                progress_url, timeout_sec=self.poll_timeout_sec
            )
            payload = self.client.get_alpha(alpha_id)
            experiment.alpha_id = alpha_id
            experiment.metrics = _extract_metrics(payload)
            experiment.status = "DONE"
        except Exception as exc:
            experiment.error = f"{type(exc).__name__}: {exc}"
            experiment.status = "FAILED"
        return experiment


def _extract_metrics(payload):
    is_block = payload.get("is") or {}
    checks = is_block.get("checks") or []
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
        "passed": all(bool(c.get("pass")) for c in checks),
    }


def _num(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
