class Reflector:
    def __init__(self, memory, success_sharpe=1.0, promising_sharpe=0.5):
        self.memory = memory
        self.success_sharpe = success_sharpe
        self.promising_sharpe = promising_sharpe

    def reflect(self, round_no, hypothesis, experiments):
        results = []
        for exp in experiments:
            verdict = self._classify(exp)
            results.append({"experiment": exp, "verdict": verdict})
            self._learn(round_no, hypothesis, exp, verdict)

        best = self._update_best(results)
        self._generate_next(round_no, hypothesis, results)
        self.memory.updated_round = round_no
        self.memory.compress()
        self.memory.save()

        return self._summary(round_no, hypothesis, results, best)

    def _classify(self, exp):
        if exp.status == "FAILED":
            return {"label": "FAIL", "reason": self._diagnose_error(exp)}
        metrics = exp.metrics or {}
        checks = metrics.get("checks") or []
        failed_checks = [c["name"] for c in checks if not c["pass"]]
        sharpe = metrics.get("sharpe")
        turnover = metrics.get("turnover")
        if sharpe is None:
            return {"label": "FAIL", "reason": "missing sharpe metric"}
        reason = []
        if failed_checks:
            reason.append(f"checks failed: {failed_checks}")
        if sharpe <= 0:
            reason.append("sharpe <= 0 (no predictive power)")
        if turnover is not None and turnover > 1.5:
            reason.append(f"turnover too high: {turnover:.2f}")
        if sharpe >= self.success_sharpe:
            return {"label": "SUCCESS", "reason": " or ".join(reason) or "passed thresholds"}
        if sharpe >= self.promising_sharpe:
            return {"label": "PROMISING", "reason": " or ".join(reason) or "weak positive signal"}
        return {"label": "FAIL", "reason": " or ".join(reason) or f"sharpe too low: {sharpe:.3f}"}

    def _diagnose_error(self, exp):
        error = exp.error or ""
        if "Simulation rejected" in error or "422" in error or "400" in error:
            return f"syntax/settings rejection: {error[:120]}"
        if "timed out" in error.lower():
            return "polling timed out"
        return f"runtime error: {error[:120]}"

    def _learn(self, round_no, hypothesis, exp, verdict):
        if exp.status == "FAILED":
            self.memory.add_avoid(
                self._direction_key(exp),
                self._diagnose_error(exp),
                round_no,
            )
            return

        metrics = exp.metrics or {}
        fields = exp.fields_used
        field_label = ",".join(fields) if fields else "?"

        if verdict["label"] == "SUCCESS":
            self.memory.add_lesson(
                f"Combination {exp.expression} achieves Sharpe {metrics.get('sharpe')} "
                f"on fields [{field_label}].",
                round_no,
                evidence=3,
                confidence=0.7,
            )
        elif verdict["label"] == "PROMISING":
            self.memory.add_lesson(
                f"Fields [{field_label}] show weak positive signal; worth iterating.",
                round_no,
                evidence=1,
                confidence=0.3,
            )
            self.memory.add_next(
                f"Iterate on [{field_label}] with smoothing / neutralization variants.",
                priority=4,
                source=round_no,
                round_no=round_no,
            )
        else:
            self.memory.add_avoid(
                self._direction_key(exp),
                f"sharpe={metrics.get('sharpe')}, turnover={metrics.get('turnover')}",
                round_no,
            )
            self.memory.add_lesson(
                f"Direction on [{field_label}] has no predictive power (Sharpe {metrics.get('sharpe')}).",
                round_no,
                evidence=2,
                confidence=0.4,
            )

        turnover = metrics.get("turnover")
        if turnover is not None and turnover > 1.0:
            self.memory.add_lesson(
                f"Raw expression on [{field_label}] has turnover {turnover:.2f}; smoothing needed.",
                round_no,
                evidence=1,
                confidence=0.3,
            )

    @staticmethod
    def _direction_key(exp):
        return exp.expression[:80]

    def _score(self, exp):
        metrics = exp.metrics or {}
        fitness = metrics.get("fitness")
        sharpe = metrics.get("sharpe")
        if fitness is not None and fitness != 0:
            return fitness
        return sharpe if sharpe is not None else -1.0

    def _update_best(self, results):
        done = [r for r in results if r["experiment"].metrics]
        if not done:
            return self.memory.current_best
        best_exp = max(done, key=lambda r: self._score(r["experiment"]))["experiment"]
        best_score = self._score(best_exp)
        current_score = None
        if self.memory.current_best and self.memory.current_best.get("metrics"):
            current_score = self._score_of(self.memory.current_best["metrics"])
        if current_score is None or best_score > current_score:
            self.memory.set_current_best(best_exp)
            return best_exp.to_dict()
        return self.memory.current_best

    @staticmethod
    def _score_of(metrics):
        fitness = metrics.get("fitness")
        sharpe = metrics.get("sharpe")
        if fitness is not None and fitness != 0:
            return fitness
        return sharpe if sharpe is not None else -1.0

    def _generate_next(self, round_no, hypothesis, results):
        successes = [r for r in results if r["verdict"]["label"] == "SUCCESS"]
        promising = [r for r in results if r["verdict"]["label"] == "PROMISING"]
        failed_all = all(r["verdict"]["label"] == "FAIL" for r in results)

        if successes:
            for r in successes[:2]:
                self.memory.add_next(
                    f"Deepen successful expression: {r['experiment'].expression}",
                    priority=5,
                    source=round_no,
                    round_no=round_no,
                )
        if promising:
            for r in promising[:2]:
                self.memory.add_next(
                    f"Improve promising expression: {r['experiment'].expression} "
                    f"(smoothing, neutralization, field swap)",
                    priority=4,
                    source=round_no,
                    round_no=round_no,
                )
        if failed_all:
            tags = hypothesis.get("tags", [])
            self.memory.add_next(
                f"Switch research direction away from tags {tags}",
                priority=3,
                source=round_no,
                round_no=round_no,
            )

    def _summary(self, round_no, hypothesis, results, best):
        labels = {}
        for r in results:
            labels[r["verdict"]["label"]] = labels.get(r["verdict"]["label"], 0) + 1
        return {
            "round": round_no,
            "hypothesis": hypothesis.get("statement", ""),
            "experiment_count": len(results),
            "verdicts": labels,
            "best": (
                {
                    "expression": best["expression"],
                    "sharpe": (best.get("metrics") or {}).get("sharpe"),
                    "fitness": (best.get("metrics") or {}).get("fitness"),
                }
                if best
                else None
            ),
        }
