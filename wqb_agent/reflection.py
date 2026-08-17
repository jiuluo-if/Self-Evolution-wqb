from .diversity import is_redundant
from .state import AlphaRecord, score_of


class Reflector:
    def __init__(
        self,
        memory,
        good_sharpe=1.0,
        good_fitness=1.0,
        promising_sharpe=0.5,
        high_sharpe=2.5,
        high_fitness=2.0,
        max_turnover=1.5,
    ):
        self.memory = memory
        self.good_sharpe = good_sharpe
        self.good_fitness = good_fitness
        self.promising_sharpe = promising_sharpe
        self.high_sharpe = high_sharpe
        self.high_fitness = high_fitness
        self.max_turnover = max_turnover

    def reflect(self, round_no, hypothesis, experiments):
        results = []
        for exp in experiments:
            verdict = self._classify(exp)
            results.append({"experiment": exp, "verdict": verdict})
            self._learn(round_no, hypothesis, exp, verdict)

        suspicious = self._flag_suspicious(results)
        self._update_pool(results, round_no)
        best = self._update_best(results)
        self._generate_next(round_no, hypothesis, results)
        self._touch_lineages(results, round_no)

        self.memory.updated_round = round_no
        self.memory.compress()
        self.memory.save()

        return self._summary(round_no, hypothesis, results, best, suspicious)

    def _classify(self, exp):
        if exp.status == "FAILED":
            return {"label": "FAIL", "reason": self._diagnose_error(exp)}
        metrics = exp.metrics or {}
        checks = metrics.get("checks") or []
        failed_checks = [c["name"] for c in checks if not c["pass"]]
        sharpe = metrics.get("sharpe")
        fitness = metrics.get("fitness")
        turnover = metrics.get("turnover")
        if sharpe is None:
            return {"label": "FAIL", "reason": "missing sharpe metric"}

        reasons = []
        if failed_checks:
            reasons.append(f"checks failed: {failed_checks}")
        if turnover is not None and turnover > self.max_turnover:
            reasons.append(f"turnover too high: {turnover:.2f}")

        if not failed_checks and (
            sharpe >= self.high_sharpe
            or (fitness is not None and fitness >= self.high_fitness)
        ):
            return {
                "label": "SUSPICIOUS",
                "reason": f"abnormally high signal: sharpe={sharpe}, fitness={fitness}",
            }

        if (
            not failed_checks
            and sharpe >= self.good_sharpe
            and (fitness is None or fitness >= self.good_fitness)
            and turnover is not None
            and turnover <= self.max_turnover
        ):
            return {"label": "SUCCESS", "reason": " or ".join(reasons) or "Good alpha"}
        if sharpe >= self.promising_sharpe:
            return {
                "label": "PROMISING",
                "reason": " or ".join(reasons) or "weak positive signal",
            }
        return {
            "label": "FAIL",
            "reason": " or ".join(reasons) or f"sharpe too low: {sharpe:.3f}",
        }

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
                self._direction_key(exp), self._diagnose_error(exp), round_no
            )
            if self.memory.is_avoided(self._direction_key(exp)):
                self.memory.archive("repeat_fail", self._direction_key(exp), round_no)
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
        elif verdict["label"] == "SUSPICIOUS":
            self.memory.add_lesson(
                f"Fields [{field_label}] produced an unusually high signal "
                f"(Sharpe {metrics.get('sharpe')}); needs robustness validation.",
                round_no,
                evidence=1,
                confidence=0.2,
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
                f"Direction on [{field_label}] has no predictive power "
                f"(Sharpe {metrics.get('sharpe')}).",
                round_no,
                evidence=2,
                confidence=0.4,
            )

        turnover = metrics.get("turnover")
        if turnover is not None and turnover > 1.0:
            self.memory.add_lesson(
                f"Raw expression on [{field_label}] has turnover {turnover:.2f}; "
                "smoothing needed.",
                round_no,
                evidence=1,
                confidence=0.3,
            )

    def _flag_suspicious(self, results):
        suspicious = []
        for r in results:
            if r["verdict"]["label"] == "SUSPICIOUS":
                exp = r["experiment"]
                rec = self._record_from(exp, exp.round)
                rec.status = AlphaRecord.STATUS_SUSPICIOUS
                suspicious.append(rec)
        return suspicious

    def _update_pool(self, results, round_no):
        for r in results:
            if r["verdict"]["label"] != "SUCCESS":
                continue
            exp = r["experiment"]
            rec = self._record_from(exp, round_no)
            redundant, keeper = is_redundant(rec, self.memory.submission_pool)
            if redundant:
                if rec.score > score_of(keeper.get("metrics")):
                    self.memory.replace_alpha(rec, keeper["id"])
                    self.memory.archive(
                        "redundant_dup",
                        f"{rec.expression} replaced weaker duplicate {keeper['expression']}",
                        round_no,
                    )
                else:
                    self.memory.archive(
                        "redundant_dup",
                        f"{rec.expression} redundant with {keeper['expression']}",
                        round_no,
                    )
                continue
            self.memory.add_alpha(rec)

    def _record_from(self, exp, round_no):
        return AlphaRecord(
            expression=exp.expression,
            metrics=exp.metrics,
            fields_used=exp.fields_used,
            datasets=exp.datasets,
            hypothesis_id=exp.hypothesis_id,
            lineage=exp.lineage,
            round_no=round_no,
            mutation=exp.mutation,
        )

    def _touch_lineages(self, results, round_no):
        for r in results:
            label = r["verdict"]["label"]
            if label not in ("SUCCESS", "PROMISING"):
                continue
            exp = r["experiment"]
            self.memory.touch_lineage(
                exp.expression,
                exp.lineage,
                score_of(exp.metrics),
                round_no,
                fields_used=exp.fields_used,
            )

    @staticmethod
    def _direction_key(exp):
        return exp.expression[:80]

    def _update_best(self, results):
        done = [r for r in results if r["experiment"].metrics]
        if not done:
            return self.memory.current_best
        best_exp = max(done, key=lambda r: score_of(r["experiment"].metrics))["experiment"]
        best_score = score_of(best_exp.metrics)
        current_score = -1.0
        if self.memory.current_best and self.memory.current_best.get("metrics"):
            current_score = score_of(self.memory.current_best["metrics"])
        if best_score > current_score:
            self.memory.set_current_best(best_exp)
            return best_exp.to_dict()
        return self.memory.current_best

    def _generate_next(self, round_no, hypothesis, results):
        successes = [r for r in results if r["verdict"]["label"] == "SUCCESS"]
        promising = [r for r in results if r["verdict"]["label"] == "PROMISING"]
        suspicious = [r for r in results if r["verdict"]["label"] == "SUSPICIOUS"]
        failed_all = all(r["verdict"]["label"] == "FAIL" for r in results)

        if successes:
            for r in successes[:2]:
                self.memory.add_next(
                    f"Deepen successful lineage: {r['experiment'].expression}",
                    priority=5,
                    source=round_no,
                    round_no=round_no,
                )
        if promising:
            for r in promising[:2]:
                self.memory.add_next(
                    f"Improve promising expression: {r['experiment'].expression} "
                    "(smoothing, neutralization, field swap)",
                    priority=4,
                    source=round_no,
                    round_no=round_no,
                )
        if suspicious:
            for r in suspicious[:2]:
                self.memory.add_next(
                    f"Validate suspicious high signal: {r['experiment'].expression}",
                    priority=5,
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

    def _summary(self, round_no, hypothesis, results, best, suspicious):
        labels = {}
        for r in results:
            labels[r["verdict"]["label"]] = labels.get(r["verdict"]["label"], 0) + 1
        return {
            "round": round_no,
            "hypothesis": hypothesis.get("statement", ""),
            "experiment_count": len(results),
            "verdicts": labels,
            "suspicious": [s.to_dict() for s in suspicious],
            "pool_size": len(self.memory.submission_pool),
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
