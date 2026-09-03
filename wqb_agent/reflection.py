from .beliefs import belief_claim, belief_identity
from .diversity import is_redundant
from .failures import FailureKind, classify_error, is_research_relevant
from .state import AlphaRecord, score_of


def _passed(metrics):
    """Tri-state pass result: True / False / None (UNKNOWN, no checks)."""
    checks = metrics.get("checks") or []
    if metrics.get("passed") is not None:
        return metrics["passed"]
    if not checks:
        return None
    return all(bool(c.get("pass")) for c in checks)


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
            kind = classify_error(exp.error)
            return {
                "label": "FAIL",
                "kind": kind,
                "reason": self._diagnose_error(exp),
            }
        if exp.status == "SUBMIT_UNKNOWN":
            # The submit may or may not have been accepted by the backend; the
            # outcome carries no research signal and must never enter research
            # memory as support or contradiction.
            return {
                "label": "FAIL",
                "kind": FailureKind.INFRA,
                "reason": "submit outcome unknown (may have been accepted)",
            }
        metrics = exp.metrics or {}
        passed = _passed(metrics)
        failed_checks = [
            c["name"] for c in (metrics.get("checks") or []) if not c["pass"]
        ]
        sharpe = metrics.get("sharpe")
        fitness = metrics.get("fitness")
        turnover = metrics.get("turnover")
        if sharpe is None:
            return {
                "label": "FAIL",
                "kind": "RESEARCH",
                "reason": "missing sharpe metric",
            }

        reasons = []
        if passed is None:
            reasons.append("no checks returned (UNKNOWN)")
        elif not passed:
            reasons.append(f"checks failed: {failed_checks}")
        if turnover is not None and turnover > self.max_turnover:
            reasons.append(f"turnover too high: {turnover:.2f}")

        if passed is True and (
            sharpe >= self.high_sharpe
            or (fitness is not None and fitness >= self.high_fitness)
        ):
            return {
                "label": "SUSPICIOUS",
                "kind": "RESEARCH",
                "reason": f"abnormally high signal: sharpe={sharpe}, fitness={fitness}",
            }

        if (
            passed is True
            and sharpe >= self.good_sharpe
            and (fitness is None or fitness >= self.good_fitness)
            and turnover is not None
            and turnover <= self.max_turnover
        ):
            return {"label": "SUCCESS", "kind": "RESEARCH",
                    "reason": " or ".join(reasons) or "Good alpha"}
        if sharpe >= self.promising_sharpe:
            return {
                "label": "PROMISING",
                "kind": "RESEARCH",
                "reason": " or ".join(reasons) or "weak positive signal",
            }
        return {
            "label": "FAIL",
            "kind": "RESEARCH",
            "reason": " or ".join(reasons) or f"sharpe too low: {sharpe:.3f}",
        }

    def _diagnose_error(self, exp):
        error = exp.error or ""
        if "Simulation rejected" in error or "422" in error or "400" in error:
            return f"syntax/settings rejection: {error[:120]}"
        if "timed out" in error.lower():
            return "polling timed out"
        return f"runtime error: {error[:120]}"

    def _source_of(self, exp):
        metrics = exp.metrics or {}
        return {
            "experiment_id": exp.id,
            "round": exp.round,
            "hypothesis_id": exp.hypothesis_id,
            "expression": exp.expression,
            "lineage": list(exp.lineage),
            "mutation": exp.mutation,
            "fields": list(exp.fields_used),
            "metrics": {
                "sharpe": metrics.get("sharpe"),
                "fitness": metrics.get("fitness"),
                "turnover": metrics.get("turnover"),
            }
            if exp.metrics
            else None,
        }

    def _learn(self, round_no, hypothesis, exp, verdict):
        # The belief identity/claim is shared by every evidence-recording
        # branch below; compute it once instead of re-deriving the field
        # normalization per branch.
        bkey = belief_identity(exp.hypothesis_id, exp.fields_used,
                               hypothesis=hypothesis)
        bclaim = belief_claim(exp.hypothesis_id, exp.fields_used,
                              hypothesis=hypothesis)
        source = self._source_of(exp)
        if exp.status in ("FAILED", "SUBMIT_UNKNOWN"):
            kind = verdict.get("kind", "RESEARCH")
            if not is_research_relevant(kind):
                # Infrastructure / auth / rate-limit / timeout failures carry no
                # information about the hypothesis; keep them out of research
                # memory and garbage. They never count as support or
                # contradiction for any belief.
                return
            direction = self._direction_key(exp)
            # Every research-relevant failure teaches what not to try again.
            # The avoid entry is updated (or created) and the direction is
            # archived so the failure is never silently forgotten.
            self.memory.add_avoid(
                direction,
                self._diagnose_error(exp),
                round_no,
                source=source,
            )
            self.memory.archive("repeat_fail", direction, round_no)
            if kind in (FailureKind.SYNTAX, FailureKind.DATA):
                # Expression construction / data failures teach how to build
                # the expression, never whether the economic hypothesis is
                # wrong. They stay in avoid as construction lessons and do not
                # enter belief accounting.
                return
            # A valid RESEARCH failure (low sharpe / failed checks with metrics)
            # is financial negative evidence for the belief on these fields.
            self.memory.record_evidence(
                bkey,
                bclaim,
                "contradict",
                round_no,
                source=source,
                kind="financial_negative",
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
                evidence=1,
                confidence=0.7,
                source=source,
            )
            self.memory.record_evidence(
                bkey,
                bclaim,
                "support",
                round_no,
                source=source,
                kind="success",
            )
        elif verdict["label"] == "SUSPICIOUS":
            self.memory.add_lesson(
                f"Fields [{field_label}] produced an unusually high signal "
                f"(Sharpe {metrics.get('sharpe')}); needs robustness validation.",
                round_no,
                evidence=1,
                confidence=0.2,
                source=source,
            )
            # Pending: a high-signal hit is not yet strong support. It only
            # counts once robustness validation confirms it (or becomes a
            # contradiction when validation rejects it).
            self.memory.record_evidence(
                bkey,
                bclaim,
                "pending",
                round_no,
                source=source,
                kind="suspicious_high_signal",
            )
        elif verdict["label"] == "PROMISING":
            self.memory.add_lesson(
                f"Fields [{field_label}] show weak positive signal; worth iterating.",
                round_no,
                evidence=1,
                confidence=0.3,
                source=source,
            )
            self.memory.record_evidence(
                bkey,
                bclaim,
                "support",
                round_no,
                source=source,
                kind="promising",
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
                source=source,
            )
            # Financial negative evidence: a valid research result opposing the
            # belief on these fields. Not a lesson to be text-merged with
            # successes (token similarity cannot decide polarity).
            self.memory.record_evidence(
                bkey,
                bclaim,
                "contradict",
                round_no,
                source=source,
                kind="financial_negative",
            )

        turnover = metrics.get("turnover")
        if turnover is not None and turnover > 1.0:
            self.memory.add_lesson(
                f"Raw expression on [{field_label}] has turnover {turnover:.2f}; "
                "smoothing needed.",
                round_no,
                evidence=1,
                confidence=0.3,
                source=source,
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
                hypothesis_id=exp.hypothesis_id,
            )

    @staticmethod
    def _direction_key(exp):
        return exp.expression[:80]

    def _update_best(self, results):
        """Best may only come from experiments that fully passed WQB checks and
        are not suspicious high-signal alphas waiting for validation."""
        eligible = []
        for r in results:
            exp = r["experiment"]
            if exp.metrics is None:
                continue
            if r["verdict"]["label"] in ("FAIL", "SUSPICIOUS"):
                continue
            if _passed(exp.metrics) is not True:
                continue
            eligible.append(exp)
        if not eligible:
            return self.memory.current_best
        best_exp = max(eligible, key=lambda e: score_of(e.metrics))
        best_score = score_of(best_exp.metrics)
        current = self.memory.current_best
        current_score = score_of((current or {}).get("metrics")) if current else -1.0
        if best_score > current_score:
            self.memory.set_current_best(best_exp)
            return best_exp.to_dict()
        return current

    def _generate_next(self, round_no, hypothesis, results):
        successes = [r for r in results if r["verdict"]["label"] == "SUCCESS"]
        promising = [r for r in results if r["verdict"]["label"] == "PROMISING"]
        suspicious = [r for r in results if r["verdict"]["label"] == "SUSPICIOUS"]
        fails = [r for r in results if r["verdict"]["label"] == "FAIL"]
        # A "failed round" only steers the research direction when the failures
        # are research-level; all-infra failures carry no research signal.
        research_fails = [
            r for r in fails
            if is_research_relevant(r["verdict"].get("kind", "RESEARCH"))
        ]
        failed_all = bool(fails) and len(research_fails) == len(fails)

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
