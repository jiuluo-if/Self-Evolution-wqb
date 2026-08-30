import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_agent import (
    BASE_CONFIG,
    FakeClient,
    _fake_metrics,
    make_agent,
)

from wqb_agent.beliefs import belief_claim, belief_identity
from wqb_agent.candidate import CandidateBuilder
from wqb_agent.memory import ExperienceMemory
from wqb_agent.reflection import Reflector
from wqb_agent.state import AlphaRecord, Experiment
from wqb_agent.validation import HighSignalValidator


K = belief_identity("h", ["returns"])


def _fresh_memory():
    tmp = tempfile.mkdtemp()
    return ExperienceMemory(state_dir=tmp)


def _done_experiment(expression, metrics, fields=None):
    e = Experiment(1, "h", expression, {}, fields or [])
    e.metrics = metrics
    e.status = "DONE"
    return e


class TestFieldsUsedPrecision(unittest.TestCase):
    def test_explore_fields_used_only_actual(self):
        builder = CandidateBuilder()
        fields = [
            {"id": "returns"},
            {"id": "volume"},
            {"id": "close"},
        ]
        cands = builder.build_explore({"direction": "reversal"}, fields, 4)
        self.assertTrue(cands)
        for c in cands:
            for fid in c["fields_used"]:
                self.assertIn(fid, c["expression"],
                              f"field {fid} not actually used in {c['expression']}")
            used = set(c["fields_used"])
            self.assertTrue(
                used <= {"returns", "volume", "close"},
                f"unknown fields recorded: {used}",
            )

    def test_deepen_fields_used_only_actual(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        alpha = {
            "expression": "rank(ts_mean(returns, 5))",
            "lineage": [],
            "attempts": 0,
            "best_score": 1.1,
            "fields_used": ["returns"],
        }
        cands = builder.build_deepen(fields, [alpha], 4)
        for c in cands:
            for fid in c["fields_used"]:
                self.assertIn(fid, c["expression"])


class TestSemanticFieldSwap(unittest.TestCase):
    def test_field_swap_prefers_same_dataset(self):
        builder = CandidateBuilder()
        fields = [
            {"id": "returns", "dataset": "pv1", "category": "price_volume"},
            {"id": "volume", "dataset": "pv1", "category": "price_volume"},
            {"id": "target_price", "dataset": "analyst4", "category": "analyst"},
        ]
        alpha = {
            "expression": "rank(returns)",
            "lineage": [],
            "attempts": 0,
            "best_score": 0.6,
            "fields_used": ["returns"],
        }
        cands = builder.build_deepen(fields, [alpha], 4)
        swaps = [c for c in cands if c["mutation"] == "field-swap"]
        self.assertTrue(swaps, "expected at least one field swap")
        swapped_expr = swaps[0]["expression"]
        self.assertIn("volume", swapped_expr)
        self.assertNotIn("target_price", swapped_expr)

    def test_field_swap_falls_back_to_known_fields(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}, {"id": "close"}]
        alpha = {
            "expression": "rank(returns)",
            "lineage": [],
            "attempts": 0,
            "best_score": 0.6,
            "fields_used": ["returns"],
        }
        cands = builder.build_deepen(fields, [alpha], 4)
        swaps = [c for c in cands if c["mutation"] == "field-swap"]
        self.assertTrue(swaps)
        self.assertNotIn("target_price", swaps[0]["expression"])


class TestBestFiltering(unittest.TestCase):
    def test_suspicious_not_best(self):
        memory = _fresh_memory()
        reflector = Reflector(memory, high_sharpe=2.0, high_fitness=2.0)
        e = _done_experiment(
            "rank(ts_mean(returns, 5))",
            {"sharpe": 3.0, "fitness": 3.0, "turnover": 0.3,
             "checks": [{"name": "x", "pass": True}], "passed": True},
            ["returns"],
        )
        summary = reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertEqual(summary["verdicts"].get("SUSPICIOUS"), 1)
        self.assertIsNone(summary["best"])
        self.assertIsNone(memory.current_best)

    def test_failed_checks_not_best(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        high_but_failing = _done_experiment(
            "rank(returns)",
            {"sharpe": 2.0, "fitness": 2.0, "turnover": 0.3,
             "checks": [{"name": "limitations", "pass": False}], "passed": False},
            ["returns"],
        )
        weak_but_clean = _done_experiment(
            "rank(close)",
            {"sharpe": 0.7, "fitness": 0.7, "turnover": 0.4,
             "checks": [{"name": "limitations", "pass": True}], "passed": True},
            ["close"],
        )
        reflector.reflect(
            1, {"tags": ["return"]},
            [high_but_failing, weak_but_clean],
        )
        self.assertEqual(memory.current_best["expression"], "rank(close)")


class TestMemoryTiering(unittest.TestCase):
    def test_single_success_never_promotes_to_long_term(self):
        memory = _fresh_memory()
        # Two successes in the SAME round = one independent experiment group.
        memory.add_lesson(
            "Combination rank(returns) works", 1, evidence=1,
            source={"experiment_id": "a", "round": 1},
        )
        memory.add_lesson(
            "Combination rank(returns) works", 1, evidence=1,
            source={"experiment_id": "b", "round": 1},
        )
        lesson = memory.lessons[0]
        self.assertEqual(lesson["evidence"], 2)
        self.assertEqual(lesson["tier"], "short")

    def test_cross_round_evidence_promotes_to_long_term(self):
        memory = _fresh_memory()
        for round_no in (1, 2, 3):
            memory.add_lesson(
                "Fields returns rank weakly", round_no, evidence=1,
                source={"experiment_id": f"e{round_no}", "round": round_no},
            )
        lesson = memory.lessons[0]
        self.assertGreaterEqual(lesson["evidence"], 3)
        self.assertEqual(lesson["tier"], "long")


class TestInfraFailuresDoNotPollute(unittest.TestCase):
    def test_network_failure_stays_out_of_research_memory(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        e = Experiment(1, "h", "rank(returns)", {}, ["returns"])
        e.status = "FAILED"
        e.error = "WQBSimulationError: network error: Connection refused"
        reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertEqual(len(memory.avoid), 0)
        self.assertEqual(len(memory.lessons), 0)
        self.assertEqual(len(memory.garbage), 0)
        # No research-level "switch direction" plan for pure infra failure.
        self.assertEqual(len(memory.next), 0)

    def test_syntax_failure_does_pollute_research_memory(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        e = Experiment(1, "h", "badfield(x)", {}, ["x"])
        e.status = "FAILED"
        e.error = "WQBRejectedError: Simulation rejected (422): invalid expression"
        reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertEqual(len(memory.avoid), 1)
        self.assertIn("syntax", memory.avoid[0]["reason"])


class TestValidationRobustness(unittest.TestCase):
    def test_lucky_single_perturbation_not_enough(self):
        validator = HighSignalValidator(None, {}, min_valid_fitness=1.0)
        record = {
            "expression": "rank(ts_mean(returns, 5))",
            "metrics": {"sharpe": 3.0, "fitness": 3.0},
            "fields_used": ["returns"],
        }
        results = [
            {"expression": "a", "score": 1.0, "checks_passed": True},
            {"expression": "b", "score": 0.1, "checks_passed": True},
            {"expression": "c", "score": 0.2, "checks_passed": True},
        ]
        stable, _ = validator.decide(record, results)
        self.assertFalse(stable)

    def test_majority_pass_with_retention_is_stable(self):
        validator = HighSignalValidator(None, {}, min_valid_fitness=1.0)
        record = {
            "expression": "rank(ts_mean(returns, 5))",
            "metrics": {"sharpe": 2.0, "fitness": 2.0},
            "fields_used": ["returns"],
        }
        results = [
            {"expression": "a", "score": 1.6, "checks_passed": True},
            {"expression": "b", "score": 1.4, "checks_passed": True},
            {"expression": "c", "score": 0.2, "checks_passed": True},
        ]
        stable, _ = validator.decide(record, results)
        self.assertTrue(stable)

    def test_failing_checks_rule_out_perturbation(self):
        validator = HighSignalValidator(None, {}, min_valid_fitness=1.0)
        record = {
            "expression": "rank(ts_mean(returns, 5))",
            "metrics": {"sharpe": 3.0, "fitness": 3.0},
            "fields_used": ["returns"],
        }
        results = [
            {"expression": "a", "score": 1.6, "checks_passed": True},
            {"expression": "b", "score": 1.4, "checks_passed": True},
            {"expression": "c", "score": 1.2, "checks_passed": False},
        ]
        stable, _ = validator.decide(record, results)
        self.assertFalse(stable)


class TestResearchInvariant(unittest.TestCase):
    """Bidirectional belief accounting: Claim -> supporting + contradicting
    evidence -> confidence. Support strengthens, contradiction weakens,
    same-lineage repeats do not inflate, infra/syntax never contradict."""

    @staticmethod
    def _source(exp_id, round_no, lineage=None, expression="rank(x)"):
        return {
            "experiment_id": exp_id,
            "round": round_no,
            "expression": expression,
            "lineage": list(lineage or []),
        }

    def _belief(self, memory):
        return memory.get_belief(K)

    def test_single_success_is_short_term(self):
        memory = _fresh_memory()
        memory.record_evidence(
            K, "Fields [returns] are predictive",
            "support", 1, source=self._source("e1", 1),
        )
        b = self._belief(memory)
        self.assertEqual(b["support_count"], 1)
        self.assertEqual(b["tier"], "short")

    def test_same_lineage_repeats_do_not_escalate(self):
        memory = _fresh_memory()
        for r in (1, 2, 3):
            memory.record_evidence(
                K, "claim", "support", r,
                source=self._source(
                    f"e{r}", r, lineage=["rootX", "mid", "parent"]
                ),
            )
        b = self._belief(memory)
        self.assertEqual(b["support_count"], 3)
        self.assertEqual(len(b["support_lineage_roots"]), 1)
        self.assertEqual(b["confidence"], 0.7)  # one line, no creep
        self.assertEqual(b["tier"], "short")

    def test_two_independent_lineages_raise_confidence(self):
        memory = _fresh_memory()
        memory.record_evidence(
            K, "claim", "support", 1,
            source=self._source("e1", 1, lineage=["rootA"]),
        )
        self.assertEqual(self._belief(memory)["confidence"], 0.7)
        memory.record_evidence(
            K, "claim", "support", 2,
            source=self._source("e2", 2, lineage=["rootB"]),
        )
        b = self._belief(memory)
        self.assertEqual(len(b["support_lineage_roots"]), 2)
        self.assertGreater(b["confidence"], 0.7)

    def test_independent_contradiction_lowers_confidence(self):
        memory = _fresh_memory()
        memory.record_evidence(
            K, "claim", "support", 1,
            source=self._source("e1", 1, lineage=["rootA"]),
        )
        before = self._belief(memory)["confidence"]
        memory.record_evidence(
            K, "claim", "contradict", 2,
            source=self._source("e2", 2, lineage=["rootB"]),
        )
        b = self._belief(memory)
        self.assertEqual(b["contradiction_count"], 1)
        self.assertLess(b["confidence"], before)

    def test_strong_contradiction_demotes_long_to_short(self):
        memory = ExperienceMemory(
            state_dir=tempfile.mkdtemp(), promote_evidence=2
        )
        for i, r in enumerate((1, 2)):
            memory.record_evidence(
                K, "claim", "support", r,
                source=self._source(f"s{i}", r, lineage=[f"rootS{i}"]),
            )
        self.assertEqual(self._belief(memory)["tier"], "long")
        for i, r in enumerate((3, 4)):
            memory.record_evidence(
                K, "claim", "contradict", r,
                source=self._source(f"c{i}", r, lineage=[f"rootC{i}"]),
            )
        b = self._belief(memory)
        self.assertEqual(len(b["contradiction_lineage_roots"]), 2)
        self.assertEqual(b["tier"], "short")  # c >= s -> demoted

    def test_same_experiment_replay_does_not_double_count(self):
        memory = _fresh_memory()
        src = self._source("e1", 1, lineage=["rootA"])
        memory.record_evidence(
            K, "claim", "support", 1, source=src,
        )
        memory.record_evidence(
            K, "claim", "support", 1, source=src,
        )
        b = self._belief(memory)
        self.assertEqual(b["support_count"], 1)
        self.assertEqual(len(b["support_lineage_roots"]), 1)

    def test_persistence_preserves_evidence_accounting(self):
        memory = _fresh_memory()
        memory.record_evidence(
            K, "claim", "support", 1,
            source=self._source("e1", 1, lineage=["rootA"]),
        )
        memory.record_evidence(
            K, "claim", "support", 2,
            source=self._source("e2", 2, lineage=["rootB"]),
        )
        memory.record_evidence(
            K, "claim", "contradict", 3,
            source=self._source("e3", 3, lineage=["rootC"]),
        )
        memory.save()
        loaded = ExperienceMemory(state_dir=memory.state_dir).load()
        b = loaded.get_belief(K)
        self.assertEqual(b["support_count"], 2)
        self.assertEqual(b["contradiction_count"], 1)
        self.assertEqual(b["support_lineage_roots"], {"rootA", "rootB"})
        self.assertEqual(b["contradiction_lineage_roots"], {"rootC"})
        self.assertEqual(len(b["source_rounds"]), 3)
        self.assertEqual(b["confidence"], 0.7)

    def test_infra_failure_changes_no_belief(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        e = Experiment(1, "h", "rank(returns)", {}, ["returns"])
        e.status = "FAILED"
        e.error = "WQBSimulationError: network error: Connection refused"
        reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertEqual(len(memory.beliefs), 0)

    def test_syntax_failure_is_not_contradiction(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        e = Experiment(1, "h", "badfield(x)", {}, ["x"])
        e.status = "FAILED"
        e.error = "WQBRejectedError: Simulation rejected (422): invalid expression"
        reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertEqual(len(memory.beliefs), 0)
        self.assertEqual(len(memory.avoid), 1)  # construction lesson kept

    def test_reflection_records_support_and_contradiction_on_same_belief(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        ok = _done_experiment(
            "rank(ts_mean(returns, 5))",
            {"sharpe": 1.2, "fitness": 1.2, "turnover": 0.3,
             "checks": [{"name": "x", "pass": True}], "passed": True},
            ["returns"],
        )
        reflector.reflect(1, {"tags": ["return"]}, [ok])
        bad = _done_experiment(
            "rank(close)",
            {"sharpe": 0.1, "fitness": 0.1, "turnover": 1.8,
             "checks": [{"name": "x", "pass": True}], "passed": True},
            ["returns"],
        )
        reflector.reflect(1, {"tags": ["return"]}, [bad])
        b = memory.get_belief(K)
        self.assertIsNotNone(b)
        self.assertEqual(b["support_count"], 1)
        self.assertEqual(b["contradiction_count"], 1)
        self.assertEqual(b["confidence"], 0.5)

    def test_validation_failure_contradicts_high_signal_claim(self):
        memory = _fresh_memory()
        tmpdir = tempfile.mkdtemp()
        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = tmpdir
        agent = AgentForTest(config, FakeClient())
        rec = AlphaRecord(
            expression="rank(ts_mean(returns, 5))",
            metrics={"sharpe": 3.0, "fitness": 3.0},
            fields_used=["returns"],
            lineage=["rank(returns)"],
            round_no=1,
            hypothesis_id="h",
        ).to_dict()
        agent._apply_validation(rec, stable=False, round_no=1, sims=3)
        b = agent.memory.get_belief(K)
        self.assertEqual(b["contradiction_count"], 1)
        self.assertEqual(b["support_count"], 0)
        self.assertEqual(b["evidence_log"][0]["kind"], "validation_failure")

    def test_validation_success_supports_high_signal_claim(self):
        memory = _fresh_memory()
        tmpdir = tempfile.mkdtemp()
        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = tmpdir
        agent = AgentForTest(config, FakeClient())
        rec = AlphaRecord(
            expression="rank(ts_mean(returns, 5))",
            metrics={"sharpe": 3.0, "fitness": 3.0},
            fields_used=["returns"],
            lineage=["rank(returns)"],
            round_no=1,
            hypothesis_id="h",
        ).to_dict()
        agent._apply_validation(rec, stable=True, round_no=1, sims=3)
        b = agent.memory.get_belief(K)
        self.assertEqual(b["support_count"], 1)
        self.assertEqual(b["evidence_log"][0]["kind"], "validated_high_signal")


class TestBeliefIdentity(unittest.TestCase):
    """A belief is one falsifiable research proposition, never a field family.
    Identity = hypothesis_id + normalized fields + direction, built by the
    single canonical helper shared by Reflection and Validation."""

    REV = {
        "id": "h-seed-reversal",
        "statement": "Short-term return reversal over 5 days.",
        "tags": ["reversal", "return", "price", "short-term"],
        "direction": "reversal",
        "datasets": ["pv1", "pv13"],
    }
    ANALYST = {
        "id": "h-seed-analyst",
        "statement": "Analyst revisions upward predict outperformance.",
        "tags": ["analyst", "forecast", "revision", "target"],
        "direction": "long",
        "datasets": ["analyst4"],
    }

    @staticmethod
    def _source(exp_id, round_no, expression="rank(x)"):
        return {
            "experiment_id": exp_id,
            "round": round_no,
            "expression": expression,
        }

    def test_same_hypothesis_fields_direction_aggregate(self):
        memory = _fresh_memory()
        k = belief_identity("h-seed-reversal", ["returns"],
                            hypothesis=self.REV)
        claim = belief_claim("h-seed-reversal", ["returns"],
                             hypothesis=self.REV)
        memory.record_evidence(k, claim, "support", 1,
                               source=self._source("e1", 1))
        memory.record_evidence(k, claim, "support", 2,
                               source=self._source("e2", 2))
        self.assertEqual(len(memory.beliefs), 1)
        self.assertEqual(memory.get_belief(k)["support_count"], 2)
        self.assertEqual(memory.get_belief(k)["claim"], claim)

    def test_different_hypotheses_stay_separate(self):
        memory = _fresh_memory()
        k1 = belief_identity("h-seed-reversal", ["returns"],
                             hypothesis=self.REV)
        k2 = belief_identity("h-seed-analyst", ["returns"],
                             hypothesis=self.ANALYST)
        self.assertNotEqual(k1, k2)
        memory.record_evidence(k1, "c1", "support", 1,
                               source=self._source("e1", 1))
        memory.record_evidence(k2, "c2", "support", 1,
                               source=self._source("e2", 1))
        self.assertEqual(len(memory.beliefs), 2)

    def test_opposite_directions_stay_separate(self):
        rev = belief_identity("h-seed-reversal", ["returns"],
                              direction="reversal")
        mom = belief_identity("h-seed-reversal", ["returns"],
                              direction="long")
        self.assertNotEqual(rev, mom)

    def test_parameter_mutations_share_a_belief(self):
        memory = _fresh_memory()
        k = belief_identity("h-seed-reversal", ["returns"],
                            hypothesis=self.REV)
        for i, expr in enumerate(
            (
                "rank(ts_mean(returns, 5))",
                "rank(ts_mean(returns, 21))",
                "-rank(returns)",
            )
        ):
            memory.record_evidence(
                k, "c", "support", 1,
                source=self._source(f"e-{i}-{expr[:12]}", 1,
                                    expression=expr),
            )
        self.assertEqual(len(memory.beliefs), 1)
        self.assertEqual(memory.get_belief(k)["support_count"], 3)

    def test_validation_and_suspicious_share_belief(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        exp = _done_experiment(
            "-rank(returns)",
            {"sharpe": 3.0, "fitness": 3.0, "turnover": 0.3,
             "checks": [{"name": "x", "pass": True}], "passed": True},
            ["returns"],
        )
        exp.hypothesis_id = "h-seed-reversal"
        reflector.reflect(1, dict(self.REV), [exp])
        self.assertEqual(len(memory.beliefs), 1)
        belief = memory.beliefs[0]
        expected = belief_identity("h-seed-reversal", ["returns"],
                                   hypothesis=self.REV)
        self.assertEqual(belief["belief_key"], expected)
        self.assertEqual(belief["evidence_log"][0]["kind"],
                         "suspicious_high_signal")

        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = tempfile.mkdtemp()
        agent = AgentForTest(config, FakeClient())
        agent.memory = memory
        rec = AlphaRecord(
            expression="-rank(returns)",
            metrics={"sharpe": 3.0, "fitness": 3.0},
            fields_used=["returns"],
            lineage=["-rank(returns)"],
            round_no=1,
            hypothesis_id="h-seed-reversal",
        ).to_dict()
        agent._apply_validation(rec, stable=True, round_no=2, sims=3)
        self.assertEqual(len(memory.beliefs), 1)  # still ONE belief
        self.assertEqual(memory.beliefs[0]["support_count"], 1)

    def test_one_hypothesis_failure_does_not_affect_another(self):
        memory = _fresh_memory()
        k1 = belief_identity("h-seed-reversal", ["returns"],
                             hypothesis=self.REV)
        k2 = belief_identity("h-seed-analyst", ["returns"],
                             hypothesis=self.ANALYST)
        memory.record_evidence(k1, "c1", "support", 1,
                               source=self._source("e1", 1))
        memory.record_evidence(k2, "c2", "support", 1,
                               source=self._source("e2", 1))
        c1 = memory.get_belief(k1)["confidence"]
        c2 = memory.get_belief(k2)["confidence"]
        memory.record_evidence(k2, "c3", "contradict", 2,
                               source=self._source("e3", 2))
        self.assertEqual(memory.get_belief(k1)["confidence"], c1)
        self.assertLess(memory.get_belief(k2)["confidence"], c2)

    def test_reflection_replay_does_not_double_count(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        exp = _done_experiment(
            "rank(ts_mean(returns, 5))",
            {"sharpe": 1.2, "fitness": 1.2, "turnover": 0.3,
             "checks": [{"name": "x", "pass": True}], "passed": True},
            ["returns"],
        )
        exp.hypothesis_id = "h-seed-reversal"
        reflector.reflect(1, dict(self.REV), [exp])
        reflector.reflect(2, dict(self.REV), [exp])  # resume replay
        b = memory.get_belief(
            belief_identity("h-seed-reversal", ["returns"],
                            hypothesis=self.REV)
        )
        self.assertEqual(b["support_count"], 1)
        self.assertEqual(b["confidence"], 0.7)

    def test_belief_identity_survives_save_load(self):
        memory = _fresh_memory()
        k = belief_identity("h-seed-reversal", ["returns", "volume"],
                            hypothesis=self.REV)
        memory.record_evidence(k, "c", "support", 1,
                               source=self._source("e1", 1))
        memory.save()
        loaded = ExperienceMemory(state_dir=memory.state_dir).load()
        self.assertEqual(loaded.get_belief(k)["belief_key"], k)
        # field order must not create a second belief
        k2 = belief_identity("h-seed-reversal", ["volume", "returns"],
                             hypothesis=self.REV)
        self.assertEqual(k, k2)

    def test_replay_idempotency_survives_log_truncation(self):
        memory = _fresh_memory()
        k = belief_identity("h-seed-reversal", ["returns"],
                            hypothesis=self.REV)
        # Fill the evidence_log past its 30-entry bound so the first entry is
        # evicted; a replay of that first experiment must still be skipped.
        for i in range(35):
            memory.record_evidence(
                k, "c", "support", 1,
                source=self._source(f"e{i}", 1),
            )
        self.assertEqual(len(memory.get_belief(k)["evidence_log"]), 30)
        memory.record_evidence(
            k, "c", "support", 1,
            source=self._source("e0", 1),
        )
        b = memory.get_belief(k)
        self.assertEqual(b["support_count"], 35)

    def test_replay_idempotency_survives_save_load(self):
        memory = _fresh_memory()
        k = belief_identity("h-seed-reversal", ["returns"],
                            hypothesis=self.REV)
        memory.record_evidence(k, "c", "support", 1,
                               source=self._source("e1", 1))
        memory.save()
        loaded = ExperienceMemory(state_dir=memory.state_dir).load()
        loaded.record_evidence(k, "c", "support", 1,
                               source=self._source("e1", 1))
        self.assertEqual(loaded.get_belief(k)["support_count"], 1)

    def test_add_avoid_keeps_history(self):
        memory = _fresh_memory()
        memory.add_avoid("d1", "r1", 1,
                         source={"experiment_id": "e1", "expression": "a"})
        memory.add_avoid("d1", "r2", 2,
                         source={"experiment_id": "e2", "expression": "b"})
        self.assertEqual(len(memory.avoid), 1)
        item = memory.avoid[0]
        self.assertEqual(item["support_count"], 2)
        self.assertEqual(item["last_seen"], 2)
        self.assertEqual(len(item["evidence_log"]), 2)
        self.assertEqual(item["evidence_log"][0]["experiment_id"], "e2")
        self.assertEqual(item["evidence_log"][1]["experiment_id"], "e1")


class TestMemoryMethodUniqueness(unittest.TestCase):
    """A duplicate method definition silently shadows the earlier one. Key
    memory behaviors (e.g. add_avoid) must never be redefined by accident."""

    def test_no_duplicate_method_definitions_in_memory(self):
        import ast

        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo, "wqb_agent", "memory.py")
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            names = [
                n.name
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            dups = {n for n in names if names.count(n) > 1}
            self.assertEqual(
                dups, set(),
                f"duplicate method definitions in {node.name}: {dups}",
            )


class TestNoChecksUnknown(unittest.TestCase):
    def test_empty_checks_never_counts_as_success(self):
        memory = _fresh_memory()
        reflector = Reflector(memory)
        e = _done_experiment(
            "rank(returns)",
            {"sharpe": 2.0, "fitness": 2.0, "turnover": 0.3,
             "checks": [], "passed": None},
            ["returns"],
        )
        summary = reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertNotEqual(summary["verdicts"].get("SUCCESS"), 1)
        self.assertIsNone(summary["best"])
        self.assertEqual(len(memory.submission_pool), 0)


class TestValidationBudget(unittest.TestCase):
    def test_validation_never_exceeds_round_budget(self):
        class AllHighClient(FakeClient):
            def get_alpha(self, alpha_id):
                expression = self._alpha_expr.get(alpha_id, "")
                metrics = dict(_fake_metrics(expression))
                metrics.update({
                    "sharpe": 3.0,
                    "fitness": 3.0,
                    "turnover": 0.3,
                    "checks": [{"name": "limitations", "pass": True}],
                })
                return {"is": metrics, "regular": expression}

        tmpdir = tempfile.mkdtemp()
        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = tmpdir
        config["agent"]["max_rounds"] = 1
        config["agent"]["validation_budget_per_round"] = 2
        client = AllHighClient()
        agent = AgentForTest(config, client)
        agent.run_one_round(1)

        validation_sims = agent.scheduler_stats["validation"]
        self.assertLessEqual(validation_sims, 2)


from wqb_agent.agent import Agent


class AgentForTest(Agent):
    def __init__(self, config, client):
        super().__init__(client, config)
        self.scheduler_stats = {"validation": 0}

    def _new_scheduler(self, round_no):
        from wqb_agent.scheduler import BacktestScheduler

        path = os.path.join(self.state_dir, f"round_{round_no}_jobs.json")
        return BacktestScheduler(
            self.client,
            self.simulator,
            max_concurrent=self.max_concurrent_sims,
            budget=self.sim_budget_per_round + self.validation_budget,
            poll_timeout_sec=self.poll_timeout_sec,
            checkpoint_path=path,
            checkpoint_every=1,
        )

    def _validate_suspicious(self, scheduler, summary, round_no):
        used = super()._validate_suspicious(scheduler, summary, round_no)
        self.scheduler_stats["validation"] = used
        return used


if __name__ == "__main__":
    unittest.main()
