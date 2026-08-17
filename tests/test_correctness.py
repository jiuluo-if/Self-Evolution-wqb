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

from wqb_agent.candidate import CandidateBuilder
from wqb_agent.memory import ExperienceMemory
from wqb_agent.reflection import Reflector
from wqb_agent.state import AlphaRecord, Experiment
from wqb_agent.validation import HighSignalValidator


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
