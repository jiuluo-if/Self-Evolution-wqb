import json
import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wqb_agent.agent import Agent, SEED_HYPOTHESES
from wqb_agent.candidate import CandidateBuilder
from wqb_agent.discovery import FieldDiscovery
from wqb_agent.memory import ExperienceMemory
from wqb_agent.reflection import Reflector
from wqb_agent.simulator import Simulator
from wqb_agent.state import Experiment

FAKE_FIELDS = {
    "pv1": [
        {"id": "close", "name": "Close price", "description": "daily close price of the stock"},
        {"id": "returns", "name": "Returns", "description": "daily simple returns"},
        {"id": "volume", "name": "Volume", "description": "daily trading volume"},
        {"id": "adv20", "name": "Average daily volume 20d", "description": "20 day average trading volume"},
        {"id": "high", "name": "High price", "description": "daily high price"},
        {"id": "low", "name": "Low price", "description": "daily low price"},
        {"id": "open", "name": "Open price", "description": "daily open price"},
        {"id": "vwap", "name": "VWAP", "description": "volume weighted average price"},
    ],
    "pv13": [
        {"id": "sector", "name": "Sector", "description": "sector classification"},
        {"id": "market_cap", "name": "Market cap", "description": "market capitalization"},
        {"id": "spread", "name": "Bid-ask spread", "description": "liquidity spread"},
    ],
    "analyst4": [
        {"id": "target_price", "name": "Analyst target price", "description": "consensus analyst target price"},
        {"id": "recommendation", "name": "Recommendation", "description": "analyst recommendation rating"},
        {"id": "eps_estimate", "name": "EPS estimate", "description": "analyst eps estimate"},
        {"id": "num_analysts", "name": "Number of analysts", "description": "analyst coverage count"},
    ],
    "option8": [
        {"id": "implied_vol", "name": "Implied volatility", "description": "option implied volatility"},
        {"id": "put_call_ratio", "name": "Put call ratio", "description": "option put call volume ratio"},
        {"id": "iv_skew", "name": "IV skew", "description": "implied volatility skew"},
        {"id": "option_volume", "name": "Option volume", "description": "total option trading volume"},
    ],
    "model16": [
        {"id": "risk_score", "name": "Model risk score", "description": "composite model risk score"},
        {"id": "model_factor", "name": "Model factor", "description": "model factor loading"},
        {"id": "pred_ret", "name": "Predicted return", "description": "model predicted return"},
    ],
    "news12": [
        {"id": "news_sentiment", "name": "News sentiment", "description": "news sentiment score"},
        {"id": "news_count", "name": "News count", "description": "number of news articles"},
        {"id": "headline_buzz", "name": "Headline buzz", "description": "headline attention score"},
    ],
}


class FakeClient:
    def __init__(self, latency=0.01):
        self.counter = 0
        self.latency = latency
        self._expr_by_url = {}
        self._alpha_expr = {}
        self.sim_calls = []
        self.max_active = 0
        self._active = 0
        self._lock = threading.Lock()
        self.datafield_calls = []

    def get_datafields(self, dataset_id, limit=50, offset=0):
        self.datafield_calls.append((dataset_id, limit, offset))
        all_fields = FAKE_FIELDS.get(dataset_id, [])
        return all_fields[offset : offset + limit], len(all_fields)

    def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
        with self._lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)
        time.sleep(self.latency)
        with self._lock:
            self._active -= 1
        self.counter += 1
        url = f"progress-{self.counter}"
        self._expr_by_url[url] = expression
        self.sim_calls.append(expression)
        return url

    def poll_progress(self, progress_url, timeout_sec=900):
        time.sleep(self.latency)
        expression = self._expr_by_url[progress_url]
        alpha_id = f"alpha-{abs(hash(progress_url))}"
        self._alpha_expr[alpha_id] = expression
        return alpha_id

    def get_alpha(self, alpha_id):
        expression = self._alpha_expr.get(alpha_id, "")
        return {"is": _fake_metrics(expression), "regular": expression}


def _fake_metrics(expression):
    sharpe = 0.1
    turnover = 1.8
    if "ts_mean" in expression:
        sharpe += 1.0
        turnover = 0.4
    if "ts_rank" in expression:
        sharpe += 0.3
    if "group_neutralize" in expression:
        sharpe += 0.2
    if "zscore" in expression:
        sharpe -= 0.1
    if expression.startswith("rank(close") or expression.startswith("-rank(close"):
        sharpe = -0.3
    if "badfield" in expression:
        return {
            "sharpe": 0.0,
            "fitness": 0.0,
            "turnover": 0.5,
            "margin": 0.0,
            "returns": 0.0,
            "checks": [{"name": "syntax", "pass": False}],
        }
    return {
        "sharpe": sharpe,
        "fitness": sharpe,
        "turnover": turnover,
        "margin": sharpe * 0.1,
        "returns": sharpe * 0.05,
        "checks": [{"name": "limitations", "pass": True}],
    }


BASE_CONFIG = {
    "simulation": {
        "instrumentType": "EQUITY",
        "region": "USA",
        "universe": "TOP3000",
        "delay": 1,
        "decay": 0,
        "neutralization": "SUBINDUSTRY",
        "truncation": 0.08,
        "pasteurization": "ON",
        "unitHandling": "VERIFY",
        "nanHandling": "ON",
    },
    "agent": {
        "max_rounds": 2,
        "candidates_per_round": 6,
        "max_concurrent_sims": 3,
        "fields_per_discovery": 6,
        "pagination_limit": 50,
        "max_pagination_pages": 20,
        "state_dir": None,
        "poll_timeout_sec": 30,
    },
}


def make_agent(tmpdir, rounds=2):
    config = json.loads(json.dumps(BASE_CONFIG))
    config["agent"]["state_dir"] = str(tmpdir)
    config["agent"]["max_rounds"] = rounds
    client = FakeClient()
    return Agent(client, config), client


class TestFieldDiscovery(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self.discovery = FieldDiscovery(self.client, pagination_limit=2, max_pages=20)

    def test_selects_relevant_dataset_and_fields(self):
        hypothesis = {
            "statement": "Stocks with high trading volume predict returns.",
            "tags": ["volume", "return"],
            "direction": "long",
        }
        fields = self.discovery.discover(hypothesis, target_count=4)
        ids = [f["id"] for f in fields]
        self.assertTrue(any("volume" in f for f in ids))
        self.assertTrue(any("return" in f for f in ids))
        self.assertLessEqual(len(fields), 4)
        calls = self.client.datafield_calls
        self.assertTrue(all(c[1] == 2 for c in calls))

    def test_stops_when_enough_fields(self):
        hypothesis = {
            "statement": "reversal on price.",
            "tags": ["reversal", "price"],
            "direction": "reversal",
        }
        fields = self.discovery.discover(hypothesis, target_count=2)
        self.assertLessEqual(len(fields), 2)

    def test_no_fake_fields(self):
        hypothesis = {
            "statement": "option volatility.",
            "tags": ["option", "volatility"],
            "direction": "reversal",
        }
        fields = self.discovery.discover(hypothesis, target_count=6)
        known = {f["id"] for f in sum(FAKE_FIELDS.values(), [])}
        self.assertTrue(all(f["id"] in known for f in fields))


class TestCandidateBuilder(unittest.TestCase):
    def test_explore_from_scratch(self):
        builder = CandidateBuilder(neutralization="subindustry")
        fields = [{"id": "returns"}, {"id": "volume"}]
        hypothesis = {"direction": "reversal", "tags": ["return"]}
        candidates = builder.build(hypothesis, fields, None, count=6)
        self.assertEqual(len(candidates), 6)
        explore = [c for c in candidates if c["mutation"] == "explore"]
        deepen = [c for c in candidates if c["mutation"] != "explore"]
        self.assertEqual(len(explore), 6)
        self.assertEqual(len(deepen), 0)
        self.assertTrue(all(c["parent"] is None for c in candidates))

    def test_build_pools_splits_explore_and_deepen(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        best = {
            "id": "b1",
            "expression": "rank(returns)",
            "fields_used": ["returns"],
            "lineage": [],
            "attempts": 0,
            "metrics": {"sharpe": 0.6, "fitness": 0.6},
        }
        candidates = builder.build_pools(
            {}, fields, [best], total=6, explore_ratio=0.5
        )
        explore = [c for c in candidates if c["mutation"] == "explore"]
        deepen = [c for c in candidates if c["mutation"] != "explore"]
        self.assertEqual(len(candidates), 6)
        self.assertGreater(len(explore), 0)
        self.assertGreater(len(deepen), 0)
        self.assertTrue(all(c["parent"] == "rank(returns)" for c in deepen))

    def test_reversal_flips_sign(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}]
        candidates = builder.build_explore({"direction": "reversal"}, fields, 1)
        self.assertTrue(candidates[0]["expression"].startswith("-rank"))

    def test_deepen_respects_max_attempts(self):
        builder = CandidateBuilder(max_deepen_per_lineage=2)
        fields = [{"id": "returns"}, {"id": "volume"}]
        alpha = {
            "expression": "rank(returns)",
            "lineage": [],
            "attempts": 2,
            "best_score": 0.6,
        }
        self.assertEqual(builder.build_deepen(fields, [alpha], 4), [])

    def test_deepen_lineage_recorded(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        alpha = {
            "expression": "rank(ts_mean(returns, 5))",
            "lineage": ["rank(returns)"],
            "attempts": 1,
            "best_score": 1.1,
        }
        cands = builder.build_deepen(fields, [alpha], 3)
        self.assertGreater(len(cands), 0)
        for c in cands:
            self.assertEqual(c["lineage"][0], "rank(ts_mean(returns, 5))")
            self.assertEqual(c["lineage"][1], "rank(returns)")


class TestSimulator(unittest.TestCase):
    def test_concurrency_limited(self):
        client = FakeClient(latency=0.05)
        sim = Simulator(client, max_concurrent=3, poll_timeout_sec=30)
        exps = [Experiment(1, "h", f"rank(field{i})", {}, []) for i in range(7)]
        sim.run(exps)
        self.assertEqual(len(exps), 7)
        self.assertTrue(all(e.status == "DONE" for e in exps))
        self.assertLessEqual(client.max_active, 3)
        self.assertEqual(len(client.sim_calls), 7)

    def test_failure_recorded(self):
        class FailingClient(FakeClient):
            def submit_simulation(self, expression, settings, alpha_type="REGULAR"):
                raise Exception("Simulation rejected (422): bad expression")

        sim = Simulator(FailingClient(), max_concurrent=3, poll_timeout_sec=5)
        exp = Experiment(1, "h", "badfield(x)", {}, [])
        sim.run([exp])
        self.assertEqual(exp.status, "FAILED")
        self.assertIn("Simulation rejected", exp.error)


class TestReflection(unittest.TestCase):
    def test_updates_best_and_lessons(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_mem")
        reflector = Reflector(memory)
        exps = []
        for i, expr in enumerate(
            ["rank(close)", "rank(ts_mean(returns, 5))", "rank(ts_rank(returns, 20))"]
        ):
            e = Experiment(1, "h", expr, {}, ["returns"])
            e.metrics = _fake_metrics(expr)
            e.status = "DONE"
            exps.append(e)
        summary = reflector.reflect(1, {"tags": ["return"], "direction": "long"}, exps)
        self.assertIsNotNone(memory.current_best)
        self.assertEqual(memory.current_best["expression"], "rank(ts_mean(returns, 5))")
        self.assertGreater(len(memory.lessons), 0)
        self.assertIn("SUCCESS", summary["verdicts"])
        self.assertGreaterEqual(len(memory.submission_pool), 1)

    def test_fail_diagnosis_adds_avoid(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_mem2")
        reflector = Reflector(memory)
        e = Experiment(1, "h", "rank(close)", {}, ["close"])
        e.status = "FAILED"
        e.error = "Simulation rejected (422): syntax error"
        reflector.reflect(1, {"tags": ["price"], "direction": "long"}, [e])
        self.assertEqual(len(memory.avoid), 1)
        self.assertIn("syntax", memory.avoid[0]["reason"])

    def test_suspicious_high_signal_flagged(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_susp")
        reflector = Reflector(memory, high_sharpe=2.0, high_fitness=2.0)
        e = Experiment(1, "h", "rank(ts_mean(returns, 5))", {}, ["returns"])
        e.metrics = {
            "sharpe": 3.0,
            "fitness": 3.0,
            "turnover": 0.3,
            "checks": [{"name": "limitations", "pass": True}],
        }
        e.status = "DONE"
        summary = reflector.reflect(1, {"tags": ["return"]}, [e])
        self.assertEqual(summary["verdicts"].get("SUSPICIOUS"), 1)
        self.assertEqual(len(summary["suspicious"]), 1)
        self.assertEqual(
            summary["suspicious"][0]["status"], "SUSPICIOUS_HIGH_SIGNAL"
        )

    def test_pool_rejects_redundant(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_pool")
        reflector = Reflector(memory)
        exps = []
        for expr in [
            "rank(ts_mean(returns, 5))",
            "rank(ts_mean(volume, 5))",
            "rank(ts_mean(close, 5))",
        ]:
            e = Experiment(1, "h", expr, {}, ["returns", "volume", "close"])
            e.metrics = _fake_metrics(expr)
            e.status = "DONE"
            exps.append(e)
        reflector.reflect(1, {"tags": ["return"], "direction": "long"}, exps)
        self.assertEqual(len(memory.submission_pool), 1)


class TestMemory(unittest.TestCase):
    def test_persistence_roundtrip(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_mem3")
        memory.add_lesson("Smoothing lowers turnover", 1, evidence=2)
        memory.add_avoid("rank(close)", "low sharpe", 1)
        memory.save()
        loaded = ExperienceMemory(state_dir="/tmp/wqb_test_mem3").load()
        self.assertEqual(len(loaded.lessons), 1)
        self.assertEqual(loaded.avoid[0]["direction"], "rank(close)")

    def test_compress_dedupes_lessons(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_mem4")
        memory.add_lesson("Rank fields on returns is weak", 1, evidence=2)
        memory.add_lesson("Rank fields on returns is weak", 2, evidence=1)
        memory.compress()
        self.assertEqual(len(memory.lessons), 1)

    def test_lesson_promotion_to_long_term(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_tier")
        memory.add_lesson("Field returns ranks weakly", 1, evidence=2)
        memory.add_lesson("Field returns ranks weakly", 2, evidence=2)
        lesson = memory.lessons[0]
        self.assertEqual(lesson["tier"], "long")
        self.assertGreaterEqual(lesson["evidence"], memory.promote_evidence)

    def test_stale_short_lesson_archived(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_stale")
        memory.add_lesson("Transient observation", 1, evidence=1)
        memory.updated_round = 5
        memory.compress()
        kinds = {g["kind"] for g in memory.garbage}
        self.assertIn("stale_short_lesson", kinds)

    def test_garbage_archive(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_garbage")
        memory.archive("repeat_fail", "rank(close)", 1)
        memory.archive("unreproducible_high_signal", "rank(x)", 2)
        self.assertEqual(len(memory.garbage), 2)

    def test_lineage_deepen_attempts_tracked(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_lineage")
        memory.touch_lineage("rank(returns)", [], 0.6, 1)
        memory.touch_lineage("rank(returns)", [], 0.8, 2)
        self.assertEqual(memory.lineage_attempts("rank(returns)"), 2)
        targets = memory.deepening_targets(max_deepen_per_lineage=3, limit=4)
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["best_score"], 0.8)


class TestDiversity(unittest.TestCase):
    def test_redundancy_detected(self):
        from wqb_agent.diversity import is_redundant

        record = {
            "expression": "rank(ts_mean(returns, 10))",
            "fields_used": ["returns"],
        }
        pool = [
            {
                "expression": "rank(ts_mean(returns, 5))",
                "fields_used": ["returns"],
                "metrics": {"sharpe": 1.0, "fitness": 1.0},
            }
        ]
        redundant, keeper = is_redundant(record, pool)
        self.assertTrue(redundant)
        self.assertEqual(keeper["expression"], "rank(ts_mean(returns, 5))")

    def test_dedup_keeps_best(self):
        from wqb_agent.diversity import deduplicate

        pool = [
            {
                "expression": "rank(ts_mean(returns, 5))",
                "fields_used": ["returns"],
                "metrics": {"sharpe": 1.0, "fitness": 1.0},
            },
            {
                "expression": "rank(ts_mean(returns, 10))",
                "fields_used": ["returns"],
                "metrics": {"sharpe": 1.2, "fitness": 1.2},
            },
            {
                "expression": "-rank(ts_rank(volume, 20))",
                "fields_used": ["volume"],
                "metrics": {"sharpe": 1.5, "fitness": 1.5},
            },
        ]
        kept, dropped = deduplicate(pool)
        self.assertEqual(len(kept), 2)
        exprs = {r["expression"] for r in kept}
        self.assertIn("rank(ts_mean(returns, 10))", exprs)
        self.assertIn("-rank(ts_rank(volume, 20))", exprs)
        self.assertEqual(len(dropped), 1)


class TestHighSignalValidation(unittest.TestCase):
    def test_stable_signal_validated(self):
        from wqb_agent.validation import HighSignalValidator

        client = FakeClient()
        validator = HighSignalValidator(
            client,
            {},
            max_concurrent=3,
            poll_timeout_sec=30,
            min_valid_fitness=1.0,
        )
        record = {
            "expression": "rank(ts_mean(returns, 5))",
            "fields_used": ["returns"],
            "round_no": 1,
            "hypothesis_id": "h1",
            "datasets": [],
        }
        stable, details = validator.validate(record, alt_fields=["volume"])
        self.assertTrue(stable)
        self.assertGreaterEqual(len(details), 2)

    def test_unstable_signal_rejected(self):
        from wqb_agent.validation import HighSignalValidator

        class CollapseClient(FakeClient):
            def get_alpha(self, alpha_id):
                expression = self._alpha_expr.get(alpha_id, "")
                metrics = _fake_metrics(expression)
                # Signal collapses unless the expression keeps its original window.
                if "returns, 5" not in expression and "ts_mean" in expression:
                    metrics = {
                        "sharpe": 0.1,
                        "fitness": 0.1,
                        "turnover": 0.5,
                        "margin": 0.0,
                        "returns": 0.0,
                        "checks": [{"name": "limitations", "pass": True}],
                    }
                return {"is": metrics, "regular": expression}

        client = CollapseClient()
        validator = HighSignalValidator(
            client,
            {},
            max_concurrent=3,
            poll_timeout_sec=30,
            min_valid_fitness=1.0,
        )
        record = {
            "expression": "rank(ts_mean(returns, 5))",
            "fields_used": ["returns"],
            "round_no": 1,
            "hypothesis_id": "h1",
            "datasets": [],
        }
        stable, details = validator.validate(record, alt_fields=["volume"])
        self.assertFalse(stable)


class TestAgentLoop(unittest.TestCase):
    def test_high_signal_validated_enters_pool(self):
        class HighSignalClient(FakeClient):
            def get_alpha(self, alpha_id):
                expression = self._alpha_expr.get(alpha_id, "")
                if "returns" in expression:
                    metrics = {
                        "sharpe": 3.0,
                        "fitness": 3.0,
                        "turnover": 0.3,
                        "margin": 0.3,
                        "returns": 0.1,
                        "checks": [{"name": "limitations", "pass": True}],
                    }
                else:
                    metrics = _fake_metrics(expression)
                return {"is": metrics, "regular": expression}

        tmpdir = "/tmp/wqb_test_hisig"
        if os.path.exists(tmpdir):
            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    os.remove(os.path.join(root, fn))
        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = tmpdir
        config["agent"]["max_rounds"] = 1
        agent = Agent(HighSignalClient(), config)
        agent.run_one_round(1)
        self.assertGreaterEqual(len(agent.memory.submission_pool), 1)
        statuses = {r["status"] for r in agent.memory.submission_pool}
        self.assertIn("VALIDATED_HIGH_SIGNAL", statuses)

    def test_full_loop_state_persisted(self):
        tmpdir = "/tmp/wqb_test_agent"
        if os.path.exists(tmpdir):
            for root, _, files in os.walk(tmpdir):
                for fn in files:
                    os.remove(os.path.join(root, fn))
        agent, client = make_agent(tmpdir, rounds=2)
        agent.run()
        self.assertTrue(
            os.path.exists(os.path.join(tmpdir, "experience.json"))
        )
        self.assertTrue(
            os.path.exists(os.path.join(tmpdir, "trajectory.json"))
        )
        round_files = [
            f
            for f in os.listdir(tmpdir)
            if f.startswith("round_") and f.endswith(".json") and "_jobs" not in f
        ]
        self.assertEqual(len(round_files), 2)
        with open(os.path.join(tmpdir, "experience.json")) as f:
            memory_data = json.load(f)
        self.assertIsNotNone(memory_data["current_best"])
        self.assertGreater(len(memory_data["next"]), 0)
        seen_hypotheses = {e.hypothesis_id for e in agent.trajectory.experiments}
        self.assertEqual(len(seen_hypotheses), 2)

    def test_rounds_use_distinct_seeds(self):
        tmpdir = "/tmp/wqb_test_agent2"
        agent, client = make_agent(tmpdir, rounds=2)
        s1 = agent.run_one_round(1)
        s2 = agent.run_one_round(2)
        self.assertEqual(s1["round"], 1)
        self.assertEqual(s2["round"], 2)
        ids = {e.hypothesis_id for e in agent.trajectory.experiments}
        self.assertEqual(len(ids), 2)


if __name__ == "__main__":
    unittest.main()
