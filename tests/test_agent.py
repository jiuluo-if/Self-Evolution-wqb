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
    def test_from_scratch_returns_six(self):
        builder = CandidateBuilder(neutralization="subindustry")
        fields = [{"id": "returns"}, {"id": "volume"}]
        hypothesis = {"direction": "reversal", "tags": ["return"]}
        candidates = builder.build(hypothesis, fields, None, count=6)
        self.assertEqual(len(candidates), 6)
        self.assertTrue(all("returns" in c["expression"] for c in candidates))

    def test_reversal_flips_sign(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}]
        candidates = builder.build(
            {"direction": "reversal"}, fields, None, count=2
        )
        self.assertTrue(candidates[0]["expression"].startswith("-rank"))

    def test_mutates_best_single_variable(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        best = {
            "id": "b1",
            "expression": "rank(returns)",
            "fields_used": ["returns"],
            "metrics": {"sharpe": 0.6, "fitness": 0.6},
        }
        candidates = builder.build({}, fields, best, count=6)
        exprs = [c["expression"] for c in candidates]
        self.assertNotIn("rank(returns)", exprs)
        self.assertEqual(len(exprs), 6)
        self.assertTrue(all(c["parent"] == "b1" for c in candidates))


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

    def test_fail_diagnosis_adds_avoid(self):
        memory = ExperienceMemory(state_dir="/tmp/wqb_test_mem2")
        reflector = Reflector(memory)
        e = Experiment(1, "h", "rank(close)", {}, ["close"])
        e.status = "FAILED"
        e.error = "Simulation rejected (422): syntax error"
        reflector.reflect(1, {"tags": ["price"], "direction": "long"}, [e])
        self.assertEqual(len(memory.avoid), 1)
        self.assertIn("syntax", memory.avoid[0]["reason"])


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


class TestAgentLoop(unittest.TestCase):
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
            f for f in os.listdir(tmpdir) if f.startswith("round_")
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
