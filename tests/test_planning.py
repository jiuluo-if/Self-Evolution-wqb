import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_agent import make_agent

from wqb_agent.state import AlphaRecord, Experiment


class TestPlanning(unittest.TestCase):
    def test_explore_biased_when_pool_empty(self):
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        plan = agent._plan_round(1)
        self.assertGreaterEqual(plan["split"]["explore"], plan["split"]["deepen"])
        self.assertTrue(plan["hypothesis"]["id"].startswith("h-seed-"))

    def test_proven_lineage_increases_deepening(self):
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        plan_before = agent._plan_round(1)
        agent.memory.add_alpha(
            AlphaRecord(
                expression="rank(returns)",
                metrics={"fitness": 1.2, "sharpe": 1.2},
            )
        )
        agent.memory.touch_lineage(
            "rank(ts_mean(returns, 5))", [], 1.5, 1, fields_used=["returns"]
        )
        plan_after = agent._plan_round(2)
        self.assertGreater(
            plan_after["split"]["deepen"], plan_before["split"]["deepen"]
        )

    def test_failed_hypothesis_parked_and_not_rechosen(self):
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        for _ in range(2):
            e = Experiment(1, "h-seed-reversal", "rank(x)", {}, [])
            e.status = "FAILED"
            e.error = "WQBRejectedError: 422 invalid"
            agent.trajectory.add(e)
        self.assertIn("h-seed-reversal", agent._parked_hypotheses())
        chosen = agent._choose_hypothesis()
        self.assertNotEqual(chosen["id"], "h-seed-reversal")

    def test_infra_failures_do_not_park_hypothesis(self):
        # Timeouts / auth / rate-limit / 5xx failures are environmental; they
        # must not pause a research direction or inflate its failure count.
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        for error in (
            "WQBTimeoutError: Simulation polling timed out.",
            "WQBSimulationError: connection refused",
            "WQBAuthError: Authentication rejected (401).",
            "WQBRateLimitError: 429 rate limited",
        ):
            e = Experiment(1, "h-seed-reversal", "rank(x)", {}, [])
            e.status = "FAILED"
            e.error = error
            agent.trajectory.add(e)
        self.assertNotIn("h-seed-reversal", agent._parked_hypotheses())
        self.assertEqual(
            agent._hypothesis_failure_counts().get("h-seed-reversal", 0), 0
        )

    def test_research_failure_mixed_with_infra_does_not_park(self):
        # One genuine research failure plus an infra failure is not two
        # research failures in a row.
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        for error in (
            "WQBRejectedError: 422 invalid",
            "WQBTimeoutError: Simulation polling timed out.",
        ):
            e = Experiment(1, "h-seed-reversal", "rank(x)", {}, [])
            e.status = "FAILED"
            e.error = error
            agent.trajectory.add(e)
        self.assertNotIn("h-seed-reversal", agent._parked_hypotheses())

    def test_unused_hypotheses_preferred(self):
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        e = Experiment(1, "h-seed-analyst", "rank(target_price)", {}, [])
        e.status = "DONE"
        e.metrics = {"sharpe": 0.5, "fitness": 0.5, "checks": [], "passed": None}
        agent.trajectory.add(e)
        chosen = agent._choose_hypothesis()
        self.assertNotEqual(chosen["id"], "h-seed-analyst")

    def test_memory_next_and_lineages_steer_next_round(self):
        # The next/lessons/lineage written by Reflection must change what the
        # next round plans to do.
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=3)
        plan_before = agent._plan_round(1)

        agent.memory.add_next(
            "Deepen successful lineage: rank(ts_mean(returns, 5))",
            priority=5, source=1, round_no=1,
        )
        agent.memory.add_alpha(
            AlphaRecord(
                expression="rank(returns)",
                metrics={"fitness": 1.2, "sharpe": 1.2},
            )
        )
        agent.memory.touch_lineage(
            "rank(ts_mean(returns, 5))", [], 1.5, 1, fields_used=["returns"]
        )
        plan_after = agent._plan_round(2)

        self.assertGreater(
            plan_after["split"]["deepen"], plan_before["split"]["deepen"],
            "promising evidence must tilt the next round toward deepening",
        )
        targets = agent.memory.deepening_targets(agent.max_deepen_per_lineage)
        self.assertTrue(targets)
        self.assertEqual(
            targets[0]["expression"], "rank(ts_mean(returns, 5))"
        )


if __name__ == "__main__":
    unittest.main()
