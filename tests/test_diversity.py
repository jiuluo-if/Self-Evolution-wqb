import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_agent import BASE_CONFIG, FakeClient, _fake_metrics, make_agent

from wqb_agent.diversity import (
    concentration,
    dataset_family,
    deduplicate,
    filter_candidates,
    fingerprint,
    is_redundant,
    lineage_root,
    operator_family,
    pool_diversity_summary,
    select_diverse,
)
from wqb_agent.memory import ExperienceMemory
from wqb_agent.reflection import Reflector
from wqb_agent.state import AlphaRecord, Experiment
from wqb_agent.validation import HighSignalValidator


def rec(expr, fields=None, hypothesis=None, datasets=None, lineage=None,
        metrics=None, mutation=None, parent=None, research_question=None):
    r = {
        "expression": expr,
        "fields_used": fields or [],
        "hypothesis_id": hypothesis,
        "datasets": datasets or [],
        "lineage": lineage or [],
        "metrics": metrics or {"sharpe": 0.5, "fitness": 0.5},
    }
    if mutation:
        r["mutation"] = mutation
    if parent:
        r["parent"] = parent
    if research_question:
        r["research_question"] = research_question
    return r


def alpha(expression, score, hypothesis="h-a", datasets=None, lineage=None,
          fields=None):
    return AlphaRecord(
        expression=expression,
        metrics={"sharpe": score, "fitness": score},
        fields_used=fields or ["returns"],
        datasets=datasets or ["pv1"],
        hypothesis_id=hypothesis,
        lineage=lineage or [],
    )


class TestFingerprint(unittest.TestCase):
    def test_operator_family_normalizes_structure(self):
        # Same structure, different field -> same family.
        self.assertEqual(
            operator_family("rank(ts_mean(returns, 20))"),
            operator_family("rank(ts_mean(volume, 20))"),
        )
        self.assertEqual(
            operator_family("rank(ts_mean(returns, 20))"), "rank-ts_mean"
        )
        # Different structure -> different family.
        self.assertNotEqual(
            operator_family("rank(returns)"),
            operator_family("rank(ts_mean(returns, 20))"),
        )

    def test_sign_ignored_in_operator_family(self):
        self.assertEqual(operator_family("-rank(returns)"), "rank")
        self.assertEqual(operator_family("rank(returns)"), "rank")

    def test_dataset_family_strips_trailing_digits(self):
        self.assertEqual(dataset_family("pv1"), "pv")
        self.assertEqual(dataset_family("pv13"), "pv")
        self.assertEqual(dataset_family("analyst4"), "analyst")
        self.assertEqual(dataset_family("fundamental2"), "fundamental")

    def test_lineage_root_is_deepest_ancestor(self):
        self.assertEqual(lineage_root({"lineage": ["b", "a-root"]}), "a-root")
        self.assertEqual(
            lineage_root({"expression": "rank(x)", "lineage": []}),
            "rank(x)",
        )

    def test_fingerprint_has_required_dimensions(self):
        f = fingerprint(
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-seed-reversal", datasets=["pv1", "pv13"],
                lineage=["rank(returns)"])
        )
        for key in ("hypothesis_id", "dataset_family", "fields",
                    "operator_family", "lineage_root"):
            self.assertIn(key, f)
        self.assertEqual(f["operator_family"], "rank-ts_mean")
        self.assertIn("pv", f["dataset_family"])
        self.assertEqual(f["lineage_root"], "rank(returns)")


class TestRedundancy(unittest.TestCase):
    def test_identical_expression_redundant(self):
        pool = [rec("rank(returns)", fields=["returns"])]
        redundant, keeper = is_redundant(
            rec("rank(returns)", fields=["returns"]), pool
        )
        self.assertTrue(redundant)
        self.assertEqual(keeper["expression"], "rank(returns)")

    def test_same_field_family_hypothesis_redundant(self):
        # Same hypothesis, same operator family, same fields but different
        # expression tokens -> still highly redundant.
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        redundant, _ = is_redundant(
            rec("rank(ts_mean(returns, 10))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"]),
            pool,
        )
        self.assertTrue(redundant)

    def test_same_lineage_different_window_redundant_in_pool(self):
        # window-up / window-down / smoothing of one lineage share a root and
        # must not occupy independent pool slots.
        pool = [
            rec("-rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"], lineage=["-rank(returns)"])
        ]
        redundant, _ = is_redundant(
            rec("-rank(ts_mean(returns, 20))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"], lineage=["-rank(returns)"]),
            pool,
        )
        self.assertTrue(redundant)

    def test_different_hypothesis_similar_expression_not_deduped(self):
        # A similar expression under a different hypothesis is a new economic
        # question and must not be deduplicated.
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        redundant, _ = is_redundant(
            rec("rank(ts_mean(returns, 10))", fields=["returns"],
                hypothesis="h-b", datasets=["pv1"]),
            pool,
        )
        self.assertFalse(redundant)

    def test_different_dataset_increases_novelty(self):
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        redundant, _ = is_redundant(
            rec("rank(ts_mean(returns, 10))", fields=["returns"],
                hypothesis="h-a", datasets=["analyst4"]),
            pool,
        )
        self.assertFalse(redundant)

    def test_near_identical_cross_hypothesis_still_deduped(self):
        # The one exception: byte-for-byte the same expression is never worth
        # two pool slots even under different hypotheses.
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        redundant, _ = is_redundant(
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-b", datasets=["pv1"]),
            pool,
        )
        self.assertTrue(redundant)

    def test_opposite_direction_not_redundant(self):
        # rank(x) and -rank(x) share an operator family but are opposite
        # research directions; one must never evict the other.
        pool = [
            rec("rank(ts_delta(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        redundant, _ = is_redundant(
            rec("-rank(ts_delta(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"]),
            pool,
        )
        self.assertFalse(redundant)

    def test_dedup_keeps_best_regression(self):
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                metrics={"sharpe": 1.0, "fitness": 1.0}),
            rec("rank(ts_mean(returns, 10))", fields=["returns"],
                metrics={"sharpe": 1.2, "fitness": 1.2}),
            rec("-rank(ts_rank(volume, 20))", fields=["volume"],
                metrics={"sharpe": 1.5, "fitness": 1.5}),
        ]
        kept, dropped = deduplicate(pool)
        self.assertEqual(len(kept), 2)
        exprs = {r["expression"] for r in kept}
        self.assertIn("rank(ts_mean(returns, 10))", exprs)
        self.assertIn("-rank(ts_rank(volume, 20))", exprs)
        self.assertEqual(len(dropped), 1)


class TestSelectDiverse(unittest.TestCase):
    def test_single_lineage_root_capped(self):
        # One dominant root plus one other: the root holds at most
        # max_per_lineage slots and the other root still gets in.
        pool = [
            alpha(f"rank(ts_mean(returns, {w}))", 1.9 - i * 0.1,
                  lineage=["-rank(returns)"])
            for i, w in enumerate((5, 10, 20, 60, 120))
        ]
        pool.append(alpha("rank(volume)", 0.8, lineage=["rank(volume)"]))
        selected = select_diverse([r.to_dict() for r in pool], n=4,
                                  max_per_lineage=2)
        roots = [lineage_root(r) for r in selected]
        self.assertEqual(roots.count("-rank(returns)"), 2)
        self.assertIn("rank(volume)", roots)

    def test_single_dataset_family_not_fill_pool(self):
        # Six high-scoring pv-family alphas plus two analyst-family ones: with
        # a dataset-family cap the pool keeps coverage of both sources.
        pool = [
            alpha(f"rank(ts_mean(ret{i}, 5))", 2.0 - i * 0.05,
                  hypothesis="h-a", datasets=["pv1"])
            for i in range(6)
        ]
        pool += [
            alpha("rank(target_price)", 1.0, hypothesis="h-a",
                  datasets=["analyst4"]),
            alpha("rank(eps_estimate)", 0.9, hypothesis="h-a",
                  datasets=["analyst4"]),
        ]
        selected = select_diverse(
            [r.to_dict() for r in pool], n=5,
            hypothesis_cap_ratio=1.0,  # isolate the dataset-family cap
        )
        fams = concentration(selected, "dataset_family")
        self.assertGreaterEqual(len(fams), 2)
        self.assertLessEqual(fams.get("pv", 0), 3)

    def test_top_score_not_sole_retention_criterion(self):
        # Two near-duplicate high scores on one root must not crowd out a
        # genuinely different lower-scoring lineage.
        pool = [
            alpha("rank(ts_mean(returns, 5))", 1.9, lineage=["root-returns"]),
            alpha("rank(ts_mean(returns, 10))", 1.8, lineage=["root-returns"]),
            alpha("rank(ts_mean(returns, 20))", 1.7, lineage=["root-returns"]),
            alpha("rank(ts_mean(volume, 5))", 1.1, lineage=["root-volume"]),
        ]
        selected = select_diverse([r.to_dict() for r in pool], n=2,
                                  max_per_lineage=2)
        roots = {lineage_root(r) for r in selected}
        self.assertIn("root-returns", roots)
        self.assertIn("root-volume", roots)

    def test_trim_keeps_multiple_hypothesis_dataset_operator_family(self):
        # After trimming, the pool still spans hypotheses, datasets and
        # operator families rather than the top scores of one family.
        pool = []
        for i, (h, ds, opf_expr) in enumerate([
            ("h-a", ["pv1"], "rank(returns)"),
            ("h-b", ["analyst4"], "rank(ts_mean(target_price, 5))"),
            ("h-c", ["model16"], "-rank(ts_rank(risk_score, 20))"),
        ]):
            pool.append(alpha(opf_expr, 1.5 - i * 0.1, hypothesis=h, datasets=ds))
        # many same-root high-scoring duplicates of h-a
        pool += [
            alpha(f"rank(ts_mean(returns, {w}))", 1.9, hypothesis="h-a",
                  datasets=["pv1"], lineage=["rank(returns)"])
            for w in (10, 20, 60)
        ]
        selected = select_diverse([r.to_dict() for r in pool], n=5,
                                  max_per_lineage=2)
        self.assertGreaterEqual(len(concentration(selected, "hypothesis_id")), 2)
        self.assertGreaterEqual(len(concentration(selected, "dataset_family")), 2)
        self.assertGreaterEqual(len(concentration(selected, "operator_family")), 2)


class TestPoolPolicy(unittest.TestCase):
    def test_same_lineage_single_pool_slot(self):
        # window variants of one lineage: only the stronger one holds the slot.
        memory = ExperienceMemory(state_dir=tempfile.mkdtemp())
        reflector = Reflector(memory)
        exps = []
        for expr in ("rank(ts_mean(returns, 5))", "rank(ts_mean(returns, 20))"):
            e = Experiment(1, "h", expr, {}, ["returns"],
                           datasets=["pv1"], lineage=["rank(returns)"])
            e.metrics = _fake_metrics(expr)
            e.status = "DONE"
            exps.append(e)
        reflector.reflect(1, {"tags": ["return"], "direction": "long"}, exps)
        self.assertEqual(len(memory.submission_pool), 1)

    def test_trim_pool_via_memory_caps_lineage(self):
        memory = ExperienceMemory(state_dir=tempfile.mkdtemp(), max_pool=12)
        for w in (5, 10, 20, 60, 120, 240, 480, 960, 1920, 3840):
            memory.submission_pool.append(
                alpha(f"rank(ts_mean(returns, {w}))", 1.0,
                      lineage=["-rank(returns)"]).to_dict()
            )
        for i, w in enumerate((5, 10, 20, 60)):
            memory.submission_pool.append(
                alpha(f"rank(ts_mean(volume, {w}))", 0.8,
                      lineage=["-rank(volume)"]).to_dict()
            )
        memory._trim_pool()
        self.assertLessEqual(len(memory.submission_pool), memory.max_pool)
        roots = [lineage_root(r) for r in memory.submission_pool]
        self.assertLessEqual(roots.count("-rank(returns)"), memory.max_per_lineage)
        self.assertIn("-rank(volume)", roots)

    def test_deepen_targets_capped_per_root(self):
        memory = ExperienceMemory(state_dir=tempfile.mkdtemp())
        for w, score in ((5, 1.9), (10, 1.8), (20, 1.7), (60, 1.6)):
            memory.touch_lineage(
                f"rank(ts_mean(returns, {w}))", ["-rank(returns)"], score, 1
            )
        memory.touch_lineage("rank(ts_mean(volume, 5))", ["-rank(volume)"], 1.1, 1)
        targets = memory.deepening_targets(
            max_deepen_per_lineage=3, limit=4, max_per_lineage=2
        )
        roots = [lineage_root(t) for t in targets]
        self.assertLessEqual(roots.count("-rank(returns)"), 2)
        self.assertIn("-rank(volume)", roots)

    def test_current_best_does_not_concentrate_deepening(self):
        # A high current_best must tilt toward deepening, but never let one
        # root monopolize the deepening targets.
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        for w, score in ((5, 1.9), (10, 1.8), (20, 1.7), (60, 1.6)):
            agent.memory.touch_lineage(
                f"rank(ts_mean(returns, {w}))", ["-rank(returns)"], score, 1
            )
        agent.memory.touch_lineage(
            "rank(ts_mean(volume, 5))", ["-rank(volume)"], 1.2, 1
        )
        agent.memory.current_best = {
            "expression": "rank(ts_mean(returns, 5))",
            "metrics": {"sharpe": 3.0, "fitness": 3.0},
        }
        targets = agent.memory.deepening_targets(agent.max_deepen_per_lineage)
        roots = [lineage_root(t) for t in targets]
        self.assertLessEqual(roots.count("-rank(returns)"), 2)
        self.assertIn("-rank(volume)", roots)


class TestEarlyFilter(unittest.TestCase):
    def test_redundant_explore_candidate_filtered(self):
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        cand = rec("rank(ts_mean(returns, 10))", fields=["returns"],
                   hypothesis="h-a", datasets=["pv1"], mutation="explore")
        kept, blocked = filter_candidates([cand], [], pool)
        self.assertEqual(len(kept), 0)
        self.assertEqual(blocked[0][1], "redundant")

    def test_novel_explore_candidate_kept(self):
        pool = [
            rec("rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"])
        ]
        cand = rec("rank(target_price)", fields=["target_price"],
                   hypothesis="h-a", datasets=["analyst4"], mutation="explore")
        kept, blocked = filter_candidates([cand], [], pool)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(blocked), 0)

    def test_research_mutation_not_blocked(self):
        # window-up of an existing lineage is a research mutation: it may be
        # simulated even though it shares a root with pool entries.
        pool = [
            rec("-rank(ts_mean(returns, 5))", fields=["returns"],
                hypothesis="h-a", datasets=["pv1"],
                lineage=["-rank(returns)"])
        ]
        cand = rec("-rank(ts_mean(returns, 20))", fields=["returns"],
                   hypothesis="h-a", datasets=["pv1"],
                   lineage=["-rank(returns)"], mutation="window-up",
                   parent="-rank(ts_mean(returns, 5))",
                   research_question="Does a longer horizon survive?")
        kept, blocked = filter_candidates([cand], [], pool)
        self.assertEqual(len(kept), 1)
        self.assertEqual(len(blocked), 0)

    def test_validation_perturbation_not_intercepted(self):
        client = FakeClient()
        validator = HighSignalValidator(client, BASE_CONFIG["simulation"])
        record = rec("rank(ts_mean(returns, 5))", fields=["returns"],
                     datasets=["pv1"], metrics={"sharpe": 2.6, "fitness": 2.6})
        jobs, _ = validator.build_perturbation_jobs(record)
        self.assertTrue(jobs)
        for job in jobs:
            self.assertTrue(job.mutation.startswith("validation-"))
            cand = {
                "expression": job.expression,
                "mutation": job.mutation,
                "parent": record["expression"],
                "research_question": "robustness probe",
            }
            kept, blocked = filter_candidates([cand], [], [])
            self.assertEqual(len(kept), 1)
            self.assertEqual(len(blocked), 0)

    def test_already_simulated_expression_blocked(self):
        cand = rec("rank(returns)", fields=["returns"], mutation="explore")
        kept, blocked = filter_candidates(
            [cand], {"rank(returns)"}, []
        )
        self.assertEqual(len(kept), 0)
        self.assertEqual(blocked[0][1], "already-simulated")


class TestAgentIntegration(unittest.TestCase):
    def test_agent_round_reports_pool_diversity(self):
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        summary = agent.run_one_round(1)
        self.assertIn("diversity", summary)
        self.assertIsInstance(summary["diversity"], dict)
        for dim in ("hypothesis_id", "dataset_family",
                    "operator_family", "lineage_root"):
            self.assertIn(dim, summary["diversity"])

    def test_agent_rejects_redundant_simulations(self):
        # Simulating the same expression twice is blocked by the early filter.
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        agent.run_one_round(1)
        simulated = agent._simulated_expressions()
        self.assertTrue(simulated)
        expr = next(iter(simulated))
        cand = rec(expr, fields=["returns"], mutation="explore")
        kept, blocked = filter_candidates([cand], simulated, [])
        self.assertEqual(len(kept), 0)
        self.assertEqual(blocked[0][1], "already-simulated")


if __name__ == "__main__":
    unittest.main()
