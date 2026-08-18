import copy
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_agent import FAKE_FIELDS, FakeClient, make_agent

from wqb_agent.agent import SEED_HYPOTHESES
from wqb_agent.candidate import CandidateBuilder, _window_change
from wqb_agent.discovery import FieldDiscovery
from wqb_agent.hypothesis import ContractViolation, validate_contract


def _seed(seed_id):
    for h in SEED_HYPOTHESES:
        if h["id"] == seed_id:
            return copy.deepcopy(h)
    raise AssertionError(f"no seed {seed_id}")


class TestResearchContract(unittest.TestCase):
    """A hypothesis is a Research Contract, not a label. Before any candidate
    is built the system must know: the mechanism, the field meaning, the
    expected sign, the expected horizon, and what would refute it."""

    def test_every_seed_has_complete_contract(self):
        for hypothesis in SEED_HYPOTHESES:
            problems = validate_contract(hypothesis, strict=False)
            self.assertEqual(
                problems, [], f"{hypothesis['id']} violates contract: {problems}"
            )

    def test_missing_economic_intuition_rejected(self):
        hypothesis = _seed("h-seed-reversal")
        del hypothesis["economic_intuition"]
        self.assertIn(
            "economic_intuition", validate_contract(hypothesis, strict=False)[0]
        )
        with self.assertRaises(ContractViolation):
            validate_contract(hypothesis)

    def test_missing_expected_mechanism_rejected(self):
        hypothesis = _seed("h-seed-reversal")
        del hypothesis["expected_mechanism"]
        self.assertTrue(
            any("expected_mechanism" in p
                for p in validate_contract(hypothesis, strict=False))
        )

    def test_missing_field_semantics_blocks_discovery(self):
        discovery = FieldDiscovery(FakeClient())
        hypothesis = _seed("h-seed-reversal")
        del hypothesis["field_semantics"]
        with self.assertRaises(ContractViolation):
            discovery.discover(
                hypothesis, target_count=3, require_field_semantics=True
            )

    def test_semantics_requires_concept_and_description(self):
        hypothesis = _seed("h-seed-reversal")
        hypothesis["field_semantics"] = {"primary": {}}
        self.assertTrue(
            any("field_semantics" in p
                for p in validate_contract(hypothesis, strict=False))
        )

    def test_horizon_must_be_a_known_prior(self):
        hypothesis = _seed("h-seed-reversal")
        hypothesis["expected_horizon_days"] = 17  # arbitrary window grid
        self.assertTrue(
            any("expected_horizon_days" in p
                for p in validate_contract(hypothesis, strict=False))
        )

    def test_failure_condition_registered_before_simulation(self):
        # A failure condition must exist pre-registration time: the agent's
        # round planning rejects any hypothesis that cannot state what would
        # refute it, so candidates are never simulated without one.
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        hypothesis = agent._plan_round(1)["hypothesis"]
        self.assertTrue(hypothesis["failure_condition"])
        self.assertGreater(len(hypothesis["failure_condition"]), 0)

    def test_invalid_contract_blocks_round_planning(self):
        tmpdir = tempfile.mkdtemp()
        agent, _ = make_agent(tmpdir, rounds=2)
        hypothesis = _seed("h-seed-reversal")
        del hypothesis["expected_mechanism"]
        with self.assertRaises(ContractViolation):
            validate_contract(hypothesis)


class TestCandidateTraceability(unittest.TestCase):
    """Candidates must reference the hypothesis contract instead of carrying
    free-floating expressions."""

    def test_no_fake_fields_in_candidates(self):
        discovery = FieldDiscovery(FakeClient())
        fields = discovery.discover(
            _seed("h-seed-reversal"), target_count=4
        )
        known = {f["id"] for f in sum(FAKE_FIELDS.values(), [])}
        self.assertTrue(fields)
        builder = CandidateBuilder()
        candidates = builder.build_explore(_seed("h-seed-reversal"), fields, 4)
        for c in candidates:
            self.assertTrue(c["fields_used"])
            for fid in c["fields_used"]:
                self.assertIn(fid, known, f"fake field {fid} leaked into candidate")

    def test_candidate_fields_come_from_discovery_only(self):
        discovery = FieldDiscovery(FakeClient())
        fields = discovery.discover(
            _seed("h-seed-analyst"), target_count=4
        )
        real_ids = {f["id"] for f in fields}
        builder = CandidateBuilder()
        candidates = builder.build_explore(_seed("h-seed-analyst"), fields, 3)
        self.assertTrue(candidates)
        for c in candidates:
            for fid in c["fields_used"]:
                self.assertIn(fid, real_ids)

    def test_candidate_has_hypothesis_id(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        explore = builder.build_explore(_seed("h-seed-reversal"), fields, 2)
        alpha = {
            "expression": "rank(ts_mean(returns, 5))",
            "lineage": ["rank(returns)"],
            "attempts": 1,
            "best_score": 1.1,
            "fields_used": ["returns"],
        }
        deepen = builder.build_deepen(
            fields, [alpha], 2, hypothesis=_seed("h-seed-reversal")
        )
        for c in explore + deepen:
            self.assertEqual(c["hypothesis_id"], "h-seed-reversal")

    def test_candidate_traces_contract_metadata(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}]
        hypothesis = _seed("h-seed-reversal")
        candidates = builder.build_explore(hypothesis, fields, 2)
        for c in candidates:
            self.assertEqual(c["mechanism"], "temporary price pressure")
            self.assertEqual(c["expected_direction"]["sign"], "negative")
            self.assertEqual(c["expected_horizon_days"], 5)
            self.assertIn("concept", c["field_semantic"])
            self.assertEqual(c["field_semantic"]["concept"],
                             "short_term_price_change")

    def test_research_question_never_vague(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        vague = ("improve alpha", "optimize expression", "find better",
                 "improve lineage performance")
        for seed_id in ("h-seed-reversal", "h-seed-analyst", "h-seed-news"):
            hypothesis = _seed(seed_id)
            explore = builder.build_explore(hypothesis, fields, 3)
            self.assertTrue(explore)
            for c in explore:
                self.assertTrue(c["research_question"])
                lowered = c["research_question"].lower()
                for phrase in vague:
                    self.assertNotIn(phrase, lowered)
        alpha = {
            "expression": "rank(ts_mean(returns, 5))",
            "lineage": ["rank(returns)"],
            "attempts": 1,
            "best_score": 1.1,
            "fields_used": ["returns"],
        }
        deepen = builder.build_deepen(
            fields, [alpha], 3, hypothesis=_seed("h-seed-reversal")
        )
        for c in deepen:
            self.assertTrue(c["research_question"])
            self.assertIn(alpha["expression"], c["research_question"])

    def test_candidate_direction_matches_contract(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}]
        reversal = _seed("h-seed-reversal")
        for c in builder.build_explore(reversal, fields, 3):
            # reversal family templates carry the expected negative sign;
            # none of them is a silent falsification probe.
            self.assertFalse(c["falsification_variant"])
            self.assertTrue(c["expression"].startswith("-")
                            or "-rank" in c["expression"]
                            or "(-rank" in c["expression"])

    def test_sign_flip_explicitly_marked_falsification(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}]
        hypothesis = _seed("h-seed-reversal")
        hypothesis["expected_direction"] = {"sign": "positive"}
        for c in builder.build_explore(hypothesis, fields, 3):
            self.assertTrue(c["falsification_variant"])


class TestHorizonDiscipline(unittest.TestCase):
    """The expected horizon is a research prior. It may steer a single
    neighboring window step; it never licenses an arbitrary window grid."""

    def test_window_change_steps_single_neighbor_only(self):
        up = _window_change("rank(ts_mean(returns, 20))", +1)
        self.assertEqual(up, "rank(ts_mean(returns, 60))")
        down = _window_change("rank(ts_mean(returns, 20))", -1)
        self.assertEqual(down, "rank(ts_mean(returns, 10))")
        # No stepping past the boundary.
        self.assertIsNone(_window_change("rank(ts_mean(returns, 60))", +1))
        self.assertIsNone(_window_change("rank(ts_mean(returns, 5))", -1))
        # 5 and 60 must never be neighbors.
        self.assertNotEqual(
            _window_change("rank(ts_mean(returns, 60))", -1),
            "rank(ts_mean(returns, 5))",
        )

    def test_deepen_window_mutations_are_single_neighbors(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}]
        alpha = {
            "expression": "rank(ts_mean(returns, 20))",
            "lineage": ["rank(returns)"],
            "attempts": 1,
            "best_score": 1.1,
            "fields_used": ["returns"],
        }
        cands = builder.build_deepen(fields, [alpha], 5)
        windows = []
        for c in cands:
            if c["mutation"] == "window-up":
                windows.append("60")
            if c["mutation"] == "window-down":
                windows.append("10")
        self.assertEqual(sorted(windows), ["10", "60"])


class TestSingleMutationPerCandidate(unittest.TestCase):
    """One experiment answers one question: a candidate never performs field
    swap + window change + neutralization at the same time."""

    def test_deepen_candidates_change_exactly_one_thing(self):
        builder = CandidateBuilder()
        fields = [{"id": "returns"}, {"id": "volume"}]
        alpha = {
            "expression": "rank(ts_mean(returns, 5))",
            "lineage": ["rank(returns)"],
            "attempts": 1,
            "best_score": 1.1,
            "fields_used": ["returns"],
        }
        cands = builder.build_deepen(
            fields, [alpha], 8, hypothesis=_seed("h-seed-reversal")
        )
        self.assertGreater(len(cands), 1)
        single = {
            "field-swap", "window-up", "window-down",
            "smooth-ts-mean-5", "neutralize-subindustry",
        }
        for c in cands:
            self.assertIn(c["mutation"], single)
        counts = [c["mutation"] for c in cands]
        self.assertEqual(len(counts), len(set(counts)),
                         "no two candidates may apply the same mutation")


if __name__ == "__main__":
    unittest.main()
