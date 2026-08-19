import copy
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_agent import BASE_CONFIG, FAKE_FIELDS, FakeClient

from wqb_agent.agent import Agent, SEED_HYPOTHESES
from wqb_agent.candidate import CandidateBuilder
from wqb_agent.discovery import FieldDiscovery
from wqb_agent.hypothesis import ContractViolation


def _seed(seed_id):
    for h in SEED_HYPOTHESES:
        if h["id"] == seed_id:
            return copy.deepcopy(h)
    raise AssertionError(f"no seed {seed_id}")


class TestDatasetBoundary(unittest.TestCase):
    """hypothesis.datasets is an allowlist. Discovery must never query a
    dataset outside it; cross-dataset exploration requires an explicit
    new hypothesis / dataset variant."""

    def test_declared_allowlist_cannot_be_silently_extended(self):
        client = FakeClient()
        discovery = FieldDiscovery(client)
        # reversal declares pv1/pv13; univ1 shares its category but is NOT
        # declared, so it must never be queried.
        fields = discovery.discover(_seed("h-seed-reversal"), target_count=8)
        calls = {c[0] for c in client.datafield_calls}
        self.assertLessEqual(calls, {"pv1", "pv13"})
        self.assertNotIn("univ1", calls)
        for f in fields:
            self.assertIn(f["dataset"], {"pv1", "pv13"})

    def test_statement_keywords_never_widen_declared_datasets(self):
        client = FakeClient()
        discovery = FieldDiscovery(client)
        # The statement is packed with model/news keywords, but the contract
        # declares analyst4 only: Discovery stays inside the boundary.
        h = _seed("h-seed-analyst")
        h["statement"] = (
            "Model risk scores and news forecasts predict future target "
            "price levels with high probability."
        )
        fields = discovery.discover(h, target_count=4)
        calls = {c[0] for c in client.datafield_calls}
        self.assertEqual(calls, {"analyst4"})
        for f in fields:
            self.assertEqual(f["dataset"], "analyst4")

    def test_agent_round_only_queries_declared_datasets(self):
        tmpdir = tempfile.mkdtemp()
        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = str(tmpdir)
        config["agent"]["max_rounds"] = 1
        client = FakeClient()
        agent = Agent(client, config)
        agent.run_one_round(1)
        calls = {c[0] for c in client.datafield_calls}
        self.assertLessEqual(calls, {"pv1", "pv13"})

    def test_field_semantics_required_in_strict_mode(self):
        discovery = FieldDiscovery(FakeClient())
        h = _seed("h-seed-reversal")
        del h["field_semantics"]
        with self.assertRaises(ContractViolation):
            discovery.discover(h, target_count=3, require_field_semantics=True)


class _ExtraRiskClient(FakeClient):
    def get_datafields(self, dataset_id, limit=50, offset=0):
        self.datafield_calls.append((dataset_id, limit, offset))
        fields = list(FAKE_FIELDS.get(dataset_id, []))
        if dataset_id == "model16":
            # id coincidentally contains "risk"; name/description carry no
            # semantic meaning.
            fields = fields + [
                {
                    "id": "risk_side",
                    "name": "side table",
                    "description": "arbitrary descriptive row",
                }
            ]
        return fields[offset : offset + limit], len(fields)


class TestSemanticScoring(unittest.TestCase):
    """Semantic name/description match ranks above raw field-id
    coincidence; a field with zero semantic match is never selected."""

    def test_semantic_zero_field_never_selected(self):
        discovery = FieldDiscovery(_ExtraRiskClient())
        fields = discovery.discover(_seed("h-seed-model"), target_count=6)
        ids = [f["id"] for f in fields]
        self.assertIn("risk_score", ids)
        self.assertNotIn("risk_side", ids)

    def test_id_coincidence_does_not_outrank_semantic_description(self):
        class IdCoincidenceClient(FakeClient):
            def get_datafields(self, dataset_id, limit=50, offset=0):
                self.datafield_calls.append((dataset_id, limit, offset))
                if dataset_id == "analyst4":
                    return [
                        {
                            "id": "target_ratio_x",
                            "name": "misc ratio table",
                            "description": "unrelated daily tabular data",
                        },
                        {
                            "id": "zzz",
                            "name": "Analyst target price",
                            "description": "consensus analyst target price revision",
                        },
                    ], 2
                return [], 0

        discovery = FieldDiscovery(IdCoincidenceClient())
        fields = discovery.discover(_seed("h-seed-analyst"), target_count=2)
        ids = [f["id"] for f in fields]
        self.assertIn("zzz", ids)
        self.assertNotIn("target_ratio_x", ids)
        self.assertGreater(fields[0]["field_match"]["semantic_score"], 0)

    def test_no_valid_field_yields_zero_candidates(self):
        client = FakeClient()
        discovery = FieldDiscovery(client)
        h = _seed("h-seed-analyst")
        h["field_semantics"] = {
            "primary": {
                "concept": "night_sky_luminance",
                "description": "brightness of the night sky above the exchange",
            }
        }
        fields = discovery.discover(h, target_count=3)
        self.assertEqual(fields, [])
        self.assertTrue(discovery.last_outcome["no_match"])
        self.assertEqual(CandidateBuilder().build_explore(h, fields, 3), [])

    def test_field_match_evidence_explains_selection(self):
        discovery = FieldDiscovery(FakeClient())
        fields = discovery.discover(_seed("h-seed-analyst"), target_count=2)
        self.assertTrue(fields)
        fm = fields[0]["field_match"]
        for key in (
            "semantic_concept",
            "matched_terms",
            "id_score",
            "name_score",
            "description_score",
            "semantic_score",
            "total_score",
        ):
            self.assertIn(key, fm)
        self.assertEqual(fm["semantic_concept"], "target_price_revision")
        self.assertGreater(fm["semantic_score"], 0)
        self.assertTrue(fm["matched_terms"])


class TestCandidateFieldSource(unittest.TestCase):
    def test_candidate_fields_come_from_real_api_only(self):
        discovery = FieldDiscovery(FakeClient())
        h = _seed("h-seed-analyst")
        fields = discovery.discover(h, target_count=4)
        real_ids = {f["id"] for f in fields}
        builder = CandidateBuilder()
        candidates = builder.build_explore(h, fields, 3)
        self.assertTrue(candidates)
        for c in candidates:
            for fid in c["fields_used"]:
                self.assertIn(fid, real_ids)
            # A field_semantics concept is a meaning, never a WQB field id.
            self.assertNotIn(c["field_semantic"]["concept"], c["fields_used"])

    def test_candidate_carries_field_discovery_reason(self):
        discovery = FieldDiscovery(FakeClient())
        h = _seed("h-seed-analyst")
        fields = discovery.discover(h, target_count=2)
        builder = CandidateBuilder()
        candidate = builder.build_explore(h, fields, 1)[0]
        reason = candidate["field_discovery_reason"]
        self.assertIsNotNone(reason)
        self.assertEqual(reason["dataset"], "analyst4")
        self.assertEqual(reason["semantic_concept"], "target_price_revision")
        self.assertGreater(reason["semantic_score"], 0)
        self.assertGreater(reason["match_score"], 0)


class TestDiscoveryBudgetAndCache(unittest.TestCase):
    def test_field_api_budget_never_exceeded(self):
        client = FakeClient()
        discovery = FieldDiscovery(client, pagination_limit=2, max_pages=10)
        discovery.reset_budget(1)
        discovery.discover(_seed("h-seed-analyst"), target_count=6)
        self.assertLessEqual(len(client.datafield_calls), 1)

    def test_cache_hit_does_not_consume_budget(self):
        client = FakeClient()
        discovery = FieldDiscovery(client, pagination_limit=50)
        discovery.reset_budget(100)
        h = _seed("h-seed-reversal")
        discovery.discover(h, target_count=4)
        first_calls = len(client.datafield_calls)
        discovery.discover(h, target_count=4)
        self.assertEqual(len(client.datafield_calls), first_calls)


class TestInfraFailureClassification(unittest.TestCase):
    """An API timeout / 429 / auth / 5xx is an infrastructure failure and
    must never be read as 'the hypothesis has no matching field'."""

    def test_infra_failure_distinct_from_genuine_no_match(self):
        class TimeoutClient(FakeClient):
            def get_datafields(self, dataset_id, limit=50, offset=0):
                raise TimeoutError("API timeout after 5s")

        discovery = FieldDiscovery(TimeoutClient())
        discovery.discover(_seed("h-seed-analyst"), target_count=3)
        self.assertTrue(discovery.last_outcome["infra_failure"])
        self.assertFalse(discovery.last_outcome["no_match"])

        no_match = _seed("h-seed-analyst")
        no_match["field_semantics"] = {
            "primary": {
                "concept": "night_sky_luminance",
                "description": "brightness of the night sky above the exchange",
            }
        }
        discovery2 = FieldDiscovery(FakeClient())
        discovery2.discover(no_match, target_count=3)
        self.assertFalse(discovery2.last_outcome["infra_failure"])
        self.assertTrue(discovery2.last_outcome["no_match"])

    def test_429_failure_never_enters_research_memory(self):
        class RateLimitedClient(FakeClient):
            def get_datafields(self, dataset_id, limit=50, offset=0):
                raise Exception("429 Too Many Requests")

        tmpdir = tempfile.mkdtemp()
        config = json.loads(json.dumps(BASE_CONFIG))
        config["agent"]["state_dir"] = str(tmpdir)
        config["agent"]["max_rounds"] = 1
        agent = Agent(RateLimitedClient(), config)
        summary = agent.run_one_round(1)
        self.assertIsNone(summary)
        self.assertEqual(agent.trajectory.experiments, [])
        self.assertEqual(agent.memory.beliefs, [])
        self.assertEqual(agent.memory.lessons, [])
        self.assertEqual(agent.memory.avoid, [])


if __name__ == "__main__":
    unittest.main()
