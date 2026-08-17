import json
import os
import time

from .candidate import CandidateBuilder
from .client import WQBClient
from .discovery import FieldDiscovery
from .diversity import deduplicate
from .memory import ExperienceMemory
from .reflection import Reflector
from .simulator import Simulator
from .state import AlphaRecord, Experiment, ResearchState, Trajectory
from .validation import HighSignalValidator

SEED_HYPOTHESES = [
    {
        "id": "h-seed-reversal",
        "statement": "Short-term return reversal: stocks that rose sharply over the last 5 days tend to revert in the near term.",
        "tags": ["reversal", "return", "price", "short-term"],
        "direction": "reversal",
        "datasets": ["pv1", "pv13"],
    },
    {
        "id": "h-seed-analyst",
        "statement": "Analyst target-price revisions upward predict short-term outperformance.",
        "tags": ["analyst", "forecast", "revision", "target"],
        "direction": "long",
        "datasets": ["analyst4"],
    },
    {
        "id": "h-seed-option",
        "statement": "Stocks with elevated implied volatility earn lower forward returns.",
        "tags": ["option", "volatility", "implied", "risk"],
        "direction": "reversal",
        "datasets": ["option8", "option9"],
    },
    {
        "id": "h-seed-model",
        "statement": "High model risk scores predict lower forward returns.",
        "tags": ["model", "score", "risk", "composite"],
        "direction": "reversal",
        "datasets": ["model16", "model51"],
    },
    {
        "id": "h-seed-news",
        "statement": "Positive news sentiment predicts short-term positive returns.",
        "tags": ["news", "sentiment", "positive"],
        "direction": "long",
        "datasets": ["news12", "news18"],
    },
    {
        "id": "h-seed-fundamental",
        "statement": "Firms with strong earnings growth continue to outperform.",
        "tags": ["fundamental", "growth", "earning"],
        "direction": "long",
        "datasets": ["fundamental2", "fundamental6"],
    },
]


class Agent:
    def __init__(self, client, config):
        self.client = client
        self.simulation_settings = config["simulation"]
        agent_cfg = config["agent"]
        self.state_dir = agent_cfg.get("state_dir", ".wqb_state")
        self.max_rounds = agent_cfg.get("max_rounds", 5)
        self.candidates_per_round = agent_cfg.get("candidates_per_round", 6)
        self.fields_per_discovery = agent_cfg.get("fields_per_discovery", 6)
        self.pagination_limit = agent_cfg.get("pagination_limit", 50)
        self.max_pagination_pages = agent_cfg.get("max_pagination_pages", 20)
        self.poll_timeout_sec = agent_cfg.get("poll_timeout_sec", 900)
        self.max_concurrent_sims = agent_cfg.get("max_concurrent_sims", 3)

        self.explore_ratio = agent_cfg.get("explore_ratio", 0.5)
        self.good_sharpe = agent_cfg.get("good_sharpe", 1.0)
        self.good_fitness = agent_cfg.get("good_fitness", 1.0)
        self.high_sharpe = agent_cfg.get("high_signal_sharpe", 2.5)
        self.high_fitness = agent_cfg.get("high_signal_fitness", 2.0)
        self.max_deepen_per_lineage = agent_cfg.get("max_deepen_per_lineage", 3)
        self.sim_budget_per_round = agent_cfg.get("sim_budget_per_round", 10)
        self.validation_budget = agent_cfg.get("validation_budget_per_round", 4)
        self.discovery_budget_per_round = agent_cfg.get(
            "discovery_budget_per_round", 8
        )

        self.memory = ExperienceMemory(self.state_dir)
        self.trajectory = Trajectory()
        self.builder = CandidateBuilder(
            neutralization=self.simulation_settings.get(
                "neutralization", "SUBINDUSTRY"
            ),
            max_deepen_per_lineage=self.max_deepen_per_lineage,
        )
        self.discovery = FieldDiscovery(
            self.client,
            pagination_limit=self.pagination_limit,
            max_pages=self.max_pagination_pages,
        )
        self.simulator = Simulator(
            self.client,
            max_concurrent=self.max_concurrent_sims,
            poll_timeout_sec=self.poll_timeout_sec,
        )
        self.reflector = Reflector(
            self.memory,
            good_sharpe=self.good_sharpe,
            good_fitness=self.good_fitness,
            high_sharpe=self.high_sharpe,
            high_fitness=self.high_fitness,
        )
        self.validator = HighSignalValidator(
            self.client,
            self.simulation_settings,
            max_concurrent=self.max_concurrent_sims,
            poll_timeout_sec=self.poll_timeout_sec,
            min_valid_fitness=self.good_fitness,
        )
        self._last_fields = []

    def run(self, max_rounds=None):
        rounds = max_rounds or self.max_rounds
        self._load_state()
        for round_no in range(1, rounds + 1):
            self.run_one_round(round_no)

    def run_one_round(self, round_no):
        start = time.time()
        self.discovery.reset_budget(self.discovery_budget_per_round)
        plan = self._plan_round(round_no)
        hypothesis = plan["hypothesis"]
        split = plan["split"]
        print(f"\n=== Round {round_no} ===")
        print(f"Hypothesis: {hypothesis['statement']}")
        print(
            f"Pool split: explore={split['explore']} deepen={split['deepen']} "
            f"(pool={len(self.memory.submission_pool)})"
        )

        fields = self._last_fields
        if split["explore"] > 0:
            fields = self.discovery.discover(
                hypothesis, target_count=self.fields_per_discovery
            )
            if not fields:
                # Fall back to previously discovered fields (cache reuse).
                fields = self._last_fields
            if not fields:
                print("No fields discovered; skipping round.")
                return None
            self._last_fields = fields
        print(f"Fields ({len(fields)}): {[f['id'] for f in fields]}")

        candidates = []
        if split["explore"] > 0:
            candidates += self.builder.build_explore(
                hypothesis, fields, split["explore"]
            )
        if split["deepen"] > 0:
            active = self.memory.deepening_targets(
                self.max_deepen_per_lineage, limit=4
            )
            if active:
                candidates += self.builder.build_deepen(
                    fields, active, split["deepen"]
                )

        budget = self.sim_budget_per_round
        if candidates:
            candidates = candidates[:budget]
        if not candidates:
            print("No candidates built; skipping round.")
            return None

        experiments = [
            Experiment(
                round_no,
                hypothesis["id"],
                c["expression"],
                self.simulation_settings,
                c.get("fields_used") or [f["id"] for f in fields],
                datasets=hypothesis.get("datasets", []),
                lineage=c.get("lineage", []),
            )
            for c in candidates
        ]
        for c, exp in zip(candidates, experiments):
            self._attach_candidate_meta(exp, c)

        self.simulator.run(experiments)
        for exp in experiments:
            self.trajectory.add(exp)
            self._print_experiment(exp)

        summary = self.reflector.reflect(round_no, hypothesis, experiments)

        validation_used = self._validate_suspicious(summary, round_no)
        self._dedup_pool(round_no)
        self._prune_stale_lineages(round_no)

        state = ResearchState(
            round_no=round_no,
            hypothesis=hypothesis,
            dataset=hypothesis.get("datasets"),
            fields_used=[f["id"] for f in fields],
        )
        self._save_state(state)
        self._print_summary(
            summary,
            elapsed=time.time() - start,
            validation_used=validation_used,
        )
        return summary

    # ---- planning ----

    def _plan_round(self, round_no):
        pool = self.memory.submission_pool
        n = self.candidates_per_round
        explore = max(1, int(round(n * self.explore_ratio)))
        deepen = n - explore
        if not pool:
            # Not enough proven directions yet: bias toward exploration.
            deepen = max(0, deepen - 1)
            explore = n - deepen

        # Exploration is always driven by a real hypothesis (hypothesis-first),
        # never by a deepen pseudo-hypothesis with weak field keywords.
        used = self._used_hypothesis_ids()
        hypothesis = None
        for seed in SEED_HYPOTHESES:
            if seed["id"] not in used:
                hypothesis = dict(seed)
                break
        if hypothesis is None:
            hypothesis = dict(SEED_HYPOTHESES[round_no % len(SEED_HYPOTHESES)])
        return {"hypothesis": hypothesis, "split": {"explore": explore, "deepen": deepen}}

    def _used_hypothesis_ids(self):
        return {e.hypothesis_id for e in self.trajectory.experiments}

    def _attach_candidate_meta(self, experiment, candidate):
        experiment.mutation = candidate.get("mutation")
        experiment.rationale = candidate.get("rationale")

    # ---- validation & pool maintenance ----

    def _validate_suspicious(self, summary, round_no):
        suspicious = summary.get("suspicious") or []
        if not suspicious:
            return 0
        alt_fields = [f["id"] for f in self._last_fields]
        used = 0
        for rec in suspicious:
            if used >= self.validation_budget:
                break
            rec_obj = AlphaRecord.from_dict(rec)
            stable, details = self.validator.validate(rec, alt_fields=alt_fields)
            used += len(details)
            if stable:
                rec_obj.status = "VALIDATED_HIGH_SIGNAL"
                self.memory.add_alpha(rec_obj)
                self.memory.touch_lineage(
                    rec_obj.expression,
                    rec_obj.lineage,
                    rec_obj.score,
                    round_no,
                    fields_used=rec_obj.fields_used,
                )
                print(
                    f"  [validated] {rec_obj.expression[:60]} "
                    f"survived perturbation checks"
                )
            else:
                self.memory.archive(
                    "unreproducible_high_signal", rec_obj.expression, round_no
                )
                print(
                    f"  [rejected] {rec_obj.expression[:60]} "
                    f"unreproducible under perturbation"
                )
        return used

    def _dedup_pool(self, round_no):
        if len(self.memory.submission_pool) < 2:
            return
        kept, dropped = deduplicate(self.memory.submission_pool)
        self.memory.submission_pool = kept
        for rec in dropped:
            self.memory.archive(
                "redundant_dup",
                f"{rec['expression']} removed by pool dedup",
                round_no,
            )

    def _prune_stale_lineages(self, round_no):
        """Stop deepening lineages that never improved after several attempts."""
        pruned = []
        remaining = []
        for item in self.memory.active_lineages:
            if (
                item.get("attempts", 0) >= self.max_deepen_per_lineage + 1
                and item.get("best_score", -1) < self.good_fitness
            ):
                pruned.append(item)
            else:
                remaining.append(item)
        self.memory.active_lineages = remaining
        for item in pruned:
            self.memory.archive(
                "stale_lineage",
                f"{item['expression']} gave up after {item.get('attempts')} attempts",
                round_no,
            )

    # ---- output ----

    def _print_experiment(self, exp):
        if exp.metrics:
            m = exp.metrics
            print(
                f"  [{exp.id}] {exp.expression[:70]} "
                f"sharpe={m.get('sharpe')} fitness={m.get('fitness')} "
                f"turnover={m.get('turnover')} passed={m.get('passed')}"
            )
        else:
            print(f"  [{exp.id}] {exp.expression[:70]} FAILED: {exp.error}")

    def _print_summary(self, summary, elapsed=None, validation_used=0):
        print(
            f"Round {summary['round']} verdicts: {summary['verdicts']} "
            f"pool={summary['pool_size']}"
        )
        if summary["best"]:
            b = summary["best"]
            print(
                f"  best={b['expression'][:70]} sharpe={b.get('sharpe')} "
                f"fitness={b.get('fitness')}"
            )
        if validation_used:
            print(f"  validation sims={validation_used}")
        if elapsed:
            print(f"  elapsed={elapsed:.1f}s")

    # ---- persistence ----

    def _load_state(self):
        self.memory.load()
        path = os.path.join(self.state_dir, "trajectory.json")
        if os.path.exists(path):
            with open(path) as f:
                self.trajectory = Trajectory.from_dict(json.load(f))

    def _save_state(self, state):
        os.makedirs(self.state_dir, exist_ok=True)
        traj_path = os.path.join(self.state_dir, "trajectory.json")
        with open(traj_path, "w") as f:
            json.dump(self.trajectory.to_dict(), f, indent=2)
        state_path = os.path.join(self.state_dir, f"round_{state.round_no}.json")
        with open(state_path, "w") as f:
            json.dump(state.to_dict(), f, indent=2)
