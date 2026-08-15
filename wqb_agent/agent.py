import json
import os
import time

from .candidate import CandidateBuilder
from .client import WQBClient
from .discovery import FieldDiscovery
from .memory import ExperienceMemory
from .reflection import Reflector
from .simulator import Simulator
from .state import Experiment, ResearchState, Trajectory

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

        self.memory = ExperienceMemory(self.state_dir)
        self.trajectory = Trajectory()
        self.builder = CandidateBuilder(
            neutralization=self.simulation_settings.get("neutralization", "SUBINDUSTRY")
        )
        self.discovery = FieldDiscovery(
            self.client,
            pagination_limit=self.pagination_limit,
            max_pages=self.max_pagination_pages,
        )
        self.simulator = Simulator(
            self.client,
            max_concurrent=agent_cfg.get("max_concurrent_sims", 3),
            poll_timeout_sec=self.poll_timeout_sec,
        )
        self.reflector = Reflector(self.memory)

    def run(self, max_rounds=None):
        rounds = max_rounds or self.max_rounds
        self._load_state()
        for round_no in range(1, rounds + 1):
            self.run_one_round(round_no)

    def run_one_round(self, round_no):
        start = time.time()
        hypothesis = self._form_hypothesis(round_no)
        print(f"\n=== Round {round_no} ===")
        print(f"Hypothesis: {hypothesis['statement']}")

        fields = self.discovery.discover(
            hypothesis, target_count=self.fields_per_discovery
        )
        if not fields:
            print("No fields discovered; skipping round.")
            self._remember_direction(round_no, hypothesis)
            return None
        print(f"Fields ({len(fields)}): {[f['id'] for f in fields]}")

        candidates = self.builder.build(
            hypothesis, fields, self.memory.current_best, self.candidates_per_round
        )
        experiments = [
            Experiment(
                round_no,
                hypothesis["id"],
                c["expression"],
                self.simulation_settings,
                [f["id"] for f in fields],
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
        self._remember_direction(round_no, hypothesis)
        state = ResearchState(
            round_no=round_no,
            hypothesis=hypothesis,
            dataset=summary.get("datasets"),
            fields_used=[f["id"] for f in fields],
        )
        self._save_state(state)
        self._print_summary(summary, elapsed=time.time() - start)
        return summary

    def _form_hypothesis(self, round_no):
        next_ideas = self.memory.top_next(1)
        if next_ideas and self.memory.current_best:
            best = self.memory.current_best
            metrics = best.get("metrics") or {}
            direction = "reversal" if (metrics.get("sharpe") or 0) < 0 else "long"
            fields = best.get("fields_used") or []
            tags = ["iterate", "best"]
            if fields:
                tags.append(fields[0])
            return {
                "id": f"h-iter-r{round_no}",
                "statement": next_ideas[0]["idea"],
                "tags": tags,
                "direction": direction,
            }
        used = self._used_hypothesis_ids()
        for seed in SEED_HYPOTHESES:
            if seed["id"] not in used:
                return dict(seed)
        return dict(SEED_HYPOTHESES[round_no % len(SEED_HYPOTHESES)])

    def _used_hypothesis_ids(self):
        return {e.hypothesis_id for e in self.trajectory.experiments}

    def _remember_direction(self, round_no, hypothesis):
        self.memory.add_avoid(
            f"hypothesis:{hypothesis['id']}",
            "seed direction attempted this round",
            round_no,
        )

    def _attach_candidate_meta(self, experiment, candidate):
        experiment.hypothesis_id = candidate.get("parent") or experiment.hypothesis_id
        experiment.mutation = candidate.get("mutation")
        experiment.rationale = candidate.get("rationale")

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

    def _print_summary(self, summary, elapsed=None):
        print(
            f"Round {summary['round']} verdicts: {summary['verdicts']}"
        )
        if summary["best"]:
            b = summary["best"]
            print(
                f"  best={b['expression'][:70]} sharpe={b.get('sharpe')} "
                f"fitness={b.get('fitness')}"
            )
        if elapsed:
            print(f"  elapsed={elapsed:.1f}s")

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
