import json
import logging
import os
import time

from .beliefs import belief_claim, belief_identity
from .candidate import CandidateBuilder
from .discovery import FieldDiscovery
from .diversity import (
    deduplicate,
    filter_candidates,
    pool_diversity_summary,
)
from .failures import classify_error, is_research_relevant
from .hypothesis import validate_contract
from .memory import ExperienceMemory
from .reflection import Reflector
from .scheduler import BacktestScheduler
from .simulator import Simulator
from .state import (
    AlphaRecord,
    Experiment,
    ResearchState,
    Trajectory,
    atomic_write_json,
    score_of,
)
from .validation import HighSignalValidator

logger = logging.getLogger("wqb.agent")

SEED_HYPOTHESES = [
    {
        "id": "h-seed-reversal",
        "statement": "Short-term return reversal: stocks that rose sharply over the last 5 days tend to revert in the near term.",
        "research_question": "Does the raw short-window price-change signal revert with the expected negative sign?",
        "tags": ["reversal", "return", "price", "short-term"],
        "direction": "reversal",
        "datasets": ["pv1", "pv13"],
        "economic_intuition": (
            "Short-horizon liquidity demand and temporary buying/selling "
            "pressure push prices past fair value; when the pressure abates "
            "the price mean-reverts."
        ),
        "expected_mechanism": "temporary price pressure",
        "field_semantics": {
            "primary": {
                "concept": "short_term_price_change",
                "description": (
                    "recent short-window price change (short-horizon return) "
                    "proxying temporary price pressure"
                ),
            },
            "secondary": [],
        },
        "expected_direction": {
            "sign": "negative",
            "note": "higher recent price change -> lower expected forward return",
        },
        "expected_horizon_days": 5,
        "failure_condition": [
            "related short-window price fields consistently fail across lineages",
            "the sign is positive across independent lineages",
            "the effect disappears after basic neutralization",
            "reasonable horizon neighbors (10d/20d) collapse the signal",
        ],
    },
    {
        "id": "h-seed-analyst",
        "statement": "Analyst target-price revisions upward predict short-term outperformance.",
        "research_question": "Does a rise in analyst target price carry the expected positive forward-return sign?",
        "tags": ["analyst", "forecast", "revision", "target"],
        "direction": "long",
        "datasets": ["analyst4"],
        "economic_intuition": (
            "Analysts revise target prices when new information arrives; the "
            "market underreacts temporarily and prices drift toward the "
            "revised fair value."
        ),
        "expected_mechanism": "analyst information revision",
        "field_semantics": {
            "primary": {
                "concept": "target_price_revision",
                "description": "change in analyst target price or recommendation level",
            },
            "secondary": [
                {
                    "concept": "estimate_revision",
                    "description": "change in analyst EPS estimate",
                },
            ],
        },
        "expected_direction": {
            "sign": "positive",
            "note": "upward revision -> higher expected forward return",
        },
        "expected_horizon_days": 10,
        "failure_condition": [
            "revision fields show the opposite sign across lineages",
            "the signal vanishes after subindustry neutralization",
            "level fields behave identically to revision fields (no revision content)",
        ],
    },
    {
        "id": "h-seed-option",
        "statement": "Stocks with elevated implied volatility earn lower forward returns.",
        "research_question": "Does elevated implied volatility predict lower forward returns (risk premium)?",
        "tags": ["option", "volatility", "implied", "risk"],
        "direction": "reversal",
        "datasets": ["option8", "option9"],
        "economic_intuition": (
            "Option-implied volatility is high when the market demands "
            "compensation for uncertainty; high-volatility names subsequently "
            "earn lower forward returns."
        ),
        "expected_mechanism": "risk-premium compensation",
        "field_semantics": {
            "primary": {
                "concept": "implied_volatility",
                "description": "option-implied volatility level or skew",
            },
            "secondary": [],
        },
        "expected_direction": {
            "sign": "negative",
            "note": "higher implied volatility -> lower expected forward return",
        },
        "expected_horizon_days": 20,
        "failure_condition": [
            "implied-vol fields show positive sign across lineages",
            "the effect disappears after volatility neutralization",
            "short-horizon neighbors invert the sign",
        ],
    },
    {
        "id": "h-seed-model",
        "statement": "High model risk scores predict lower forward returns.",
        "research_question": "Do high composite model risk scores predict lower forward returns?",
        "tags": ["model", "score", "risk", "composite"],
        "direction": "reversal",
        "datasets": ["model16", "model51"],
        "economic_intuition": (
            "Composite model risk scores aggregate exposure to hard-to-price "
            "uncertainty; risk-loaded names earn lower forward returns in "
            "aggregate."
        ),
        "expected_mechanism": "risk-premium compensation",
        "field_semantics": {
            "primary": {
                "concept": "model_risk_score",
                "description": "composite model risk or factor-loading score",
            },
            "secondary": [],
        },
        "expected_direction": {
            "sign": "negative",
            "note": "higher risk score -> lower expected forward return",
        },
        "expected_horizon_days": 20,
        "failure_condition": [
            "risk-score fields predict positive returns across lineages",
            "the effect is explained away by sector exposure",
            "horizon neighbors collapse the signal",
        ],
    },
    {
        "id": "h-seed-news",
        "statement": "Positive news sentiment predicts short-term positive returns.",
        "research_question": "Does positive news sentiment predict positive forward returns over the near term?",
        "tags": ["news", "sentiment", "positive"],
        "direction": "long",
        "datasets": ["news12", "news18"],
        "economic_intuition": (
            "Attention-grabbing positive news is incompletely priced at the "
            "close, so sentiment continues to drift up over the following days."
        ),
        "expected_mechanism": "sentiment continuation",
        "field_semantics": {
            "primary": {
                "concept": "news_sentiment",
                "description": "aggregate news sentiment or headline tone score",
            },
            "secondary": [
                {
                    "concept": "news_attention",
                    "description": "news volume or headline buzz",
                },
            ],
        },
        "expected_direction": {
            "sign": "positive",
            "note": "more positive sentiment -> higher forward return",
        },
        "expected_horizon_days": 5,
        "failure_condition": [
            "sentiment fields show negative sign across lineages",
            "the effect reverses after basic neutralization",
            "a 10d horizon neighbor collapses the continuation",
        ],
    },
    {
        "id": "h-seed-fundamental",
        "statement": "Firms with strong earnings growth continue to outperform.",
        "research_question": "Does strong earnings growth predict continued outperformance?",
        "tags": ["fundamental", "growth", "earning"],
        "direction": "long",
        "datasets": ["fundamental2", "fundamental6"],
        "economic_intuition": (
            "Earnings information diffuses gradually; firms with accelerating "
            "earnings keep outperforming as the information is impounded."
        ),
        "expected_mechanism": "earnings information diffusion",
        "field_semantics": {
            "primary": {
                "concept": "earnings_growth",
                "description": "earnings or revenue growth level and its recent change",
            },
            "secondary": [],
        },
        "expected_direction": {
            "sign": "positive",
            "note": "higher growth -> higher expected forward return",
        },
        "expected_horizon_days": 60,
        "failure_condition": [
            "growth fields predict negative returns across lineages",
            "the effect is fully neutralized by value/size exposure",
            "quarterly changes show no predictive content",
        ],
    },
]

# Lookup index for belief derivation; _hypothesis_of hits it instead of
# scanning the whole list for every validated record.
_HYPOTHESES_BY_ID = {h["id"]: h for h in SEED_HYPOTHESES}


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
            cache_path=os.path.join(self.state_dir, "fields_cache.json"),
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
            # A round whose ResearchState was saved is fully finished; a crash
            # before that point is resumed via the round's job checkpoint.
            if os.path.exists(os.path.join(self.state_dir, f"round_{round_no}.json")):
                logger.info("ROUND_SKIPPED round=%d reason=already-completed",
                            round_no)
                continue
            self.run_one_round(round_no)

    def run_one_round(self, round_no):
        start = time.time()
        self.discovery.reset_budget(self.discovery_budget_per_round)
        logger.info("ROUND_STARTED round=%d sim_budget=%d validation_budget=%d",
                    round_no, self.sim_budget_per_round, self.validation_budget)
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
                hypothesis,
                target_count=self.fields_per_discovery,
                require_field_semantics=True,
            )
            outcome = getattr(self.discovery, "last_outcome", None)
            if not fields and outcome and outcome.get("infra_failure"):
                # Field Discovery API failure (timeout / 429 / auth / 5xx)
                # carries no research signal. Skipping the round here must
                # never be read as "the hypothesis has no matching fields".
                logger.info(
                    "ROUND_SKIPPED round=%d reason=field-discovery-infra-failure",
                    round_no,
                )
                print(
                    "Field Discovery API failed; skipping round "
                    "(not a research conclusion)."
                )
                return None
            if not fields:
                # Fall back to previously discovered fields (cache reuse).
                fields = self._last_fields
            if not fields:
                logger.info("ROUND_SKIPPED round=%d reason=no-fields", round_no)
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
                    fields, active, split["deepen"], hypothesis=hypothesis
                )

        if not candidates:
            logger.info("ROUND_SKIPPED round=%d reason=no-candidates", round_no)
            print("No candidates built; skipping round.")
            return None

        # Early redundancy filter: reject candidates that are duplicate /
        # redundant with already-simulated work or the submission pool before
        # any simulation budget is spent. Research mutations (single-variable
        # robustness probes) are still allowed through.
        candidates, blocked = filter_candidates(
            candidates,
            self._simulated_expressions(),
            self.memory.submission_pool,
        )
        for cand, reason in blocked:
            logger.info(
                "CANDIDATE_BLOCKED round=%d reason=%s expr=%s",
                round_no, reason, (cand.get("expression") or "")[:70],
            )
        if not candidates:
            logger.info(
                "ROUND_SKIPPED round=%d reason=all-candidates-redundant",
                round_no,
            )
            print("All candidates redundant with existing work; skipping round.")
            return None

        experiments = self._build_experiments(round_no, hypothesis, fields, candidates)
        scheduler = self._new_scheduler(round_no)
        scheduler.add_jobs(experiments)
        scheduler.run()

        for exp in experiments:
            self.trajectory.add(exp)
            self._print_experiment(exp)

        summary = self.reflector.reflect(round_no, hypothesis, experiments)

        validation_used = self._validate_suspicious(scheduler, summary, round_no)
        self._dedup_pool(round_no)
        self._prune_stale_lineages(round_no)
        summary["diversity"] = pool_diversity_summary(
            self.memory.submission_pool
        )

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
        logger.info(
            "ROUND_FINISHED round=%d verdicts=%s pool=%d validation_sims=%d "
            "elapsed=%.1fs",
            round_no, summary["verdicts"], summary["pool_size"],
            validation_used, time.time() - start,
        )
        return summary

    # ---- experiments & scheduling ----

    def _simulated_expressions(self):
        """Every expression that has ever been submitted to the simulator in
        this session; a duplicate simulation is pure budget waste."""
        return {e.expression for e in self.trajectory.experiments}

    def _build_experiments(self, round_no, hypothesis, fields, candidates):
        experiments = []
        for c in candidates[: self.sim_budget_per_round]:
            exp_hypothesis_id = c.get("hypothesis_id") or hypothesis["id"]
            exp_hypothesis = self._hypothesis_of(exp_hypothesis_id) or hypothesis
            exp = Experiment(
                round_no,
                exp_hypothesis_id,
                c["expression"],
                self.simulation_settings,
                c.get("fields_used") or [f["id"] for f in fields],
                datasets=exp_hypothesis.get("datasets", []),
                lineage=c.get("lineage", []),
            )
            exp.mutation = c.get("mutation")
            exp.rationale = c.get("rationale")
            exp.research_question = c.get("research_question")
            exp.mechanism = c.get("mechanism")
            experiments.append(exp)
        return experiments

    def _new_scheduler(self, round_no):
        checkpoint_path = os.path.join(self.state_dir, f"round_{round_no}_jobs.json")
        return BacktestScheduler(
            self.client,
            self.simulator,
            max_concurrent=self.max_concurrent_sims,
            budget=self.sim_budget_per_round + self.validation_budget,
            poll_timeout_sec=self.poll_timeout_sec,
            checkpoint_path=checkpoint_path,
            checkpoint_every=1,
        )

    # ---- planning (memory-driven) ----

    def _plan_round(self, round_no):
        n = self.candidates_per_round
        pool = self.memory.submission_pool
        active = self.memory.active_lineages

        explore = max(1, int(round(n * self.explore_ratio)))
        deepen = n - explore
        if not pool:
            # Not enough proven directions yet: bias toward exploration.
            deepen = max(0, deepen - 1)
            explore = n - deepen

        # Lineage quality: a proven lineage steers budget toward deepening.
        if active and pool:
            best_lineage = max(
                (a.get("best_score", -1) for a in active), default=-1
            )
            if best_lineage >= self.good_fitness and deepen < n:
                deepen += 1
                explore = max(0, n - deepen)

        hypothesis = self._choose_hypothesis(round_no)
        # A research contract is mandatory before any candidate is built or
        # simulated: the system must already know the mechanism, the field
        # meaning, the expected sign and horizon, and the failure condition.
        validate_contract(hypothesis)
        return {
            "hypothesis": hypothesis,
            "split": {"explore": explore, "deepen": deepen},
        }

    def _choose_hypothesis(self, round_no):
        used, failure_counts, last_used, parked = self._hypothesis_stats()
        available = [h for h in SEED_HYPOTHESES if h["id"] not in parked]
        unused = [h for h in available if h["id"] not in used]
        if unused:
            candidates = unused
        else:
            candidates = sorted(
                available,
                key=lambda h: (
                    failure_counts.get(h["id"], 0),
                    last_used.get(h["id"], 0),
                ),
            )
        if not candidates:
            candidates = list(SEED_HYPOTHESES)
        return dict(candidates[0])

    def _hypothesis_stats(self):
        """One traversal of the trajectory producing every statistic the
        hypothesis-selection logic needs: used hypothesis ids, per-hypothesis
        research-failure counts, last-used round, and parked hypotheses.

        The selection helpers below are thin wrappers around it so tests keep
        calling the named accessors while a single pass serves the planner.
        """
        used = set()
        failure_counts = {}
        last_used = {}
        by_hypothesis = {}
        for exp in self.trajectory.experiments:
            hid = exp.hypothesis_id
            if hid:
                used.add(hid)
                last_used[hid] = max(last_used.get(hid, 0), exp.round or 0)
            if exp.status == "FAILED" and is_research_relevant(
                classify_error(exp.error)
            ):
                failure_counts[hid] = failure_counts.get(hid, 0) + 1
            by_hypothesis.setdefault(hid, []).append(exp)
        parked = set()
        for hid, exps in by_hypothesis.items():
            research_fails = [
                e for e in exps[-2:]
                if e.status == "FAILED"
                and is_research_relevant(classify_error(e.error))
            ]
            if len(research_fails) >= 2:
                parked.add(hid)
        return used, failure_counts, last_used, parked

    def _parked_hypotheses(self):
        """A hypothesis whose last two experiments both failed for
        research-level reasons is parked until evidence suggests revisiting it.
        Infrastructure failures (timeout / auth / rate-limit / 5xx) carry no
        research signal and must not pause a direction."""
        return self._hypothesis_stats()[3]

    def _hypothesis_failure_counts(self):
        return self._hypothesis_stats()[1]

    def _used_hypothesis_ids(self):
        return self._hypothesis_stats()[0]

    def _last_used_round(self, hypothesis_id):
        return self._hypothesis_stats()[2].get(hypothesis_id, 0)

    # ---- validation & pool maintenance ----

    def _validate_suspicious(self, scheduler, summary, round_no):
        suspicious = summary.get("suspicious") or []
        if not suspicious:
            return 0
        alt_fields = self._last_fields
        # Collect every perturbation job first, then run them in a single
        # scheduler pass. Per-record serial add_jobs+run() underutilized the
        # worker pool (one thread pool spin-up + checkpoint round per record);
        # one run lets concurrent validation jobs share the max_concurrent
        # workers. The verdict is still decided per record afterwards.
        groups = []
        validation_used = 0
        for rec in suspicious:
            if validation_used >= self.validation_budget:
                break
            jobs, _ = self.validator.build_perturbation_jobs(
                rec, alt_fields=alt_fields
            )
            if not jobs:
                continue
            take = jobs[: self.validation_budget - validation_used]
            if not take:
                break
            groups.append((rec, take))
            validation_used += len(take)
        if groups:
            scheduler.add_jobs(
                [job for _, take in groups for job in take]
            )
            scheduler.run()
            for rec, take in groups:
                results = [self._perturb_result(exp) for exp in take]
                stable, _ = self.validator.decide(rec, results)
                self._apply_validation(rec, stable, round_no, len(take))
        return validation_used

    def _apply_validation(self, rec, stable, round_no, sims):
        rec_obj = AlphaRecord.from_dict(rec)
        bkey = self._belief_key_of(rec)
        bclaim = self._belief_claim_of(rec)
        if stable:
            rec_obj.status = AlphaRecord.STATUS_VALIDATED
            self.memory.add_alpha(rec_obj)
            self.memory.touch_lineage(
                rec_obj.expression,
                rec_obj.lineage,
                rec_obj.score,
                round_no,
                fields_used=rec_obj.fields_used,
            )
            # Robustness validation confirms the high-signal hit: it becomes
            # strong support for the belief on these fields.
            self.memory.record_evidence(
                bkey,
                bclaim,
                "support",
                round_no,
                source=self._validation_source(rec, round_no),
                kind="validated_high_signal",
            )
            logger.info(
                "VALIDATION_FINISHED round=%d expression=%s stable=True sims=%d",
                round_no, rec_obj.expression[:60], sims,
            )
            print(
                f"  [validated] {rec_obj.expression[:60]} "
                f"survived perturbation checks"
            )
        else:
            self.memory.archive(
                "unreproducible_high_signal", rec_obj.expression, round_no
            )
            # Validation failure is robustness-negative evidence: the belief
            # that these fields carry a high signal is contradicted, not just
            # dropped into garbage.
            self.memory.record_evidence(
                bkey,
                bclaim,
                "contradict",
                round_no,
                source=self._validation_source(rec, round_no),
                kind="validation_failure",
            )
            logger.info(
                "VALIDATION_FINISHED round=%d expression=%s stable=False sims=%d",
                round_no, rec_obj.expression[:60], sims,
            )
            print(
                f"  [rejected] {rec_obj.expression[:60]} "
                f"unreproducible under perturbation"
            )

    @staticmethod
    def _hypothesis_of(hypothesis_id):
        """Look up the hypothesis dict by id for belief derivation. Only the
        id is looked up here; the belief key itself is built by the single
        canonical helper in `beliefs.py`, shared with the Reflector."""
        return _HYPOTHESES_BY_ID.get(hypothesis_id)

    def _belief_key_of(self, rec):
        return belief_identity(
            rec.get("hypothesis_id"),
            rec.get("fields_used"),
            hypothesis=self._hypothesis_of(rec.get("hypothesis_id")),
        )

    def _belief_claim_of(self, rec):
        return belief_claim(
            rec.get("hypothesis_id"),
            rec.get("fields_used"),
            hypothesis=self._hypothesis_of(rec.get("hypothesis_id")),
        )

    @staticmethod
    def _validation_source(rec, round_no):
        return {
            "experiment_id": f"{rec.get('id')}-validation",
            "round": round_no,
            "expression": rec.get("expression"),
            "lineage": list(rec.get("lineage") or []),
            "fields": list(rec.get("fields_used") or []),
            "mutation": "validation",
        }

    @staticmethod
    def _perturb_result(exp):
        metrics = exp.metrics or {}
        return {
            "expression": exp.expression,
            "score": score_of(metrics) if metrics else -1.0,
            "sharpe": metrics.get("sharpe"),
            "turnover": metrics.get("turnover"),
            "checks_passed": metrics.get("passed") is True,
            "error": exp.error,
        }

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
        diversity = summary.get("diversity") or {}
        if diversity:
            print(
                "Pool diversity: hypothesis={} dataset_family={} "
                "operator_family={} lineage_root={}".format(
                    len(diversity.get("hypothesis_id") or {}),
                    len(diversity.get("dataset_family") or {}),
                    len(diversity.get("operator_family") or {}),
                    len(diversity.get("lineage_root") or {}),
                )
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
        atomic_write_json(
            os.path.join(self.state_dir, "trajectory.json"),
            self.trajectory.to_dict(),
        )
        atomic_write_json(
            os.path.join(self.state_dir, f"round_{state.round_no}.json"),
            state.to_dict(),
        )
