import json
import os
import re
import time
import uuid
from functools import lru_cache

from .diversity import select_diverse
from .state import atomic_write_json, score_of

PROMOTE_EVIDENCE = 3
SHORT_LESSON_MAX_AGE_ROUNDS = 3


def _evidence_key(experiment_id, polarity):
    """Stable idempotency key for one (experiment, polarity) evidence record."""
    return f"{experiment_id}|{polarity}"


class ExperienceMemory:
    """Three-tier research memory.

    Short-term:   next plans, active lineages, recent lessons (low evidence).
    Long-term:    lessons/avoid patterns supported by multiple simulations.
    Garbage:      repeated failures, superseded / unreproducible experience.
    """

    def __init__(
        self,
        state_dir=".wqb_state",
        max_lessons=15,
        max_avoid=25,
        max_next=12,
        max_garbage=30,
        max_active_lineages=10,
        max_pool=12,
        max_per_lineage=2,
        promote_evidence=PROMOTE_EVIDENCE,
    ):
        self.state_dir = state_dir
        self.max_lessons = max_lessons
        self.max_avoid = max_avoid
        self.max_next = max_next
        self.max_garbage = max_garbage
        self.max_active_lineages = max_active_lineages
        self.max_pool = max_pool
        self.max_per_lineage = max_per_lineage
        self.promote_evidence = promote_evidence

        self.current_best = None
        self.submission_pool = []  # list of AlphaRecord dicts
        self.lessons = []  # tier: "long" | "short"
        self.beliefs = []  # Claim -> {support, contradiction} -> confidence
        self._belief_index = {}  # belief_key -> belief (O(1) lookup)
        self.avoid = []
        self.next = []
        self.active_lineages = []  # {expression, lineage, attempts, best_score, last_round}
        self.garbage = []  # {kind, reason, round_no, created}
        self.recent_rounds = []  # compact round summaries
        self.updated_round = 0
        self.created_at = time.time()
        self.updated_at = time.time()
        self._ensure_dir()

    def _ensure_dir(self):
        os.makedirs(self.state_dir, exist_ok=True)

    def memory_path(self):
        return os.path.join(self.state_dir, "experience.json")

    def load(self):
        path = self.memory_path()
        if not os.path.exists(path):
            return self
        with open(path) as f:
            data = json.load(f)
        self.current_best = data.get("current_best")
        self.submission_pool = data.get("submission_pool", [])
        self.lessons = data.get("lessons", [])
        self.beliefs = data.get("beliefs", [])
        self.avoid = data.get("avoid", [])
        self.next = data.get("next", [])
        self.active_lineages = data.get("active_lineages", [])
        self.garbage = data.get("garbage", [])
        self.recent_rounds = data.get("recent_rounds", [])
        self.updated_round = data.get("updated_round", 0)
        self.created_at = data.get("created_at", time.time())
        self.updated_at = data.get("updated_at", time.time())
        for lesson in self.lessons:
            rounds = lesson.get("source_rounds") or []
            lesson["source_rounds"] = set(rounds)
            roots = lesson.get("lineage_roots") or []
            lesson["lineage_roots"] = set(roots)
        for belief in self.beliefs:
            belief["support_lineage_roots"] = set(
                belief.get("support_lineage_roots") or []
            )
            belief["contradiction_lineage_roots"] = set(
                belief.get("contradiction_lineage_roots") or []
            )
            belief["source_rounds"] = set(belief.get("source_rounds") or [])
            # Replay idempotency survives evidence_log truncation via a
            # dedicated set. Rebuild it for older state that predates it.
            evidence_ids = belief.get("evidence_ids")
            if not evidence_ids:
                evidence_ids = {
                    _evidence_key(e.get("experiment_id"), e.get("polarity"))
                    for e in belief.get("evidence_log", [])
                    if e.get("experiment_id")
                }
            belief["evidence_ids"] = set(evidence_ids)
        self._belief_index = {b.get("belief_key"): b for b in self.beliefs}
        return self

    def save(self):
        self.updated_at = time.time()
        data = {
            "schema_version": 1,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_best": self.current_best,
            "submission_pool": self.submission_pool,
            "lessons": self.lessons,
            "beliefs": self.beliefs,
            "avoid": self.avoid,
            "next": self.next,
            "active_lineages": self.active_lineages,
            "garbage": self.garbage,
            "recent_rounds": self.recent_rounds,
            "updated_round": self.updated_round,
        }
        atomic_write_json(self.memory_path(), data)

    # ---- lessons (long-term candidates) ----

    def add_lesson(self, claim, source_round, evidence, confidence=0.5, source=None):
        for lesson in self.lessons:
            if self._similar(lesson["claim"], claim, threshold=0.8):
                lesson["source_round"] = source_round
                lesson["evidence"] = lesson.get("evidence", 0) + evidence
                lesson["confidence"] = min(1.0, lesson.get("confidence", 0) + 0.15)
                lesson.setdefault("source_rounds", set()).add(source_round)
                self._append_evidence(lesson, source)
                if source:
                    lesson.setdefault("lineage_roots", set()).add(
                        self._lineage_root(source)
                    )
                lesson["tier"] = self._tier_for(lesson)
                lesson["updated"] = time.time()
                return lesson
        entry = {
            "id": uuid.uuid4().hex[:8],
            "claim": claim,
            "source_round": source_round,
            "source_rounds": {source_round},
            "lineage_roots": {self._lineage_root(source)} if source else set(),
            "evidence": evidence,
            "confidence": min(confidence, 1.0),
            "evidence_log": [],
            "tier": "short",
            "created": time.time(),
            "updated": time.time(),
        }
        self._append_evidence(entry, source)
        entry["tier"] = self._tier_for(entry)
        self.lessons.append(entry)
        return entry

    # ---- beliefs (Claim -> Supporting + Contradicting evidence -> Confidence) ----

    def get_belief(self, belief_key):
        return self._belief_index.get(belief_key)

    def record_evidence(
        self,
        belief_key,
        claim,
        polarity,
        source_round,
        source=None,
        kind=None,
    ):
        """Record one experiment's evidence about a research belief.

        polarity:
          "support"    - a valid RESEARCH result favors the claim
          "contradict" - a valid RESEARCH result opposes the claim
          "pending"    - observed but not strong enough to count yet (e.g. a
                         high-signal hit awaiting robustness validation)

        Invariants:
        - an experiment_id contributes to a claim at most once (replay-safe)
        - repeated confirmations along ONE lineage root add a single
          independent line of evidence, so confidence does not creep up from
          parameter micro-tuning on the same lineage
        - contradiction raises contradiction_count and lowers confidence;
          support never overwrites the contradiction history
        """
        belief = self._get_or_create_belief(belief_key, claim)
        exp_id = source.get("experiment_id") if source else None
        if exp_id and _evidence_key(exp_id, polarity) in belief["evidence_ids"]:
            return belief

        root = self._lineage_root(source) if source else None
        belief["source_rounds"].add(source_round)
        if polarity == "support":
            belief["support_count"] += 1
            if root:
                belief["support_lineage_roots"].add(root)
        elif polarity == "contradict":
            belief["contradiction_count"] += 1
            if root:
                belief["contradiction_lineage_roots"].add(root)

        if source:
            if exp_id:
                # Record the idempotency marker regardless of the truncated
                # evidence_log: a replay of this experiment must be skipped
                # even after its log entry has been evicted.
                belief["evidence_ids"].add(_evidence_key(exp_id, polarity))
            log = belief.setdefault("evidence_log", [])
            log.insert(
                0,
                {
                    "experiment_id": exp_id,
                    "polarity": polarity,
                    "kind": kind,
                    "round": source.get("round") or source_round,
                    "lineage_root": root,
                    "expression": source.get("expression"),
                    "fields": list(source.get("fields") or []),
                },
            )
            del log[30:]  # keep contradiction history, bound the log

        belief["confidence"] = self._belief_confidence(belief)
        belief["tier"] = self._belief_tier_for(belief)
        belief["updated"] = time.time()
        return belief

    def _get_or_create_belief(self, belief_key, claim):
        belief = self.get_belief(belief_key)
        if belief is not None:
            return belief
        entry = {
            "id": uuid.uuid4().hex[:8],
            "belief_key": belief_key,
            "claim": claim,
            "support_count": 0,
            "contradiction_count": 0,
            "support_lineage_roots": set(),
            "contradiction_lineage_roots": set(),
            "source_rounds": set(),
            "evidence_log": [],
            "evidence_ids": set(),
            "confidence": 0.5,
            "tier": "short",
            "created": time.time(),
            "updated": time.time(),
        }
        self.beliefs.append(entry)
        self._belief_index[belief_key] = entry
        return entry

    @staticmethod
    def _belief_confidence(belief):
        """Transparent confidence: an unknown claim sits at 0.5; each
        independent supporting lineage adds +0.2, each independent
        contradicting lineage subtracts 0.2."""
        s = len(belief["support_lineage_roots"])
        c = len(belief["contradiction_lineage_roots"])
        return max(0.05, min(0.95, 0.5 + 0.2 * (s - c)))

    def _belief_tier_for(self, belief):
        """Long-term requires enough independent support across rounds AND no
        contradiction as strong as the support. Strong contradiction demotes a
        belief back to short-term."""
        s = len(belief["support_lineage_roots"])
        c = len(belief["contradiction_lineage_roots"])
        rounds = belief.get("source_rounds") or set()
        if (
            s >= self.promote_evidence
            and len(rounds) >= 2
            and s >= 2
            and not (c >= s)
        ):
            return "long"
        return "short"

    def add_avoid(self, direction, reason, source_round, source=None):
        for item in self.avoid:
            if item["direction"] == direction:
                item["support_count"] = item.get("support_count", 0) + 1
                item["reason"] = reason
                item["source_round"] = source_round
                item["last_seen"] = source_round
                item["updated"] = time.time()
                if source:
                    log = item.setdefault("evidence_log", [])
                    log.insert(0, dict(source))
                    del log[5:]
                return item
        entry = {
            "id": uuid.uuid4().hex[:8],
            "direction": direction,
            "reason": reason,
            "source_round": source_round,
            "support_count": 1,
            "last_seen": source_round,
            "evidence_log": [dict(source)] if source else [],
            "created": time.time(),
            "updated": time.time(),
        }
        self.avoid.append(entry)
        return entry

    @staticmethod
    def _lineage_root(source):
        """The research direction an evidence item genuinely starts from: the
        deepest ancestor for a deepened lineage, the expression itself for a
        fresh exploration. Repeated confirmations along one root are not
        independent evidence."""
        if not source:
            return None
        lineage = source.get("lineage") or []
        if lineage:
            return lineage[-1]
        return source.get("expression") or source.get("experiment_id")

    def _append_evidence(self, lesson, source, max_log=5):
        """Record the real experiment behind an evidence increment."""
        if not source:
            return
        log = lesson.setdefault("evidence_log", [])
        log.insert(0, dict(source))
        del log[max_log:]

    def _tier_for(self, lesson):
        """Long-term requires accumulated evidence AND confirmation across
        independent research lineages AND distinct rounds — repeated hits along
        one lineage, even spread over several rounds, are a single line of
        evidence and must not be promoted on their own."""
        evidence = lesson.get("evidence", 0)
        rounds = lesson.get("source_rounds") or set()
        roots = lesson.get("lineage_roots") or set()
        if (
            evidence >= self.promote_evidence
            and len(rounds) >= 2
            and len(roots) >= 2
        ):
            return "long"
        return "short"

    def add_next(self, idea, priority, source, round_no):
        for item in self.next:
            if item["idea"] == idea:
                item["priority"] = max(item["priority"], priority)
                item["source"] = source
                item["round"] = round_no
                item["updated"] = time.time()
                return item
        self.next.append(
            {
                "id": uuid.uuid4().hex[:8],
                "idea": idea,
                "priority": priority,
                "source": source,
                "round": round_no,
                "created": time.time(),
                "updated": time.time(),
            }
        )
        return self.next[-1]

    def is_avoided(self, direction):
        return any(item["direction"] == direction for item in self.avoid)

    # ---- garbage archive ----

    def archive(self, kind, reason, round_no):
        for item in self.garbage:
            if item["kind"] == kind and item["reason"] == reason:
                item["round_no"] = round_no
                item["updated"] = time.time()
                return item
        entry = {
            "id": uuid.uuid4().hex[:8],
            "kind": kind,
            "reason": reason,
            "round_no": round_no,
            "created": time.time(),
            "updated": time.time(),
        }
        self.garbage.append(entry)
        return entry

    # ---- submission pool ----

    def add_alpha(self, record):
        for i, rec in enumerate(self.submission_pool):
            if rec["expression"] == record.expression:
                if record.score > score_of(rec.get("metrics")):
                    self.submission_pool[i] = record.to_dict()
                return self.submission_pool[i]
        self.submission_pool.append(record.to_dict())
        self._trim_pool()
        return self.submission_pool[-1]

    def replace_alpha(self, record, keeper_id):
        self.remove_alpha(keeper_id)
        return self.add_alpha(record)

    def remove_alpha(self, alpha_id):
        self.submission_pool = [
            r for r in self.submission_pool if r["id"] != alpha_id
        ]

    def _trim_pool(self):
        """Keep the pool at max_pool via diversified greedy selection, not a
        bare score cutoff: per-lineage caps and structural coverage prevent one
        lineage / dataset / operator family from filling the pool."""
        if len(self.submission_pool) <= self.max_pool:
            return
        self.submission_pool = select_diverse(
            self.submission_pool,
            self.max_pool,
            max_per_lineage=self.max_per_lineage,
        )

    # ---- active lineages (deepening targets) ----

    def touch_lineage(self, expression, lineage, score, round_no, fields_used=None, hypothesis_id=None):
        lineage = list(lineage or [])
        fields_used = list(fields_used or [])
        for item in self.active_lineages:
            if item["expression"] == expression:
                item["attempts"] = item.get("attempts", 0) + 1
                item["best_score"] = max(item.get("best_score", -1), score)
                item["last_round"] = round_no
                if fields_used:
                    item["fields_used"] = fields_used
                if hypothesis_id:
                    item["hypothesis_id"] = hypothesis_id
                item["updated"] = time.time()
                return item
        entry = {
            "id": uuid.uuid4().hex[:8],
            "expression": expression,
            "lineage": lineage,
            "fields_used": fields_used,
            "attempts": 1,
            "best_score": score,
            "last_round": round_no,
            "hypothesis_id": hypothesis_id,
            "created": time.time(),
            "updated": time.time(),
        }
        self.active_lineages.append(entry)
        self._trim_lineages()
        return entry

    def lineage_attempts(self, expression):
        for item in self.active_lineages:
            if item["expression"] == expression:
                return item.get("attempts", 0)
        return 0

    def deepening_targets(self, max_deepen_per_lineage, limit=4, max_per_lineage=2):
        """Deepening targets, best_score first but capped per lineage root so
        the deepening budget is not monopolized by one research family."""
        candidates = [
            item
            for item in self.active_lineages
            if item.get("attempts", 0) < max_deepen_per_lineage
        ]
        candidates.sort(key=lambda x: -x.get("best_score", -1))
        selected = []
        root_counts = {}
        for item in candidates:
            root = self._lineage_root(item)
            if root and root_counts.get(root, 0) >= max_per_lineage:
                continue
            selected.append(item)
            if root:
                root_counts[root] = root_counts.get(root, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    def _trim_lineages(self):
        if len(self.active_lineages) <= self.max_active_lineages:
            return
        self.active_lineages.sort(key=lambda x: -x.get("best_score", -1))
        self.active_lineages = self.active_lineages[: self.max_active_lineages]

    # ---- current best (kept for compatibility / quick reference) ----

    def set_current_best(self, experiment):
        self.current_best = experiment.to_dict()

    # ---- compression (periodic short-term maintenance) ----

    def compress(self):
        merged = []
        for lesson in sorted(
            self.lessons, key=lambda x: -x.get("evidence", 0)
        ):
            if any(
                self._similar(lesson["claim"], m["claim"], threshold=0.75)
                for m in merged
            ):
                continue
            merged.append(lesson)

        kept = []
        for lesson in merged:
            if lesson.get("tier") == "short" and (
                self.updated_round - lesson.get("source_round", 0)
                > SHORT_LESSON_MAX_AGE_ROUNDS
            ):
                self.archive(
                    "stale_short_lesson",
                    lesson["claim"][:100],
                    self.updated_round,
                )
                continue
            kept.append(lesson)

        kept.sort(key=lambda x: (x.get("tier") == "long", -x.get("evidence", 0)), reverse=True)
        self.lessons = kept[: self.max_lessons]

        self.avoid = sorted(self.avoid, key=lambda x: -x.get("updated", 0))[
            : self.max_avoid
        ]
        self.next = sorted(self.next, key=lambda x: -x.get("priority", 0))[
            : self.max_next
        ]
        self.garbage = sorted(self.garbage, key=lambda x: -x.get("updated", 0))[
            : self.max_garbage
        ]
        self._trim_pool()
        self._trim_lineages()

    @staticmethod
    def _similar(a, b, threshold=0.8):
        ta = _tokens(a)
        tb = _tokens(b)
        if not ta or not tb:
            return False
        inter = len(set(ta) & set(tb))
        union = len(set(ta) | set(tb))
        return inter / union >= threshold


@lru_cache(maxsize=256)
def _tokens(text):
    return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 2]
