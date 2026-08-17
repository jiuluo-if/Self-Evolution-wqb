import json
import os
import re
import time
import uuid

from .state import score_of

PROMOTE_EVIDENCE = 3
SHORT_LESSON_MAX_AGE_ROUNDS = 3


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
        promote_evidence=PROMOTE_EVIDENCE,
    ):
        self.state_dir = state_dir
        self.max_lessons = max_lessons
        self.max_avoid = max_avoid
        self.max_next = max_next
        self.max_garbage = max_garbage
        self.max_active_lineages = max_active_lineages
        self.max_pool = max_pool
        self.promote_evidence = promote_evidence

        self.current_best = None
        self.submission_pool = []  # list of AlphaRecord dicts
        self.lessons = []  # tier: "long" | "short"
        self.avoid = []
        self.next = []
        self.active_lineages = []  # {expression, lineage, attempts, best_score, last_round}
        self.garbage = []  # {kind, reason, round_no, created}
        self.recent_rounds = []  # compact round summaries
        self.updated_round = 0
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
        self.avoid = data.get("avoid", [])
        self.next = data.get("next", [])
        self.active_lineages = data.get("active_lineages", [])
        self.garbage = data.get("garbage", [])
        self.recent_rounds = data.get("recent_rounds", [])
        self.updated_round = data.get("updated_round", 0)
        return self

    def save(self):
        data = {
            "current_best": self.current_best,
            "submission_pool": self.submission_pool,
            "lessons": self.lessons,
            "avoid": self.avoid,
            "next": self.next,
            "active_lineages": self.active_lineages,
            "garbage": self.garbage,
            "recent_rounds": self.recent_rounds,
            "updated_round": self.updated_round,
        }
        path = self.memory_path()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    # ---- lessons (long-term candidates) ----

    def add_lesson(self, claim, source_round, evidence, confidence=0.5):
        for lesson in self.lessons:
            if self._similar(lesson["claim"], claim, threshold=0.8):
                lesson["source_round"] = source_round
                lesson["evidence"] = lesson.get("evidence", 0) + evidence
                lesson["confidence"] = min(1.0, lesson.get("confidence", 0) + 0.15)
                lesson["tier"] = (
                    "long"
                    if lesson["evidence"] >= self.promote_evidence
                    else "short"
                )
                lesson["updated"] = time.time()
                return lesson
        entry = {
            "id": uuid.uuid4().hex[:8],
            "claim": claim,
            "source_round": source_round,
            "evidence": evidence,
            "confidence": min(confidence, 1.0),
            "tier": "long" if evidence >= self.promote_evidence else "short",
            "created": time.time(),
            "updated": time.time(),
        }
        self.lessons.append(entry)
        return entry

    def add_avoid(self, direction, reason, source_round):
        for item in self.avoid:
            if item["direction"] == direction:
                item["reason"] = reason
                item["source_round"] = source_round
                item["updated"] = time.time()
                return item
        entry = {
            "id": uuid.uuid4().hex[:8],
            "direction": direction,
            "reason": reason,
            "source_round": source_round,
            "created": time.time(),
            "updated": time.time(),
        }
        self.avoid.append(entry)
        return entry

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

    def top_next(self, n=5):
        ordered = sorted(self.next, key=lambda x: -x.get("priority", 0))
        return ordered[:n]

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

    def update_alpha_status(self, alpha_id, status):
        for rec in self.submission_pool:
            if rec["id"] == alpha_id:
                rec["status"] = status
                return rec
        return None

    def pool_exprs(self):
        return {r["expression"] for r in self.submission_pool}

    def _trim_pool(self):
        if len(self.submission_pool) <= self.max_pool:
            return
        self.submission_pool.sort(
            key=lambda r: (score_of(r.get("metrics")), r.get("created_at", 0)),
            reverse=True,
        )
        self.submission_pool = self.submission_pool[: self.max_pool]

    # ---- active lineages (deepening targets) ----

    def touch_lineage(self, expression, lineage, score, round_no, fields_used=None):
        lineage = list(lineage or [])
        fields_used = list(fields_used or [])
        for item in self.active_lineages:
            if item["expression"] == expression:
                item["attempts"] = item.get("attempts", 0) + 1
                item["best_score"] = max(item.get("best_score", -1), score)
                item["last_round"] = round_no
                if fields_used:
                    item["fields_used"] = fields_used
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

    def deepening_targets(self, max_deepen_per_lineage, limit=4):
        candidates = [
            item
            for item in self.active_lineages
            if item.get("attempts", 0) < max_deepen_per_lineage
        ]
        candidates.sort(key=lambda x: -x.get("best_score", -1))
        return candidates[:limit]

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

    # ---- helpers ----

    def long_term_view(self):
        """Read-only view of reliable long-term knowledge."""
        return {
            "lessons": [l for l in self.lessons if l.get("tier") == "long"],
            "avoid": list(self.avoid),
        }

    def short_term_view(self):
        return {
            "next": list(self.next),
            "active_lineages": list(self.active_lineages),
        }

    @staticmethod
    def _similar(a, b, threshold=0.8):
        ta = ExperienceMemory._tokens(a)
        tb = ExperienceMemory._tokens(b)
        if not ta or not tb:
            return False
        inter = len(set(ta) & set(tb))
        union = len(set(ta) | set(tb))
        return inter / union >= threshold

    @staticmethod
    def _tokens(text):
        return [w for w in re.split(r"[^a-z0-9]+", text.lower()) if len(w) > 2]
