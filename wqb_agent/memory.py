import json
import os
import re
import time
import uuid


class ExperienceMemory:
    def __init__(self, state_dir=".wqb_state", max_lessons=20, max_avoid=30, max_next=15):
        self.state_dir = state_dir
        self.max_lessons = max_lessons
        self.max_avoid = max_avoid
        self.max_next = max_next
        self.current_best = None
        self.lessons = []
        self.avoid = []
        self.next = []
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
        self.lessons = data.get("lessons", [])
        self.avoid = data.get("avoid", [])
        self.next = data.get("next", [])
        self.updated_round = data.get("updated_round", 0)
        return self

    def save(self):
        data = {
            "current_best": self.current_best,
            "lessons": self.lessons,
            "avoid": self.avoid,
            "next": self.next,
            "updated_round": self.updated_round,
        }
        path = self.memory_path()
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def add_lesson(self, claim, source_round, evidence, confidence=0.5):
        for lesson in self.lessons:
            if self._similar(lesson["claim"], claim, threshold=0.8):
                lesson["source_round"] = source_round
                lesson["evidence"] = lesson.get("evidence", 0) + evidence
                lesson["confidence"] = min(1.0, lesson["confidence"] + 0.15)
                lesson["updated"] = time.time()
                return lesson
        entry = {
            "id": uuid.uuid4().hex[:8],
            "claim": claim,
            "source_round": source_round,
            "evidence": evidence,
            "confidence": min(confidence, 1.0),
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

    def set_current_best(self, experiment):
        self.current_best = experiment.to_dict()

    def is_avoided(self, direction):
        return any(item["direction"] == direction for item in self.avoid)

    def top_next(self, n=5):
        ordered = sorted(self.next, key=lambda x: -x.get("priority", 0))
        return ordered[:n]

    def compress(self):
        merged = []
        for lesson in sorted(self.lessons, key=lambda x: -x.get("evidence", 0)):
            if not any(self._similar(lesson["claim"], m["claim"], threshold=0.75) for m in merged):
                merged.append(lesson)
        self.lessons = merged[: self.max_lessons]
        self.avoid = sorted(
            self.avoid, key=lambda x: -x.get("updated", 0)
        )[: self.max_avoid]
        self.next = sorted(
            self.next, key=lambda x: -x.get("priority", 0)
        )[: self.max_next]

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
