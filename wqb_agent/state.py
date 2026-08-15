import time
import uuid


class Experiment:
    def __init__(self, round_no, hypothesis_id, expression, settings, fields_used):
        self.id = uuid.uuid4().hex[:12]
        self.round = round_no
        self.hypothesis_id = hypothesis_id
        self.expression = expression
        self.settings = dict(settings)
        self.fields_used = list(fields_used)
        self.status = "PENDING"
        self.metrics = None
        self.error = None
        self.alpha_id = None
        self.mutation = None
        self.rationale = None
        self.created_at = time.time()

    def to_dict(self):
        return {
            "id": self.id,
            "round": self.round,
            "hypothesis_id": self.hypothesis_id,
            "expression": self.expression,
            "settings": self.settings,
            "fields_used": self.fields_used,
            "status": self.status,
            "metrics": self.metrics,
            "error": self.error,
            "alpha_id": self.alpha_id,
            "mutation": self.mutation,
            "rationale": self.rationale,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        exp = cls(
            data["round"],
            data["hypothesis_id"],
            data["expression"],
            data["settings"],
            data["fields_used"],
        )
        exp.id = data["id"]
        exp.status = data["status"]
        exp.metrics = data.get("metrics")
        exp.error = data.get("error")
        exp.alpha_id = data.get("alpha_id")
        exp.mutation = data.get("mutation")
        exp.rationale = data.get("rationale")
        exp.created_at = data.get("created_at", 0)
        return exp


class Trajectory:
    def __init__(self, max_len=100):
        self.experiments = []
        self.max_len = max_len

    def add(self, experiment):
        self.experiments.append(experiment)
        if len(self.experiments) > self.max_len:
            self.experiments = self.experiments[-self.max_len:]

    def recent(self, n=20):
        return self.experiments[-n:]

    def completed(self):
        return [e for e in self.experiments if e.metrics is not None]

    def to_dict(self):
        return {"experiments": [e.to_dict() for e in self.experiments]}

    @classmethod
    def from_dict(cls, data, max_len=100):
        traj = cls(max_len=max_len)
        traj.experiments = [Experiment.from_dict(e) for e in data.get("experiments", [])]
        return traj


class ResearchState:
    def __init__(self, round_no=0, hypothesis=None, dataset=None, fields_used=None):
        self.round_no = round_no
        self.hypothesis = hypothesis
        self.dataset = dataset
        self.fields_used = fields_used or []
        self.started_at = time.time()

    def to_dict(self):
        return {
            "round_no": self.round_no,
            "hypothesis": self.hypothesis,
            "dataset": self.dataset,
            "fields_used": self.fields_used,
            "started_at": self.started_at,
        }
