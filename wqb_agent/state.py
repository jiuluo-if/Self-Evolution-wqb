import time
import uuid


def score_of(metrics):
    """Unified quality score: fitness when present, otherwise sharpe."""
    if not metrics:
        return -1.0
    fitness = metrics.get("fitness")
    sharpe = metrics.get("sharpe")
    if fitness is not None and fitness != 0:
        return fitness
    return sharpe if sharpe is not None else -1.0


class Experiment:
    def __init__(
        self,
        round_no,
        hypothesis_id,
        expression,
        settings,
        fields_used,
        datasets=None,
        lineage=None,
    ):
        self.id = uuid.uuid4().hex[:12]
        self.round = round_no
        self.hypothesis_id = hypothesis_id
        self.expression = expression
        self.settings = dict(settings)
        self.fields_used = list(fields_used)
        self.datasets = list(datasets or [])
        self.lineage = list(lineage or [])
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
            "datasets": self.datasets,
            "lineage": self.lineage,
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
            datasets=data.get("datasets", []),
            lineage=data.get("lineage", []),
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


class AlphaRecord:
    """An alpha that earned a slot in the submission candidate pool."""

    STATUS_CANDIDATE = "CANDIDATE"
    STATUS_SUSPICIOUS = "SUSPICIOUS_HIGH_SIGNAL"
    STATUS_VALIDATED = "VALIDATED_HIGH_SIGNAL"
    STATUS_REJECTED = "REJECTED"

    def __init__(
        self,
        expression,
        metrics=None,
        fields_used=None,
        datasets=None,
        hypothesis_id=None,
        lineage=None,
        status="CANDIDATE",
        round_no=0,
        mutation=None,
    ):
        self.id = uuid.uuid4().hex[:10]
        self.expression = expression
        self.metrics = metrics or {}
        self.fields_used = list(fields_used or [])
        self.datasets = list(datasets or [])
        self.hypothesis_id = hypothesis_id
        self.lineage = list(lineage or [])
        self.status = status
        self.round_no = round_no
        self.mutation = mutation
        self.created_at = time.time()

    @property
    def score(self):
        return score_of(self.metrics)

    def to_dict(self):
        return {
            "id": self.id,
            "expression": self.expression,
            "metrics": self.metrics,
            "fields_used": self.fields_used,
            "datasets": self.datasets,
            "hypothesis_id": self.hypothesis_id,
            "lineage": self.lineage,
            "status": self.status,
            "round_no": self.round_no,
            "mutation": self.mutation,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data):
        rec = cls(
            expression=data["expression"],
            metrics=data.get("metrics") or {},
            fields_used=data.get("fields_used", []),
            datasets=data.get("datasets", []),
            hypothesis_id=data.get("hypothesis_id"),
            lineage=data.get("lineage", []),
            status=data.get("status", cls.STATUS_CANDIDATE),
            round_no=data.get("round_no", 0),
            mutation=data.get("mutation"),
        )
        rec.id = data["id"]
        rec.created_at = data.get("created_at", 0)
        return rec


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
        traj.experiments = [
            Experiment.from_dict(e) for e in data.get("experiments", [])
        ]
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
