"""Hypothesis Research Contract: what a falsifiable research proposition must
declare before any candidate is built from it.

A hypothesis stays a small dict (backward-compatible with the legacy
id / statement / tags / direction / datasets shape) but must additionally
commit to the economic content the system will test:

  economic_intuition   - why this effect should exist in the market
  expected_mechanism   - the specific change the alpha tries to capture
  field_semantics      - WHAT economic meaning a field must have; NEVER a real
                         field id (the real field comes from Field Discovery)
  expected_direction   - sign of the field -> forward-return mapping
  expected_horizon     - a research prior in days, not a window to search
  failure_condition    - pre-registered results that would lower belief

The contract exists so the system knows, before Simulation, why it believes
the effect, which field meaning it needs, which direction it expects, over
which horizon, and what would refute it. These are never answers invented
after seeing a Sharpe.
"""

REQUIRED_CONTRACT_FIELDS = (
    "economic_intuition",
    "expected_mechanism",
    "field_semantics",
    "expected_direction",
    "expected_horizon_days",
    "failure_condition",
)

# Canonical horizon prior in days. Mirrors WINDOW_STEPS in candidate.py.
HORIZON_DAYS = (5, 10, 20, 60)

# A hypothesis's expected horizon may only steer the initial operator/window
# and allow the single neighboring steps listed here. It never licenses an
# arbitrary window grid (e.g. range(2, 121)).
HORIZON_NEIGHBORS = {
    5: (5, 10),
    10: (5, 10, 20),
    20: (10, 20, 60),
    60: (20, 60),
}

# Reference mechanism vocabulary. Used for documentation and candidate
# traceability; validation only requires a non-empty string so future
# mechanisms are not blocked by an enumeration.
KNOWN_MECHANISMS = (
    "temporary price pressure",
    "analyst information revision",
    "sentiment continuation",
    "risk-premium compensation",
    "earnings information diffusion",
)


class ContractViolation(Exception):
    """Raised when a hypothesis does not satisfy its Research Contract."""


def _is_blank(value):
    return value in (None, "", [], {})


def validate_contract(hypothesis, strict=True):
    """Return the list of contract problems (empty = valid).

    With strict=True the first problem raises ContractViolation instead.
    """
    hypothesis = hypothesis or {}
    problems = []
    for field in REQUIRED_CONTRACT_FIELDS:
        if _is_blank(hypothesis.get(field)):
            problems.append(f"missing contract field: {field}")

    mechanism = hypothesis.get("expected_mechanism")
    if isinstance(mechanism, str) and not mechanism.strip():
        problems.append("expected_mechanism must be a non-empty string")

    semantics = hypothesis.get("field_semantics") or {}
    primary = semantics.get("primary") or {}
    if not primary.get("concept") or not primary.get("description"):
        problems.append(
            "field_semantics.primary must declare concept and description"
        )

    direction = hypothesis.get("expected_direction") or {}
    sign = direction.get("sign")
    if sign not in ("positive", "negative"):
        problems.append("expected_direction.sign must be 'positive' or 'negative'")

    horizon = hypothesis.get("expected_horizon_days")
    if horizon not in HORIZON_DAYS:
        problems.append(f"expected_horizon_days must be one of {HORIZON_DAYS}")

    failure_condition = hypothesis.get("failure_condition")
    if isinstance(failure_condition, str) and not failure_condition.strip():
        problems.append("failure_condition must not be empty")

    if strict and problems:
        raise ContractViolation("; ".join(problems))
    return problems


def has_field_semantics(hypothesis):
    """True when the hypothesis declares the field meaning Discovery needs."""
    semantics = (hypothesis or {}).get("field_semantics") or {}
    return bool((semantics.get("primary") or {}).get("concept"))
