"""Canonical research-belief identity.

A belief must stand for ONE falsifiable research proposition ("under
hypothesis H, fields X, used in direction D, are expected to predict forward
returns"), never for "these fields were used".

Field-level identity alone is too coarse: reversal and momentum both operate
on `returns` and encode opposite economic claims, so they must not share one
belief. A full expression is too fine: `rank(ts_mean(x, 20))` and
`rank(ts_mean(x, 21))` are the same proposition and must stay aggregated. The
identity sits in between:

    hypothesis_id + normalized fields + direction

All producers of belief evidence (Reflection, Validation, Memory callers)
derive the key through this single helper so the original suspicious
experiment and its later robustness validation can never drift into two
different beliefs.

The mechanism / operator family and expected horizon are intentionally kept
out of the identity: the seed hypothesis schema does not yet carry an
explicit horizon, and family would be derivable on the reflection side but
not always on the validation side, creating a drift risk. They remain
available for the claim description and future extension.
"""


def normalize_fields(fields):
    """Stable canonical field set: deduplicated and sorted."""
    return sorted({f for f in (fields or []) if f})


def _direction(hypothesis=None, direction=None):
    if direction is not None:
        return direction
    if hypothesis:
        return hypothesis.get("direction")
    return None


def belief_identity(hypothesis_id, fields, hypothesis=None, direction=None):
    """The unique key of a research belief.

    Two experiments share a belief if and only if they test the same
    hypothesis on the same fields in the same direction. `hypothesis` is a
    hypothesis dict carrying an optional `direction`; an explicit `direction`
    wins over the dict value.
    """
    direction = _direction(hypothesis, direction)
    norm = normalize_fields(fields)
    label = ",".join(norm) if norm else "unknown"
    return (
        f"hyp:{hypothesis_id or 'unknown'}|"
        f"fields:{label}|"
        f"dir:{direction or '?'}"
    )


def belief_claim(hypothesis_id, fields, hypothesis=None, direction=None,
                 horizon=None):
    """A human-readable statement of the belief the key represents."""
    direction = _direction(hypothesis, direction)
    if horizon is None and hypothesis:
        horizon = hypothesis.get("horizon")
    norm = normalize_fields(fields)
    label = ",".join(norm) if norm else "?"
    dir_desc = direction if direction else "the hypothesis"
    parts = [
        f"Under hypothesis {hypothesis_id or 'unknown'}",
        f"fields [{label}]",
        f"used in direction {dir_desc}",
        "are expected to predict forward returns",
    ]
    if horizon:
        parts.append(f"over horizon {horizon}")
    return " ".join(parts)
