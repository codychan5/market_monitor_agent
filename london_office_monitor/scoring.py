"""The opportunity score, computed in code from visible parts so the analyst can see why a card ranks where it does.
The model never invents a score."""

# Points by signal type: distress first, because forced sellers are where off-market deals come from.
SIGNAL_WEIGHTS = {"distress": 25, "demand": 20, "supply": 20, "pricing": 20, "policy": 15, "other": 5}
# Similarity to the mandate or a portfolio asset (0 to 1) is worth up to this many points.
MANDATE_WEIGHT = 40
CONFIDENCE_POINTS = {"high": 15, "medium": 8, "low": 0}
# (maximum age in days, points): the first band that covers the newest evidence wins; older evidence scores 0.
RECENCY_POINTS = ((7, 20), (14, 12), (30, 6))
# Each part's maximum, in the order the card shows them; the analyst sees every part out of its maximum.
COMPONENT_MAXIMUMS = {"mandate": MANDATE_WEIGHT, "signal": max(SIGNAL_WEIGHTS.values()),
                      "recency": max(points for _, points in RECENCY_POINTS),
                      "confidence": max(CONFIDENCE_POINTS.values())}
MAX_SCORE = 100
if sum(COMPONENT_MAXIMUMS.values()) != MAX_SCORE:
    raise RuntimeError(f"score parts add up to {sum(COMPONENT_MAXIMUMS.values())}, not {MAX_SCORE}")


def score_card(signal_type: str, mandate_similarity: float, newest_evidence_age_days: int,
               confidence: str) -> tuple[int, dict[str, int]]:
    """The total score (0 to 100) and its four components."""
    components = {
        "mandate": round(min(max(mandate_similarity, 0.0), 1.0) * MANDATE_WEIGHT),
        "signal": SIGNAL_WEIGHTS[str(signal_type)],
        "recency": next((points for days, points in RECENCY_POINTS if newest_evidence_age_days <= days), 0),
        "confidence": CONFIDENCE_POINTS[confidence],
    }
    return sum(components.values()), components
