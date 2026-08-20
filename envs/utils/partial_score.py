"""Helpers for household / task partial scores in [0, 1].

Descending band specs use the authoring convention ``[hi, lo)`` with ``hi > lo``
(larger error written first). That maps to ``lo < x <= hi``.
Example: ``[10, 7.5)`` → ``7.5 < x <= 10``.

Ascending ``[lo, hi)`` / ``(lo, hi]`` helpers use standard half-open intervals.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence, Tuple

Band = Tuple[float, float, float]  # (hi, lo, score)


def score_descending_bands(x: float, bands: Sequence[Band], *, default: float = 0.0) -> float:
    """Map ``x`` onto the first matching ``[hi, lo)`` band (``lo < x <= hi``)."""
    v = float(x)
    for hi, lo, score in bands:
        if float(lo) < v <= float(hi):
            return float(score)
    return float(default)


def score_half_open_intervals(
    x: float,
    intervals: Iterable[Tuple[float, float, float]],
    *,
    default: float = 0.0,
) -> float:
    """Map ``x`` onto ``[lo, hi)`` intervals given as ``(lo, hi, score)``."""
    v = float(x)
    for lo, hi, score in intervals:
        if float(lo) <= v < float(hi):
            return float(score)
    return float(default)


def score_open_closed_intervals(
    x: float,
    intervals: Iterable[Tuple[float, float, float]],
    *,
    default: float = 0.0,
) -> float:
    """Map ``x`` onto ``(lo, hi]`` intervals given as ``(lo, hi, score)``."""
    v = float(x)
    for lo, hi, score in intervals:
        if float(lo) < v <= float(hi):
            return float(score)
    return float(default)


def completed_action_score(
    owner: Any,
    base_score: float,
    *,
    completed_actions: Mapping[str, Any],
    penalties: Mapping[str, int | bool] | None = None,
) -> float:
    """Return an auditable partial score based on completed actions.

    New Base / Conceptual task scores deliberately use discrete, observable
    milestones rather than distance-to-goal estimates.  Every recorded error
    applies the same half-credit multiplier, once per event.  The explanation
    is saved on the environment so interactive results can show how the score
    was obtained.
    """
    base = min(1.0, max(0.0, float(base_score)))
    multiplier = 1.0
    penalty_rows: list[dict[str, Any]] = []
    for name, raw_count in (penalties or {}).items():
        if isinstance(raw_count, bool):
            count = int(raw_count)
        else:
            try:
                count = max(0, int(raw_count))
            except (TypeError, ValueError):
                count = 0
        if not count:
            continue
        factor = 0.5 ** count
        multiplier *= factor
        penalty_rows.append({
            "event": str(name),
            "count": count,
            "factor": factor,
        })

    final = min(1.0, max(0.0, base * multiplier))
    owner._partial_score_detail = {
        "strategy": "completed_actions",
        "base_progress": base,
        "completed_actions": dict(completed_actions),
        "penalties": penalty_rows,
        "final_score": final,
    }
    return final
