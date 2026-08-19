"""Helpers for household / task partial scores in [0, 1].

Descending band specs use the authoring convention ``[hi, lo)`` with ``hi > lo``
(larger error written first). That maps to ``lo < x <= hi``.
Example: ``[10, 7.5)`` → ``7.5 < x <= 10``.

Ascending ``[lo, hi)`` / ``(lo, hi]`` helpers use standard half-open intervals.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

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
