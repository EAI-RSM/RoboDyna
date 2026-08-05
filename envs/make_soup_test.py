"""make_soup_test: alias of make_soup on the flush countertop gas cooktop.

The shared KitchenS loader now uses ``268_countertop_gas_stove`` with a
top-facing knob for all cooking tasks; this class exists so existing
``make_soup_test`` configs / demos keep working.
"""
from __future__ import annotations

from typing import Any

from .make_soup import make_soup


class make_soup_test(make_soup):
    """Identical to ``make_soup`` (countertop gas stove is now the KitchenS default)."""

    def setup_demo(self, **kwags: Any) -> None:
        ta = dict(kwags.get("task_args") or {})
        cfg = dict(ta.get("make_soup_test") or ta.get("make_soup") or {})
        ta["make_soup"] = cfg
        kwags["task_args"] = ta
        super().setup_demo(**kwags)

    def play_once(self) -> dict[str, Any]:
        info = super().play_once()
        if isinstance(self.info.get("info"), dict):
            self.info["info"]["{C}"] = "countertop_gas_stove"
        return info
