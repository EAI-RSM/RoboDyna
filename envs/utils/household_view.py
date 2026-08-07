"""Shared head-camera framing and episode cutoff for household tasks / demos."""
from __future__ import annotations

import numpy as np
import sapien


class EpisodeTimeLimit(Exception):
    """Raised when a household episode hits the step cutoff while saving."""


# Fixed elevated head view (same for every household task).
HOUSEHOLD_HEAD_POS = np.array([0.0, -0.50, 2.0], dtype=np.float64)
HOUSEHOLD_HEAD_FORWARD = np.array([0.0, 0.45, -1.0], dtype=np.float64)
HOUSEHOLD_HEAD_LEFT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

HOUSEHOLD_TASKS = frozenset(
    {
        "boil_milk",
        "catch_cup",
        "clean_table",
        "cook_food",
        "cook_food_timer",
        "fill_coffee_jar",
        "make_soup",
        "make_soup_test",
        "measure_ingredient",
        "catch_mouse_object_drop",
        "pour_beer",
        "serve_dinner",
        "stop_ball",
        "trap_bug",
    }
)

# Shared episode cutoff for household + base tasks (sim / control steps, not wall-clock).
EPISODE_MAX_STEPS = 15000
# Back-compat alias used by record_demo / older callers.
HOUSEHOLD_MAX_STEPS = EPISODE_MAX_STEPS
# Kept for callers that still pass --max-seconds; unused as the household default.
HOUSEHOLD_MAX_SECONDS = 60.0


def configure_household_head_camera(task) -> None:
    """Force ``head_camera`` to the shared household pose."""
    cams = getattr(task, "cameras", None)
    if cams is None:
        return
    names = list(getattr(cams, "static_camera_name", []) or [])
    clist = list(getattr(cams, "static_camera_list", []) or [])
    if "head_camera" not in names:
        return
    camera = clist[names.index("head_camera")]
    forward = HOUSEHOLD_HEAD_FORWARD / np.linalg.norm(HOUSEHOLD_HEAD_FORWARD)
    left = HOUSEHOLD_HEAD_LEFT / np.linalg.norm(HOUSEHOLD_HEAD_LEFT)
    up = np.cross(forward, left)
    m = np.eye(4)
    m[:3, :3] = np.stack([forward, left, up], axis=1)
    m[:3, 3] = HOUSEHOLD_HEAD_POS
    camera.entity.set_pose(sapien.Pose(m))
