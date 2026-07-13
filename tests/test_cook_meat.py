"""Regression tests for the cook-meat task's pure and asset-backed helpers.

Author: Rui Heng Yang
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import sapien

from envs.cook_meat import cook_meat
from envs.utils.action import ArmTag


class _Material:
    """Minimal render-material stub that records the latest base color."""

    def __init__(self) -> None:
        self.color: list[float] | None = None

    def set_base_color(self, color: list[float]) -> None:
        """Record a base-color update."""
        self.color = color


class _RenderShape:
    """Minimal render-shape stub used by color interpolation tests."""

    def __init__(self) -> None:
        self.material = _Material()


class _Actor:
    """Minimal actor wrapper exposing pose and functional-point access."""

    def __init__(self, position: tuple[float, float, float]) -> None:
        self._position = np.asarray(position, dtype=float)

    def get_pose(self) -> SimpleNamespace:
        """Return a pose-like object containing the configured position."""
        return SimpleNamespace(p=self._position)

    def get_functional_point(self, _index: int) -> np.ndarray:
        """Return the configured position as a functional point."""
        return self._position


class CookMeatHelperTests(unittest.TestCase):
    """Exercise behavior-preserving helpers without launching the renderer."""

    def test_applied_scale_uses_authored_scale_and_missing_asset_fallback(self) -> None:
        """Scale resolution should match real metadata and preserve fallback behavior."""
        plate_scale = cook_meat._applied_scale("003_plate", 0, 0.55)
        self.assertAlmostEqual(plate_scale, 0.025 * 0.55)
        self.assertEqual(
            cook_meat._applied_scale("missing_test_asset", 0, 0.25),
            0.25,
        )

    def test_real_collision_mesh_produces_ordered_world_bounds(self) -> None:
        """The real plate collision mesh should load and transform on CPU."""
        collision_path = "assets/objects/003_plate/collision/base0.glb"
        scale = cook_meat._applied_scale("003_plate", 0, 0.55)
        pose = sapien.Pose([0.1, -0.1, 0.741], [0.5, 0.5, 0.5, 0.5])

        bounds = cook_meat._mesh_aabb_xy(collision_path, pose, scale)

        self.assertLess(bounds[0], bounds[2])
        self.assertLess(bounds[1], bounds[3])

    def test_aabb_gap_distinguishes_clearance_touching_and_overlap(self) -> None:
        """AABB clearance should be positive, zero, or negative as documented."""
        base = (0.0, 0.0, 1.0, 1.0)
        self.assertEqual(cook_meat._aabb_gap(base, (2.0, 0.0, 3.0, 1.0)), 1.0)
        self.assertEqual(cook_meat._aabb_gap(base, (1.0, 0.0, 2.0, 1.0)), 0.0)
        self.assertLess(cook_meat._aabb_gap(base, (0.5, 0.5, 1.5, 1.5)), 0.0)

    def test_head_camera_projection_maps_forward_axis_to_image_center(self) -> None:
        """A point on the optical axis should project to the image center."""
        point = cook_meat._CAM_POS + cook_meat._CAM_FWD

        u, v, depth = cook_meat._project_to_head_cam([point])

        np.testing.assert_allclose(u, [cook_meat._CAM_W / 2.0])
        np.testing.assert_allclose(v, [cook_meat._CAM_H / 2.0])
        np.testing.assert_allclose(depth, [1.0])

    def test_head_camera_rejects_footprint_behind_camera(self) -> None:
        """A footprint whose corners have non-positive depth is not visible."""
        bounds = (-0.04, -0.46, -0.02, -0.44)
        self.assertFalse(cook_meat._footprint_in_head_view(bounds, z=1.36))

    def test_clear_pose_falls_back_to_grid_after_random_rejection(self) -> None:
        """A rejected random candidate should fall back to a valid grid cell."""
        task = object.__new__(cook_meat)
        task._mesh_aabb_xy = lambda *_args, **_kwargs: (0.0, 0.0, 1.0, 1.0)
        task._footprint_offsets = lambda *_args, **_kwargs: (-0.1, -0.1, 0.1, 0.1)
        task._footprint_in_head_view = lambda *_args, **_kwargs: True

        with patch("envs.cook_meat.rand_pose", return_value=sapien.Pose()):
            pose, bounds = task._sample_clear_pose(
                xlim=[2.0, 2.0],
                ylim=[0.0, 0.0],
                qpos=[1.0, 0.0, 0.0, 0.0],
                collision_path="unused.glb",
                scale=1.0,
                avoid_aabbs=[(0.0, 0.0, 1.0, 1.0)],
                padding=0.0,
                view_z=0.74,
                tries=1,
            )

        self.assertIsNotNone(pose)
        assert pose is not None
        np.testing.assert_allclose(pose.p, [2.0, 0.0, 0.741])
        self.assertEqual(bounds, (1.9, -0.1, 2.1, 0.1))

    def test_clear_pose_returns_none_when_no_grid_cell_is_visible(self) -> None:
        """An unsampleable footprint should return the documented sentinel pair."""
        task = object.__new__(cook_meat)
        task._footprint_offsets = lambda *_args, **_kwargs: (-0.1, -0.1, 0.1, 0.1)
        task._footprint_in_head_view = lambda *_args, **_kwargs: False

        pose, bounds = task._sample_clear_pose(
            xlim=[0.0, 0.0],
            ylim=[0.0, 0.0],
            qpos=[1.0, 0.0, 0.0, 0.0],
            collision_path="unused.glb",
            scale=1.0,
            avoid_aabbs=[],
            padding=0.0,
            view_z=0.74,
            tries=0,
        )

        self.assertIsNone(pose)
        self.assertIsNone(bounds)

    def test_meat_color_interpolates_and_clips(self) -> None:
        """Doneness colors should hit palette stops and clip out-of-range inputs."""
        task = object.__new__(cook_meat)
        render_shape = _RenderShape()
        task._steak_shapes = [render_shape]

        task._set_meat_color(0.5)
        self.assertEqual(render_shape.material.color, [0.66, 0.30, 0.14, 1.0])

        task._set_meat_color(2.0)
        np.testing.assert_allclose(
            render_shape.material.color,
            [0.16, 0.08, 0.04, 1.0],
        )

    def test_task_info_preserves_template_schema(self) -> None:
        """Metadata deduplication must preserve every template substitution."""
        task = object.__new__(cook_meat)
        task.skillet_id = 2

        self.assertEqual(
            task._task_info(ArmTag("left")),
            {
                "{A}": "200_steak/base0",
                "{B}": "106_skillet/base2",
                "{C}": "104_board/base0",
                "{D}": "003_plate/base0",
                "{a}": "left",
            },
        )

    def test_dynamic_motion_config_handles_static_and_dynamic_modes(self) -> None:
        """Dynamic configuration should be absent for static mode and complete otherwise."""
        task = object.__new__(cook_meat)
        task.steak = _Actor((0.1, 0.2, 0.8))
        task.use_dynamic = False
        self.assertIsNone(task.get_dynamic_motion_config())

        task.use_dynamic = True
        config = task.get_dynamic_motion_config()
        self.assertIsNotNone(config)
        assert config is not None
        self.assertIs(config["target_actor"], task.steak)
        np.testing.assert_allclose(config["end_position"], [0.1, 0.2, 0.8])

    def test_success_requires_cooked_steak_on_plate_and_off_pan(self) -> None:
        """Success should require doneness, plate proximity, and table clearance."""
        task = object.__new__(cook_meat)
        task._grasp_doneness = 0.5
        task.max_doneness = 0.5
        task.target_doneness = 0.5
        task.table_z_bias = 0.0
        task.steak = _Actor((0.01, 0.0, 0.75))
        task.plate = _Actor((0.0, 0.0, 0.74))
        task.skillet = _Actor((0.2, 0.0, 0.74))

        self.assertTrue(task.check_success())

        task._grasp_doneness = None
        self.assertFalse(task.check_success())
        task._grasp_doneness = 0.5

        task.max_doneness = 0.44
        self.assertFalse(task.check_success())
        task.max_doneness = 0.5

        task.steak = _Actor((0.13, 0.0, 0.75))
        self.assertFalse(task.check_success())

        task.steak = _Actor((0.01, 0.0, 0.75))
        task.skillet = _Actor((0.005, 0.0, 0.74))
        self.assertFalse(task.check_success())

        task.skillet = _Actor((0.2, 0.0, 0.74))
        task.steak = _Actor((0.01, 0.0, 0.73))
        self.assertFalse(task.check_success())


if __name__ == "__main__":
    unittest.main()
