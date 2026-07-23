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

    def test_compose_yaw_keeps_board_flat_on_table(self) -> None:
        """World-Z yaw must not tip the board's local up off world +Z."""
        base = cook_meat.BOARD_BASE_QPOS
        # Board base is 90° about +X → local +Y ≈ world +Z (flat on table).
        for yaw in (0.0, 0.4, -0.7, 1.2):
            q = cook_meat._compose_yaw_qpos(base, yaw)
            R = sapien.Pose([0, 0, 0], q).to_transformation_matrix()[:3, :3]
            local_y = R @ np.array([0.0, 1.0, 0.0])
            # Flat: board's local +Y should stay near world +Z.
            self.assertGreater(float(local_y[2]), 0.85)

    def test_compose_yaw_qpos_is_unit_quaternion(self) -> None:
        """Z-yaw composition should keep a unit wxyz quaternion."""
        q = cook_meat._compose_yaw_qpos([1.0, 0.0, 0.0, 0.0], np.pi / 4)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)

    def test_mirrored_prop_pose_flips_x_with_side(self) -> None:
        """Left station placement should mirror the right-side nominal X."""
        task = object.__new__(cook_meat)
        task._aabb_clear = lambda *a, **k: True
        task._mesh_aabb_xy = lambda *_a, **_k: (-0.05, -0.05, 0.05, 0.05)
        with patch("envs.cook_meat.np.random.uniform", side_effect=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]):
            right, _, _ = cook_meat._sample_mirrored_prop_pose(
                task,
                side=1.0,
                nom_x=0.10,
                nom_y=-0.10,
                base_qpos=[1, 0, 0, 0],
                collision_path="unused",
                scale=1.0,
                avoid_aabbs=[],
                padding=0.0,
                view_z=0.74,
                jitter_xy=0.0,
                yaw_lim=0.0,
                tries=1,
            )
            left, _, _ = cook_meat._sample_mirrored_prop_pose(
                task,
                side=-1.0,
                nom_x=0.10,
                nom_y=-0.10,
                base_qpos=[1, 0, 0, 0],
                collision_path="unused",
                scale=1.0,
                avoid_aabbs=[],
                padding=0.0,
                view_z=0.74,
                jitter_xy=0.0,
                yaw_lim=0.0,
                tries=1,
            )
        self.assertIsNotNone(right)
        self.assertIsNotNone(left)
        assert right is not None and left is not None
        self.assertAlmostEqual(float(right.p[0]), 0.10, places=5)
        self.assertAlmostEqual(float(left.p[0]), -0.10, places=5)
        self.assertAlmostEqual(float(right.p[1]), float(left.p[1]), places=5)

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
        task.cook_button_enabled = False
        task.dual_setup_enabled = False
        task.stations = None

        self.assertEqual(
            task._task_info(ArmTag("left")),
            {
                "{A}": "200_steak/base0",
                "{B}": "106_skillet/base2",
                "{C}": "104_board/base0",
                "{a}": "left",
                "{o}": "default",
            },
        )

        task.cook_button_enabled = True
        info = task._task_info(ArmTag("right"))
        self.assertEqual(info["{o}"], "option 1")
        self.assertEqual(info["{E}"], "cook_key")
        self.assertEqual(info["{a}"], "right")

        task.dual_setup_enabled = True
        info = task._task_info()
        self.assertEqual(info["{o}"], "option 1, option 2")
        self.assertEqual(info["{a}"], "both arms")
        self.assertEqual(info["{n}"], "2")
        self.assertEqual(info["{E}"], "cook_key")  # Opt 1+2: keys on both stations
        self.assertTrue(task.use_cook_button)

    def test_opt1_plus_2_keeps_cook_button(self) -> None:
        """Opt 1+2 keeps cook keys; Opt 2 alone does not."""
        task = object.__new__(cook_meat)
        task.cook_button_enabled = True
        task.dual_setup_enabled = True
        self.assertTrue(task.use_cook_button)
        task.cook_button_enabled = False
        self.assertFalse(task.use_cook_button)
        task.dual_setup_enabled = False
        task.cook_button_enabled = True
        self.assertTrue(task.use_cook_button)

    def test_legacy_option_forces_cook_button(self) -> None:
        """CLI --option 1 must force-enable cook_button_enabled over yaml false."""
        task = object.__new__(cook_meat)
        task._cook_cfg = {"option": 1, "cook_button_enabled": False}
        task._apply_legacy_option()
        self.assertTrue(task._cook_cfg["cook_button_enabled"])

    def test_legacy_option_forces_dual_setup(self) -> None:
        """CLI --option 2 must force-enable dual_setup_enabled over yaml false."""
        task = object.__new__(cook_meat)
        task._cook_cfg = {"option": 2, "dual_setup_enabled": False}
        task._apply_legacy_option()
        self.assertTrue(task._cook_cfg["dual_setup_enabled"])

    def test_button_mode_cooks_only_while_pressed_on_pan(self) -> None:
        """Opt 1 advances doneness only when the key is pressed and steak is on the pan."""
        task = object.__new__(cook_meat)
        task.cook_button_enabled = True
        task.cook_steps = 100
        task.doneness = 0.0
        task.max_doneness = 0.0
        task._grasp_doneness = None
        st = {
            "doneness": 0.0,
            "max_doneness": 0.0,
            "grasp_doneness": None,
            "cooking_active": False,
            "steak_shapes": [],
        }
        task.stations = [st]

        with patch.object(cook_meat, "_button_is_pressed_station", return_value=True), patch.object(
            cook_meat, "_steak_on_pan_station", return_value=True
        ), patch("envs.cook_meat.Base_Task._update_kinematic_tasks", return_value=None):
            cook_meat._update_kinematic_tasks(task)
        self.assertAlmostEqual(st["doneness"], 0.01, places=5)
        self.assertAlmostEqual(task.doneness, 0.01, places=5)

        st["doneness"] = 0.0
        task.doneness = 0.0
        with patch.object(cook_meat, "_button_is_pressed_station", return_value=False), patch.object(
            cook_meat, "_steak_on_pan_station", return_value=True
        ), patch("envs.cook_meat.Base_Task._update_kinematic_tasks", return_value=None):
            cook_meat._update_kinematic_tasks(task)
        self.assertEqual(st["doneness"], 0.0)

        # After grasp_doneness is committed, further presses must not overcook.
        st["grasp_doneness"] = 0.5
        st["doneness"] = 0.5
        task.doneness = 0.5
        with patch.object(cook_meat, "_button_is_pressed_station", return_value=True), patch.object(
            cook_meat, "_steak_on_pan_station", return_value=True
        ), patch("envs.cook_meat.Base_Task._update_kinematic_tasks", return_value=None):
            cook_meat._update_kinematic_tasks(task)
        self.assertEqual(st["doneness"], 0.5)

    def test_grasp_latch_is_per_station(self) -> None:
        """Grasping one dual-station steak must not freeze the other station's cook."""
        task = object.__new__(cook_meat)
        task.cook_button_enabled = False
        task.cook_steps = 100
        task.target_doneness = 0.5
        task.doneness = 0.0
        task.max_doneness = 0.0
        task._grasp_doneness = None
        left = {
            "doneness": 0.4,
            "max_doneness": 0.4,
            "grasp_doneness": None,
            "cooking_active": True,
            "awaiting_return_grasp": True,
            "cook_phase_done": False,
            "steak_shapes": [],
            "steak_name": "200_steak_left",
        }
        right = {
            "doneness": 0.4,
            "max_doneness": 0.4,
            "grasp_doneness": None,
            "cooking_active": True,
            "awaiting_return_grasp": False,
            "cook_phase_done": False,
            "steak_shapes": [],
            "steak_name": "200_steak_right",
        }
        task.stations = [left, right]

        def _held(self, station):
            return station["steak_name"] == "200_steak_left"

        with patch.object(cook_meat, "_steak_held", _held), patch.object(
            cook_meat, "_steak_on_pan_station", return_value=True
        ), patch("envs.cook_meat.Base_Task._update_kinematic_tasks", return_value=None):
            cook_meat._update_kinematic_tasks(task)
        self.assertEqual(left["grasp_doneness"], 0.4)
        self.assertFalse(left["cooking_active"])
        self.assertIsNone(right["grasp_doneness"])
        self.assertTrue(right["cooking_active"])
        self.assertAlmostEqual(right["doneness"], 0.41, places=5)

    def test_cook_phase_done_freezes_button_cooking(self) -> None:
        """After cook wait, lingering key contact must not overcook."""
        task = object.__new__(cook_meat)
        task.cook_button_enabled = True
        task.cook_steps = 100
        task.doneness = 0.5
        task.max_doneness = 0.5
        task._grasp_doneness = None
        st = {
            "doneness": 0.5,
            "max_doneness": 0.5,
            "grasp_doneness": None,
            "cooking_active": False,
            "awaiting_return_grasp": True,
            "cook_phase_done": True,
            "steak_shapes": [],
        }
        task.stations = [st]
        with patch.object(cook_meat, "_button_is_pressed_station", return_value=True), patch.object(
            cook_meat, "_steak_on_pan_station", return_value=True
        ), patch.object(cook_meat, "_steak_held", return_value=False), patch(
            "envs.cook_meat.Base_Task._update_kinematic_tasks", return_value=None
        ):
            cook_meat._update_kinematic_tasks(task)
        self.assertEqual(st["doneness"], 0.5)

    def test_get_dynamic_motion_config_respects_flags(self) -> None:
        """Dynamic configuration should be absent for static mode and complete otherwise."""
        task = object.__new__(cook_meat)
        task.steak = _Actor((0.1, 0.2, 0.8))
        task.use_dynamic = False
        task.dual_setup_enabled = False
        self.assertIsNone(task.get_dynamic_motion_config())

        task.use_dynamic = True
        config = task.get_dynamic_motion_config()
        self.assertIsNotNone(config)
        assert config is not None
        self.assertIs(config["target_actor"], task.steak)
        np.testing.assert_allclose(config["end_position"], [0.1, 0.2, 0.8])

        task.dual_setup_enabled = True
        self.assertIsNone(task.get_dynamic_motion_config())

    def test_success_requires_cooked_steak_on_board_and_off_pan(self) -> None:
        """Success should require in-band doneness, board proximity, and off-pan."""
        task = object.__new__(cook_meat)
        task.stations = None
        task._grasp_doneness = 0.5
        task.max_doneness = 0.5
        task.target_doneness = 0.5
        task.cook_doneness_tol = 0.08
        task.table_z_bias = 0.0
        task.steak = _Actor((0.01, 0.0, 0.75))
        task.board = _Actor((0.0, 0.0, 0.74))
        task.skillet = _Actor((0.2, 0.0, 0.74))

        self.assertTrue(task.check_success())

        task._grasp_doneness = None
        self.assertFalse(task.check_success())
        task._grasp_doneness = 0.5

        # Under-cooked at grasp time.
        task._grasp_doneness = 0.35
        self.assertFalse(task.check_success())
        # Over-cooked at grasp time.
        task._grasp_doneness = 0.70
        self.assertFalse(task.check_success())
        task._grasp_doneness = 0.5

        task.steak = _Actor((0.13, 0.0, 0.75))
        self.assertFalse(task.check_success())

        task.steak = _Actor((0.01, 0.0, 0.75))
        task.skillet = _Actor((0.005, 0.0, 0.74))
        self.assertFalse(task.check_success())

        task.skillet = _Actor((0.2, 0.0, 0.74))
        task.steak = _Actor((0.01, 0.0, 0.73))
        self.assertFalse(task.check_success())

    def test_dual_success_requires_both_stations(self) -> None:
        """Opt 2 success requires every station's steak cooked and back on its board."""
        task = object.__new__(cook_meat)
        task.dual_setup_enabled = True
        task.target_doneness = 0.5
        task.cook_doneness_tol = 0.08
        task.table_z_bias = 0.0
        good = {
            "grasp_doneness": 0.5,
            "max_doneness": 0.5,
            "board_xy": (0.0, 0.0),
            "board_top": 0.75,
            "steak": _Actor((0.01, 0.0, 0.76)),
            "board": _Actor((0.0, 0.0, 0.74)),
            "skillet": _Actor((0.2, 0.0, 0.74)),
        }
        bad = dict(good)
        bad["grasp_doneness"] = None
        task.stations = [good, bad]
        self.assertFalse(task.check_success())
        task.stations = [good, dict(good)]
        self.assertTrue(task.check_success())
        # Overcooked grasp on one station fails the episode.
        over = dict(good)
        over["grasp_doneness"] = 0.75
        task.stations = [good, over]
        self.assertFalse(task.check_success())
        # Undercooked sibling also fails — both steaks must be cooked properly.
        under = dict(good)
        under["grasp_doneness"] = 0.30
        task.stations = [good, under]
        self.assertFalse(task.check_success())
        # Dual flagged but only one station present is not success.
        task.stations = [good]
        self.assertFalse(task.check_success())

    def test_sample_cook_steps_default_is_nominal_pm_jitter(self) -> None:
        """Default cook speed samples around nominal cook_steps ± 20%."""
        rng = np.random.RandomState(0)
        with patch("envs.cook_meat.np.random.uniform", side_effect=lambda lo, hi: rng.uniform(lo, hi)):
            samples = [cook_meat._sample_cook_steps({"cook_steps": 5000}) for _ in range(40)]
        self.assertTrue(all(4000 <= s <= 6000 for s in samples))
        self.assertGreater(max(samples) - min(samples), 0)

        fixed = cook_meat._sample_cook_steps({"cook_steps": 1000, "cook_speed_jitter": 0.0})
        self.assertEqual(fixed, 1000)

        legacy = cook_meat._sample_cook_steps({"cook_steps_min": 2000, "cook_steps_max": 2000})
        self.assertEqual(legacy, 2000)


if __name__ == "__main__":
    unittest.main()
