"""Focused regressions for cook_food knob and heating state."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from envs._kitchens_base_task import KitchenS_base_task
from envs.cook_food import cook_food


class CookFoodKnobTests(unittest.TestCase):
    def test_grasped_wrist_turn_drives_semantic_knob_angle(self) -> None:
        task = object.__new__(KitchenS_base_task)
        task._knob_turn_grasp_valid = True
        task._knob_turn_start_angle = 0.0
        task._knob_turn_start_ee_twist = 0.0
        task._ee_knob_twist = lambda: 0.75

        self.assertAlmostEqual(task._coupled_knob_angle(0.0), -0.75)

    def test_live_turn_commits_joint_and_stove_together(self) -> None:
        task = object.__new__(KitchenS_base_task)
        task._knob_grasp_active = True
        task._knob_turn_grasp_valid = True
        task._knob_turn_start_angle = 0.0
        task._knob_turn_start_ee_twist = 0.0
        task._ee_knob_twist = lambda: 0.5
        task._get_knob_joint_angle = lambda: 0.0
        task.knob_angle = 0.0
        joint_angles = []
        stove_angles = []
        task._set_knob_articulation_qpos = joint_angles.append
        task._commit_stove_from_knob_angle = stove_angles.append

        task._update_knob_from_physics()

        self.assertEqual(joint_angles, [-0.5])
        self.assertEqual(stove_angles, [-0.5])
        self.assertAlmostEqual(task.knob_angle, -0.5)

    def test_lit_stove_cooks_food_in_pan(self) -> None:
        task = object.__new__(cook_food)
        task.food = object()
        task._skillet_home = None
        task.fire_intensity = 0.75
        task._food_in_pan = True
        task._food_held = lambda: False
        task.cook_steps = 100
        task.doneness = 0.0
        task.max_doneness = 0.0
        task._set_food_color = lambda _value: None

        with patch.object(KitchenS_base_task, "_update_kinematic_tasks"):
            task._update_kinematic_tasks()

        self.assertAlmostEqual(task.doneness, 0.0075)
        self.assertAlmostEqual(task.max_doneness, 0.0075)

    def test_cooking_continues_while_stove_lit_even_if_phase_flags_set(self) -> None:
        """Browning must not freeze while the burner is still on."""
        task = object.__new__(cook_food)
        task.food = object()
        task._skillet_home = None
        task._cook_phase_done = True
        task._grasp_doneness = 0.4
        task.fire_intensity = 0.75
        task._food_in_pan = True
        task._food_held = lambda: False
        task.cook_steps = 100
        task.doneness = 0.4
        task.max_doneness = 0.4
        task._set_food_color = lambda _value: None

        with patch.object(KitchenS_base_task, "_update_kinematic_tasks"):
            task._update_kinematic_tasks()

        self.assertGreater(task.doneness, 0.4)

    def test_cooking_stops_when_fire_off(self) -> None:
        task = object.__new__(cook_food)
        task.food = object()
        task._skillet_home = None
        task.fire_intensity = 0.0
        task._food_in_pan = True
        task._food_held = lambda: False
        task.cook_steps = 100
        task.doneness = 0.5
        task.max_doneness = 0.5
        task._set_food_color = lambda _value: None

        with patch.object(KitchenS_base_task, "_update_kinematic_tasks"):
            task._update_kinematic_tasks()

        self.assertAlmostEqual(task.doneness, 0.5)


    def test_success_window_is_four_seconds_for_all_cook_speeds(self) -> None:
        task = object.__new__(cook_food)
        task.cook_intensity = 0.70
        task.scene = None
        dt = cook_food.SIM_DT_DEFAULT
        for steps in (3076, 1692, 2461, 3691):
            task.cook_steps = steps
            width = task._success_window_width(4.0)
            seconds = width * steps / 0.70 * dt
            self.assertAlmostEqual(seconds, 4.0, places=5)
            lo, hi = task._doneness_range_from_window(0.50, 4.0)
            self.assertAlmostEqual(hi - lo, width, places=5)

    def test_onion_window_shifts_instead_of_clipping_below_four_seconds(self) -> None:
        task = object.__new__(cook_food)
        task.cook_intensity = 0.70
        task.cook_steps = 1692
        task.scene = None
        lo, hi = task._doneness_range_from_window(0.81, 4.0)
        width = task._success_window_width(4.0)
        self.assertAlmostEqual(hi, 1.0, places=5)
        self.assertAlmostEqual(hi - lo, width, places=5)
        self.assertLess(lo, 0.81)


if __name__ == "__main__":
    unittest.main()
