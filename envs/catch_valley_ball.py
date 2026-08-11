"""Catch a valley-ramp ball by pushing a custom open box under the landing.

Same ball / ramp PhysX as ``catch_valley_ball_v1``, but the catcher is a procedural
open box under the landing. Promoted from the former ``catch_valley_ball_v2``;
the cup-place variant lives as ``catch_valley_ball_v1``. The expert slides the box
with a closed-gripper contact push (``catch_cup`` pattern). No end-of-push teleport.
"""

from __future__ import annotations

import numpy as np
import sapien
import sapien.physx

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .catch_valley_ball_v1 import catch_valley_ball_v1
from .utils import *


class catch_valley_ball(catch_valley_ball_v1):
    """Push a custom open box to catch the red ball leaving the valley ramp.

    Options / ball physics match ``catch_valley_ball_v1``. The robot presses the
    box with a closed gripper and slides it across the table to the predicted
    landing; the box is a normal dynamic body (no weld / grasp teleport).
    """

    # Reference: measured 062_plasticbox base1 (~19.2×19.2×7.0 cm). Catcher is
    # a procedural open box at 48% of that footprint (prior 60%, then −20%).
    REF_PLASTIC_HALF_XY = 0.0959
    REF_PLASTIC_HEIGHT = 0.070
    BOX_SIZE_SCALE = 0.48

    CATCHER_MODEL_DEFAULT = "custom_catch_box"
    BOWL_ID_DEFAULT = 0
    BOWL_SCALE_MULT_DEFAULT = 1.0
    # Hold the ball at its drop pose for 3 s at the top of the episode so the box
    # can be pre-positioned. Arms stay free to move during the hold.
    START_FREEZE_S_DEFAULT = 3.0
    BOX_HALF_XY = (
        REF_PLASTIC_HALF_XY * BOX_SIZE_SCALE,
        REF_PLASTIC_HALF_XY * BOX_SIZE_SCALE,
    )
    # Keep walls tall enough that the 3.6 cm ball can settle inside.
    BOX_HEIGHT = max(REF_PLASTIC_HEIGHT * BOX_SIZE_SCALE, 0.040)
    BOX_WALL = 0.007
    BOX_COLOR = [0.28, 0.68, 0.82]
    BOWL_INNER_RADIUS_DEFAULT = BOX_HALF_XY[0] - BOX_WALL
    BOWL_OUTER_RADIUS_DEFAULT = BOX_HALF_XY[0]
    BOWL_HEIGHT_DEFAULT = BOX_HEIGHT
    BOWL_MASS_DEFAULT = 0.12
    BOWL_PLACE_Z_OFFSET = 0.0

    # Procedural box is Z-up (unlike the rotated plasticbox mesh).
    BOX_Q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    PUSH_CONTACT_GAP = 0.012
    PUSH_FINGER_DROP = 0.043
    PUSH_EDGE_MARGIN = 0.030
    PUSH_BEHIND_STANDOFF = 0.09
    PUSH_FINGER_HEIGHT_FRAC = 0.40
    PUSH_LIN_DAMP = 0.85
    PUSH_MU_STATIC = 0.95
    PUSH_MU_DYNAMIC = 0.85
    PUSH_STEP_DEFAULT = 0.040
    PUSH_PLACE_TOL = 0.030
    TABLE_WIDTH = 0.70
    # Must cover catch-side landings (~0.36 m); only blocks runaway off-table aims.
    TABLE_X_LIM = 0.48
    TABLE_Y_MIN = -0.34
    TABLE_Y_MAX = 0.34

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_valley_ball", {})
        self._loaded = False
        self._ball_phase = None
        self._distractor_phase = None
        self._expert_demo = False
        self._bowl_ready = False
        self._bowl_welded = False
        self._arm_ball_contact = False
        self._push_active = False
        self._push_arm = None
        self._box_placed = False
        self._box_half_xy = [float(self.BOX_HALF_XY[0]), float(self.BOX_HALF_XY[1])]
        self._box_height = float(self.BOX_HEIGHT)
        self.distractor = None
        self._distractor_rigid = None
        self.enable_distractor = False
        self.wall_bounce_enabled = False
        self.mirrored = False
        self.side = 1.0
        Base_Task._init_task_env_(self, **kwags)
        self._prepare_push_box()
        # Eval / expert both run the ball immediately (no frozen hold at spawn).
        self._start_ball_motion(expert_demo=False)
        # Interactive teleop: hand the box to PhysX after settle (catch_cup pillow pattern).
        # The box then moves only under gripper contact — no keyboard teleport.
        if bool(
            getattr(self, "_interactive_robot_mode", False)
            or getattr(self, "_interactive_universal_controls", False)
        ):
            self._enable_box_physics()
            self._push_active = True

    def load_actors(self):
        # Parent spawns a temporary mesh catcher only to sample start XY; force a
        # known asset so create_actor succeeds, then swap to the procedural box.
        prev_model = self._cfg.get("catcher_model", None)
        prev_id = self._cfg.get("bowl_id", None)
        self._cfg["catcher_model"] = "062_plasticbox"
        self._cfg["bowl_id"] = 1
        try:
            catch_valley_ball_v1.load_actors(self)
        finally:
            if prev_model is None:
                self._cfg.pop("catcher_model", None)
            else:
                self._cfg["catcher_model"] = prev_model
            if prev_id is None:
                self._cfg.pop("bowl_id", None)
            else:
                self._cfg["bowl_id"] = prev_id
        self._swap_in_custom_box()
        self._place_ball_just_before_ramp()

    def _place_ball_just_before_ramp(self) -> None:
        """Seat the ball on the lane, just upstream of the first board.

        Parent spawns an elevated aerial drop well in front of the entry; v2
        keeps the same roll/launch PhysX but starts on-lane at the ramp lip
        (aligned in Y with ``ball_start``, tiny X/Z clearance only).
        """
        if getattr(self, "ball", None) is None or getattr(self, "ball_start", None) is None:
            return
        standoff = float(self._cfg.get("ball_entry_standoff", 0.012))
        z_clear = float(self._cfg.get("ball_entry_clearance", 0.006))
        self.drop_height = z_clear
        self._drop_forward_travel = standoff
        # Y stays on the ramp lane — not robot-side "in front" of the boards.
        self.ball_drop = np.array(
            [
                float(self.ball_start[0]) - float(self.side) * standoff,
                float(self.ball_start[1]),
                float(self.ball_start[2]) + z_clear,
            ],
            dtype=np.float64,
        )
        self.drop_time = float(self._cfg.get("drop_time", 0.20))
        self.drop_steps = max(1, int(round(self.drop_time * self.SIM_HZ)))
        pose = sapien.Pose(self.ball_drop.tolist())
        self.ball.set_pose(pose)
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
                self._ball_rigid.set_linear_velocity(np.zeros(3))
                self._ball_rigid.set_angular_velocity(np.zeros(3))
                self._ball_rigid.set_kinematic_target(pose)
            except Exception:
                pass
        self._ball_phase = "frozen"

    def _swap_in_custom_box(self) -> None:
        old = getattr(self, "bowl", None)
        if old is None:
            return
        # Keep catch-side x from the temp spawn; Y is re-parked in _prepare_push_box.
        catch_x = float(old.get_pose().p[0])
        ent = old.actor if hasattr(old, "actor") else old
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass

        self.catcher_model = "custom_catch_box"
        self.bowl_id = 0
        self._box_half_xy = [float(self.BOX_HALF_XY[0]), float(self.BOX_HALF_XY[1])]
        self._box_height = float(self.BOX_HEIGHT)
        self.bowl_outer_radius = float(self.BOX_HALF_XY[0])
        self.bowl_inner_radius = float(max(0.03, self.BOX_HALF_XY[0] - self.BOX_WALL))
        self.bowl_height = float(self.BOX_HEIGHT)
        self.bowl_mass = float(self._cfg.get("bowl_mass", self.BOWL_MASS_DEFAULT))

        # Temporary pose; _prepare_push_box moves it just before the ramp in Y.
        self.bowl = self._create_custom_catch_box(
            self._box_pose_at([catch_x, -0.18])
        )
        self.bowl.set_mass(float(self.bowl_mass))
        self.add_prohibit_area(self.bowl, padding=0.05)

    def _create_custom_catch_box(self, pose: sapien.Pose) -> Actor:
        """Open Z-up box: floor + four walls (procedural, no mesh asset)."""
        hx, hy = float(self.BOX_HALF_XY[0]), float(self.BOX_HALF_XY[1])
        hz = 0.5 * float(self.BOX_HEIGHT)
        wt = float(self.BOX_WALL)
        floor_hz = min(0.0035, wt * 0.45)
        # Walls span floor-top → +hz so outer AABB height matches BOX_HEIGHT.
        floor_top = -hz + 2.0 * floor_hz
        side_hz = 0.5 * (hz - floor_top)
        side_z = 0.5 * (floor_top + hz)
        parts = [
            (sapien.Pose([0.0, 0.0, -hz + floor_hz]), [hx, hy, floor_hz]),
            (sapien.Pose([hx - wt * 0.5, 0.0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([-hx + wt * 0.5, 0.0, side_z]), [wt * 0.5, hy, side_hz]),
            (sapien.Pose([0.0, hy - wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
            (sapien.Pose([0.0, -hy + wt * 0.5, side_z]), [hx - wt, wt * 0.5, side_hz]),
        ]

        color = list(self._cfg.get("box_color", self.BOX_COLOR))
        mat = sapien.render.RenderMaterial(base_color=[*color[:3], 1.0])
        mat.roughness = 0.55
        mat.metallic = 0.05
        phys = self.scene.create_physical_material(
            float(self.PUSH_MU_STATIC),
            float(self.PUSH_MU_DYNAMIC),
            0.0,
        )

        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")
        for local_pose, half in parts:
            builder.add_box_collision(
                pose=local_pose, half_size=list(half), material=phys
            )
        builder.set_initial_pose(pose)
        entity = builder.build(name="custom_catch_box")

        render_body = sapien.render.RenderBodyComponent()
        for local_pose, half in parts:
            shape = sapien.render.RenderShapeBox(list(half), mat)
            shape.set_local_pose(local_pose)
            render_body.attach(shape)
        entity.add_component(render_body)

        data = {
            "center": [0.0, 0.0, 0.0],
            "extents": [hx * 2.0, hy * 2.0, hz * 2.0],
            "scale": [1.0, 1.0, 1.0],
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [],
            "contact_points_group": [],
            "contact_points_mask": [],
            "contact_points_description": [],
            "functional_matrix": [],
            "transform_matrix": np.eye(4).tolist(),
        }
        return Actor(entity, data, mass=float(self.bowl_mass))

    def _prepare_push_box(self) -> None:
        """Seat the custom box, measure footprint, and park it for pushing."""
        if getattr(self, "bowl", None) is None:
            return
        self.table_near_y = -0.5 * self.TABLE_WIDTH + float(
            np.asarray(getattr(self, "table_xy_bias", [0.0, 0.0]), dtype=float)[1]
        )
        self.push_step = float(self._cfg.get("push_step", self.PUSH_STEP_DEFAULT))
        self._measure_box_extents()
        xy = np.asarray(self.bowl.get_pose().p[:2], dtype=float)
        # Do NOT park in front of the ramp exit. Sit on the catch-side x, but on
        # the robot-facing side of the boards (just before the ramp in Y) so the
        # expert must push the box up to the landing.
        xy = self._box_spawn_xy(xy[0])
        self._freeze_box(self._box_pose_at(xy))
        rigid = self._get_rigid(self.bowl)
        if rigid is not None:
            try:
                rigid.set_mass(float(self.bowl_mass))
            except Exception:
                pass
            self._tune_box_materials(rigid)

    def _box_spawn_xy(self, catch_x: float | None = None) -> np.ndarray:
        """Catch-side x; full box footprint below the platform's lower Y edge."""
        if catch_x is None:
            catch_x = float(self.side * np.random.uniform(0.30, 0.34))
            if bool(getattr(self, "mirrored", False)):
                catch_x += float(getattr(self, "MIRROR_X_SHIFT", 0.0))
        half = float(max(self._box_half_xy))
        ramp_y = float(getattr(self, "ramp_center_y", 0.08))
        ramp_hw = float(getattr(self, "ramp_half_width", 0.1625))
        gap = float(self._cfg.get("box_spawn_ramp_gap", 0.04))
        # Platform lower edge in Y (robot-facing). Box center must sit low enough
        # that the box's +Y face stays below that edge.
        platform_low_y = ramp_y - ramp_hw
        y = platform_low_y - half - gap
        y_min = float(self.table_near_y + self.PUSH_EDGE_MARGIN)
        y_max = float(platform_low_y - half - 0.01)
        y = float(np.clip(y, y_min, y_max))
        return np.array([float(catch_x), y], dtype=np.float64)

    def _box_pose_at(self, xy) -> sapien.Pose:
        z = float(self.table_top + 0.5 * self._box_height)
        return sapien.Pose(
            [float(xy[0]), float(xy[1]), z],
            self.BOX_Q.tolist(),
        )

    def _box_xy(self) -> np.ndarray:
        return np.asarray(self.bowl.get_pose().p[:2], dtype=np.float64)

    def _clamp_table_xy(self, xy) -> np.ndarray:
        """Keep push aims / freeze poses on the tabletop."""
        out = np.asarray(xy, dtype=np.float64).copy()
        margin = float(max(self._box_half_xy)) + 0.01
        out[0] = float(np.clip(out[0], -self.TABLE_X_LIM + margin, self.TABLE_X_LIM - margin))
        out[1] = float(np.clip(
            out[1],
            self.TABLE_Y_MIN + margin,
            self.TABLE_Y_MAX - margin,
        ))
        return out

    def _measure_box_extents(self) -> None:
        rigid = self._get_rigid(self.bowl)
        if rigid is None:
            return
        try:
            lo, hi = rigid.compute_global_aabb_tight()
        except Exception:
            return
        lo = np.asarray(lo, dtype=np.float64)
        hi = np.asarray(hi, dtype=np.float64)
        half = 0.5 * (hi - lo)
        self._box_half_xy = [float(half[0]), float(half[1])]
        self._box_height = float(max(hi[2] - lo[2], 0.025))
        # Keep success radii consistent with the live footprint.
        outer = float(max(self._box_half_xy))
        wall = float(self.BOX_WALL)
        self.bowl_outer_radius = outer
        self.bowl_inner_radius = max(0.025, outer - wall)
        self.bowl_height = float(self._box_height)

    def _tune_box_materials(self, rigid) -> None:
        if rigid is None:
            return
        try:
            for shape in rigid.get_collision_shapes():
                mat = shape.get_physical_material()
                mat.set_static_friction(float(self.PUSH_MU_STATIC))
                mat.set_dynamic_friction(float(self.PUSH_MU_DYNAMIC))
                mat.set_restitution(0.0)
        except Exception:
            pass

    def _enable_box_physics(self) -> None:
        """Dynamic box on the table — moves only under gripper contact."""
        rigid = self._get_rigid(self.bowl)
        if rigid is None:
            return
        if not self._push_active:
            xy = self._box_xy()
            obj = self.bowl.actor if hasattr(self.bowl, "actor") else self.bowl
            obj.set_pose(self._box_pose_at(xy))
        try:
            rigid.set_kinematic(False)
            rigid.set_mass(float(self.bowl_mass))
            rigid.set_disable_gravity(False)
            # Slide on table: free XY/yaw, lock roll/pitch.
            rigid.set_locked_motion_axes([False, False, False, True, True, False])
            rigid.set_linear_damping(float(self.PUSH_LIN_DAMP))
            rigid.set_angular_damping(2.0)
            try:
                rigid.set_max_linear_velocity(0.28)
            except Exception:
                pass
            if not self._push_active:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            for shape in rigid.get_collision_shapes():
                shape.set_collision_groups([1, 1, 0, 0])
            rigid.wake_up()
        except Exception:
            pass
        self._tune_box_materials(rigid)

    def _freeze_box(self, pose: sapien.Pose | None = None) -> None:
        if self._push_active:
            return
        if pose is None:
            pose = self._box_pose_at(self._box_xy())
        obj = self.bowl.actor if hasattr(self.bowl, "actor") else self.bowl
        obj.set_pose(pose)
        rigid = self._get_rigid(self.bowl)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_disable_gravity(True)
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(pose)
            except Exception:
                pass
        self._push_active = False

    def _push_quat(self, arm_tag):
        _ = arm_tag
        return list(GRASP_DIRECTION_DIC["top_down"])

    def _tcp_pos(self, arm_tag):
        pose = (
            self.robot.get_left_tcp_pose()
            if str(arm_tag) == "left"
            else self.robot.get_right_tcp_pose()
        )
        return np.array(pose[:3], dtype=np.float64)

    def _move_tcp(self, arm_tag, xy, z, quat) -> bool:
        self.plan_success = True
        self.move(
            self.move_to_pose(
                arm_tag,
                [float(xy[0]), float(xy[1]), float(z)] + list(quat),
            )
        )
        return bool(self.plan_success)

    def _finger_drop(self, arm_tag) -> float:
        art = (
            self.robot.left_entity
            if str(arm_tag) == "left"
            else self.robot.right_entity
        )
        lows = []
        try:
            for link in art.get_links():
                name = link.get_name().lower()
                if "finger" not in name and "gripper" not in name:
                    continue
                for c in link.get_components():
                    if isinstance(c, sapien.physx.PhysxRigidBaseComponent):
                        lows.append(float(c.compute_global_aabb_tight()[0][2]))
                        break
        except Exception:
            pass
        if not lows:
            return float(self.PUSH_FINGER_DROP)
        drop = float(self._tcp_pos(arm_tag)[2]) - min(lows)
        return float(np.clip(drop, 0.0, 0.12))

    def _push_box_to_landing(self, arm_tag) -> bool:
        """Shove the custom box to the predicted landing with closed gripper."""
        land_xy = np.array(
            [
                float(self._catch_target_x(self.landing[0])),
                float(self.landing[1]),
            ],
            dtype=np.float64,
        )
        self._measure_box_extents()
        pp0 = self._box_xy()
        delta = land_xy - pp0
        dist = float(np.linalg.norm(delta))
        if dist < 0.02:
            self._box_placed = True
            self._freeze_box(self._box_pose_at(pp0))
            return True

        direction = delta / max(dist, 1e-6)
        half_along = float(
            abs(direction[0]) * self._box_half_xy[0]
            + abs(direction[1]) * self._box_half_xy[1]
        )
        quat = self._push_quat(arm_tag)
        gap = float(self.PUSH_CONTACT_GAP)
        # Clamp only TCP aims — never pose-teleport the box itself.
        behind = self._clamp_table_xy(pp0 - direction * (half_along + self.PUSH_BEHIND_STANDOFF))
        contact = self._clamp_table_xy(pp0 - direction * (half_along + gap))
        y_min = float(self.table_near_y + self.PUSH_EDGE_MARGIN)
        behind[1] = max(float(behind[1]), y_min)
        contact[1] = max(float(contact[1]), y_min)

        self.move(self.close_gripper(arm_tag=arm_tag))
        self._push_arm = arm_tag
        self._push_active = False
        self._freeze_box(self._box_pose_at(pp0))

        drop = self._finger_drop(arm_tag)
        push_z = (
            self.table_top
            + float(self.PUSH_FINGER_HEIGHT_FRAC) * self._box_height
            + drop
        )
        hover_z = self.table_top + 0.18

        self._move_tcp(arm_tag, behind, hover_z, quat)
        self._move_tcp(arm_tag, behind, push_z, quat)
        tcp_low = self._tcp_pos(arm_tag).copy()
        z_hold = float(np.clip(tcp_low[2], push_z - 0.01, push_z + 0.03))

        self._enable_box_physics()
        self._push_active = True
        self._dwell(2)

        self._move_tcp(arm_tag, contact, z_hold, quat)
        pp = self._box_xy()
        into = self._clamp_table_xy(pp - direction * max(half_along - 0.005, gap))
        into[1] = max(float(into[1]), y_min)
        self._move_tcp(arm_tag, into, z_hold, quat)

        step = float(np.clip(self.push_step, 0.02, 0.10))
        place_tol = float(self.PUSH_PLACE_TOL)
        n_chunks = int(np.ceil(dist / step)) + 40
        n_stuck = 0
        prev_pp = self._box_xy().copy()
        for _ in range(max(1, n_chunks)):
            pp = self._box_xy()
            # Abort contact shove if the box is leaving the table (no teleport recover).
            if abs(float(pp[0])) > 0.43 or abs(float(pp[1])) > 0.30:
                break
            err = land_xy - pp
            # Tiny residual X corrections cause glancing shoves that fling the box.
            if abs(float(err[0])) < 0.035:
                err[0] *= 0.2
            err_n = float(np.linalg.norm(err))
            if err_n <= place_tol:
                self._box_placed = True
                break
            # Re-aim every chunk so Y overshoot gets corrected (no fixed heading).
            direction = err / max(err_n, 1e-6)
            half_along = float(
                abs(direction[0]) * self._box_half_xy[0]
                + abs(direction[1]) * self._box_half_xy[1]
            )
            rear_now = pp - direction * (half_along + gap)
            advance = min(step, max(err_n - 0.5 * place_tol, 0.01))
            if err_n < 0.07:
                advance = min(advance, 0.5 * step)
            aim = self._clamp_table_xy(rear_now + direction * advance)
            aim[1] = max(float(aim[1]), y_min)

            moved = float(np.linalg.norm(pp - prev_pp))
            if moved < 0.005:
                n_stuck += 1
                if n_stuck >= 2:
                    z_hold = max(self.table_top + drop + 0.008, z_hold - 0.008)
                    aim = self._clamp_table_xy(
                        pp - direction * max(half_along - 0.01, 0.0)
                    )
                    aim[1] = max(float(aim[1]), y_min)
                    n_stuck = 0
            else:
                n_stuck = 0

            self._move_tcp(arm_tag, aim, z_hold, quat)
            self._dwell(2)
            self.plan_success = True
            # Soft brake near the target so contact inertia cannot coast past it.
            if err_n < 0.08:
                rigid = self._get_rigid(self.bowl)
                if rigid is not None:
                    try:
                        v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
                        scale = 0.20 if err_n < 0.05 else 0.40
                        rigid.set_linear_velocity(scale * v)
                        rigid.set_linear_damping(1.4)
                    except Exception:
                        pass
            prev_pp = self._box_xy().copy()

        # Leave the box where contact physics put it — no teleport / snap to landing.
        self._push_active = False
        rigid = self._get_rigid(self.bowl)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass
        self._dwell(12)
        pp = self._box_xy()
        self._freeze_box(self._box_pose_at(pp))
        self._box_placed = bool(float(np.linalg.norm(pp - land_xy)) <= place_tol + 0.012)

        tcp = self._tcp_pos(arm_tag)
        self._move_tcp(arm_tag, tcp[:2], float(tcp[2] + 0.14), quat)
        self.move(self.back_to_origin(arm_tag))
        self.plan_success = True
        self._push_arm = None
        return self._box_placed

    def _update_kinematic_tasks(self):
        # Skip parent bowl-weld tracking; the box only moves under push contact.
        Base_Task._update_kinematic_tasks(self)
        if not getattr(self, "_loaded", False):
            return
        if self._ball_phase == "released":
            self._physics_step_count += 1
        self._check_arm_ball_contact()
        self._advance_ball()

    def play_once(self):
        arm_tag = ArmTag("left" if self.mirrored else "right")

        # Start the ball immediately; drop/roll/flight advance concurrently with
        # the contact push (same timing model as catch_valley_ball_v1).
        self._start_ball_motion(expert_demo=True)
        self._bowl_ready = False
        self._push_box_to_landing(arm_tag)
        self._bowl_ready = True

        remaining_drop = max(0, self.drop_steps - int(getattr(self, "_drop_i", 0)))
        remaining_roll = max(0, self.roll_steps - int(getattr(self, "_roll_i", 0)))
        self._dwell(
            self._remaining_start_freeze()
            + remaining_drop
            + remaining_roll
            + self.flight_steps
            + self.settle_steps
        )
        self._check_arm_ball_contact()

        self.info["info"] = {
            "{A}": "valley ball",
            "{B}": "custom catch box",
            "{a}": str(arm_tag),
            "{opt}": self._option_label(),
            "{flip}": "mirrored" if self.mirrored else "default",
        }
        if self._box_placed or self.check_success():
            self.plan_success = True
        return self.info

    def check_success(self):
        if not getattr(self, "_loaded", False) or self._ball_phase != "released":
            self._last_fail_reason = "ball_not_released"
            return False
        self._check_arm_ball_contact()
        _, in_bowl, behind_line, _, bowl_position, distractor_in_bowl = self._catch_state()
        if distractor_in_bowl:
            self._last_fail_reason = "distractor_in_box"
            return False
        if self._arm_ball_contact:
            self._last_fail_reason = "arm_ball_contact"
            return False
        if not in_bowl:
            self._last_fail_reason = "ball_not_in_box"
            return False
        if not behind_line:
            side = "left/−x" if float(self.side) < 0.0 else "right/+x"
            self._last_fail_reason = (
                f"box_not_past_red_line({side}: box_x={float(bowl_position[0]):.3f}, "
                f"red_line_x={float(self.red_line_x):.3f})"
            )
            return False
        self._last_fail_reason = ""
        return True
