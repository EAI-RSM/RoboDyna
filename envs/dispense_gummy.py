from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.render
import sapien.physx
import numpy as np


class dispense_gummy(Base_Task):
    """Use a button-controlled belt to catch target-colored gummies.

    The bowl starts at the belt's left endpoint. The right arm presses arrow keys to
    move it between bounded belt stations, while the left arm presses the dispense key.
    Each dispense releases one ball from both tubes; only the configured target color
    should land in the bowl.
    """

    TUBE_CAPACITY_DEFAULT = 4
    BALL_DIAMETER_DEFAULT = 0.03
    BALL_SLOT_GAP_DEFAULT = 0.003
    TUBE_INNER_RADIUS_DEFAULT = 0.017
    TUBE_WALL_THICKNESS_DEFAULT = 0.003
    TUBE_BOTTOM_Z_OFFSET_DEFAULT = 0.075
    TUBE_CENTER_Y_DEFAULT = 0.03
    TUBE_X_LEFT_DEFAULT = -0.06
    TUBE_X_RIGHT_DEFAULT = 0.06

    BOWL_RADIUS_DEFAULT = 0.06
    BOWL_ALIGN_TOL_DEFAULT = 0.035
    BOWL_CATCH_Z_TOL_DEFAULT = 0.025

    KEY_X_DEFAULT = -0.26
    KEY_Y_DEFAULT = -0.13
    KEY_HALF_DEFAULT = [0.028, 0.028, 0.016]
    KEY_HOVER_DIS_DEFAULT = 0.06
    KEY_PRESS_DEPTH_DEFAULT = 0.055
    BELT_THICKNESS_DEFAULT = 0.015
    BELT_MOVE_STEPS_DEFAULT = 30
    BELT_KEY_X_DEFAULT = 0.26
    BELT_KEY_Y_LEFT_DEFAULT = -0.14
    BELT_KEY_Y_RIGHT_DEFAULT = -0.06
    BELT_KEY_PRESS_XY_DEFAULT = 0.045
    BELT_KEY_PRESS_DZ_DEFAULT = 0.15
    EE_TO_TCP = 0.12

    DISPENSE_STEPS_DEFAULT = 24
    PRESS_HOLD_STEPS_DEFAULT = 6
    POST_PRESS_DWELL_DEFAULT = 10

    COLORS = {
        "yellow": [0.96, 0.82, 0.18],
        "blue": [0.20, 0.48, 0.92],
    }
    TUBE_COLOR = [0.84, 0.93, 1.00]
    FRAME_COLOR = [0.48, 0.48, 0.52]
    VERTICAL_CYLINDER_Q = [0.70710678, 0.0, -0.70710678, 0.0]

    def setup_demo(self, **kwags):
        kwags = dict(kwags)
        kwags["use_dynamic"] = False
        self._cfg = kwags.get("task_args", {}).get("dispense_gummy", {})
        self._tube_order = ("left", "right")
        self._tube_records = {side: [] for side in self._tube_order}
        self._tube_stack_colors = {side: [] for side in self._tube_order}
        self._dispensed_count = {side: 0 for side in self._tube_order}
        self._discard_counts = {side: 0 for side in self._tube_order}
        self._active_drops = []
        self._caught_ball_records = []
        self._pending_restack = False
        self.invalid_pattern = False
        self.yellow_caught = 0
        self.yellow_missed = 0
        self.blue_caught = 0
        self.blue_dropped = 0
        self.press_history = []
        self.tube_centers = {}
        self.discard_anchors = {}
        self.table_top = 0.0
        self.bowl = None
        self._bowl_target_x = 0.0
        self._bowl_station_idx = 0
        self._belt_key_latched = {"left": False, "right": False}
        self.belt_keys = {}
        self.belt_key_xy = {}
        super()._init_task_env_(**kwags)
        self._configure_observer_camera()

    # ------------------------------------------------------------------ helpers
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
        return None

    def _make_kinematic(self, entity):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
        except Exception:
            pass
        return rigid

    def _set_entity_pose(self, entity, pose):
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
            except Exception:
                rigid.entity.set_pose(pose)
        else:
            obj = entity.actor if hasattr(entity, "actor") else entity
            obj.set_pose(pose)

    def _configure_observer_camera(self):
        """Frame the fixture closely from the table's upper-right corner."""
        camera = getattr(getattr(self, "cameras", None), "observer_camera", None)
        if camera is None:
            return
        camera_pos = np.array([0.333, 0.280, 1.207], dtype=np.float64)
        look_at = np.array([0.0, 0.0, 0.82], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = np.stack([forward, left, up], axis=1)
        camera_matrix[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(camera_matrix))

    def _slot_pose(self, side, slot_idx):
        x, y = self.tube_centers[side]
        z = self.tube_slot_z[slot_idx]
        return sapien.Pose([x, y, z], [1, 0, 0, 0])

    def _current_bottom_color(self, side):
        idx = self._dispensed_count[side]
        if idx >= len(self._tube_stack_colors[side]):
            return None
        return self._tube_stack_colors[side][idx]

    def _current_target_side(self):
        target_sides = [
            side for side in self._tube_order
            if self._current_bottom_color(side) == self.target_color
        ]
        if len(target_sides) != 1:
            return None
        return target_sides[0]

    def _bowl_center_world(self):
        if self.bowl is None:
            return np.zeros(3, dtype=np.float64)
        return np.asarray(self.bowl.get_pose().p, dtype=np.float64)

    def _advance_bowl_on_belt(self):
        if self.bowl is None or not hasattr(self, "belt_move_steps"):
            return
        pose = self.bowl.get_pose()
        dx = float(self._bowl_target_x - pose.p[0])
        if abs(dx) < 1e-5:
            return
        step = np.sign(dx) * min(abs(dx), self.belt_step_per_sim)
        next_x = float(np.clip(pose.p[0] + step, self.bowl_x_min, self.bowl_x_max))
        if (dx > 0 and next_x > self._bowl_target_x) or (dx < 0 and next_x < self._bowl_target_x):
            next_x = self._bowl_target_x
        self._set_entity_pose(
            self.bowl,
            sapien.Pose([next_x, self.tube_center_y, self.belt_surface_z], pose.q),
        )

    def _request_bowl_station(self, direction):
        last_idx = len(self.bowl_stations) - 1
        self._bowl_station_idx = int(np.clip(self._bowl_station_idx + direction, 0, last_idx))
        self._bowl_target_x = float(self.bowl_stations[self._bowl_station_idx])

    def _detect_belt_key_presses(self):
        if not self.belt_key_xy or not hasattr(self, "robot"):
            return
        try:
            ee = np.asarray(self.robot.get_right_ee_pose()[:3], dtype=np.float64)
        except Exception:
            return
        for side, (key_x, key_y) in self.belt_key_xy.items():
            pressed = bool(
                abs(ee[0] - key_x) <= self.belt_key_press_xy
                and abs(ee[1] - key_y) <= self.belt_key_press_xy
                and ee[2] <= self.belt_key_top_z + self.belt_key_press_dz
            )
            if pressed and not self._belt_key_latched[side]:
                self._request_bowl_station(-1 if side == "left" else 1)
            self._belt_key_latched[side] = pressed

    def _caught_ball_pose(self, slot_idx):
        bowl_p = self._bowl_center_world()
        level = slot_idx // 4
        in_level = slot_idx % 4
        ang = in_level * (0.5 * np.pi)
        radius = 0.018 + 0.006 * level
        z = bowl_p[2] + 0.020 + 0.014 * level
        return sapien.Pose(
            [
                bowl_p[0] + radius * np.cos(ang),
                bowl_p[1] + radius * np.sin(ang),
                z,
            ],
            [1, 0, 0, 0],
        )

    def _update_caught_balls(self):
        for slot_idx, record in enumerate(self._caught_ball_records):
            self._set_entity_pose(record["actor"], self._caught_ball_pose(slot_idx))

    def _discard_pose(self, side, slot_idx):
        anchor = self.discard_anchors[side]
        ring = slot_idx // 4
        in_ring = slot_idx % 4
        ang = in_ring * (0.5 * np.pi)
        radius = 0.028 + 0.010 * ring
        return sapien.Pose(
            [
                anchor[0] + radius * np.cos(ang),
                anchor[1] + radius * np.sin(ang),
                anchor[2] + 0.003 * ring,
            ],
            [1, 0, 0, 0],
        )

    def _reposition_tube_balls(self):
        for side in self._tube_order:
            remaining = self._tube_records[side][self._dispensed_count[side]:]
            for slot_idx, record in enumerate(remaining):
                record["state"] = "tube"
                self._set_entity_pose(record["actor"], self._slot_pose(side, slot_idx))

    def _bowl_aligned_side(self):
        bowl_p = self._bowl_center_world()
        z_ok = abs(bowl_p[2] - self.bowl_catch_z) <= self.bowl_catch_z_tol
        if not z_ok:
            return None

        best_side = None
        best_dist = np.inf
        for side in self._tube_order:
            tube_xy = np.asarray(self.tube_centers[side], dtype=np.float64)
            dist = float(np.linalg.norm(bowl_p[:2] - tube_xy))
            if dist < best_dist:
                best_dist = dist
                best_side = side
        if best_dist <= self.bowl_align_tol:
            return best_side
        return None

    def _dwell(self, steps):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _key_tip_pose(self, tip_z_above_top):
        key_top_z = self.table_top + 2.0 * self.key_half[2]
        tcp_z = key_top_z + tip_z_above_top
        ee_z = tcp_z + self.EE_TO_TCP
        return [self.key_x, self.key_y, ee_z, *GRASP_DIRECTION_DIC["top_down"]]

    def _hover_key(self):
        return self.move_to_pose(ArmTag("left"), self._key_tip_pose(self.key_hover_dis))

    def _belt_key_tip_pose(self, side, tip_z_above_top):
        key_x, key_y = self.belt_key_xy[side]
        tcp_z = self.belt_key_top_z + tip_z_above_top
        return [key_x, key_y, tcp_z + self.EE_TO_TCP, *GRASP_DIRECTION_DIC["top_down"]]

    def _press_belt_key(self, side):
        right = ArmTag("right")
        self.move(self.move_to_pose(right, self._belt_key_tip_pose(side, self.key_hover_dis)))
        self.move(self.move_by_displacement(right, z=-self.key_press_depth))
        self._dwell(2)
        self.move(self.move_by_displacement(right, z=self.key_press_depth))
        self._dwell(self.belt_move_steps + 4)

    def _draw_arrow(self, side, key_x, key_y, z):
        # Author one left arrow, then rigidly rotate it 180 degrees for the right key.
        # This guarantees the two icons are exact opposites.
        rotation = 0.0 if side == "left" else np.pi
        color = [0.95, 0.95, 0.95]
        parts = [
            ("shaft", [0.003, 0.0], 0.0, [0.013, 0.0025, 0.001]),
            ("head_upper", [-0.011, 0.005], 0.75, [0.009, 0.0025, 0.001]),
            ("head_lower", [-0.011, -0.005], -0.75, [0.009, 0.0025, 0.001]),
        ]
        c, s = np.cos(rotation), np.sin(rotation)
        for name, (local_x, local_y), local_yaw, half_size in parts:
            x = key_x + c * local_x - s * local_y
            y = key_y + s * local_x + c * local_y
            yaw = local_yaw + rotation
            q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            create_visual_box(
                self,
                sapien.Pose([x, y, z], q),
                half_size=half_size,
                color=color,
                name=f"{side}_arrow_{name}",
            )

    def _build_transparent_tube(self, name, center_xyz, radius, half_length):
        entity = sapien.Entity()
        entity.set_name(name)
        entity.set_pose(sapien.Pose(center_xyz, self.VERTICAL_CYLINDER_Q))
        render_component = sapien.render.RenderBodyComponent()
        material = sapien.render.RenderMaterial(base_color=[*self.TUBE_COLOR, 0.16])
        material.set_transmission(1.0)
        material.set_transmission_roughness(0.05)
        material.set_roughness(0.08)
        render_component.attach(
            sapien.render.RenderShapeCylinder(
                radius=radius,
                half_length=half_length,
                material=material,
            )
        )
        entity.add_component(render_component)
        self.scene.add_entity(entity)
        return entity

    def _validate_stack_pattern(self):
        allowed = {"yellow", "blue"}
        lengths = [len(self._tube_stack_colors[side]) for side in self._tube_order]
        if lengths[0] != lengths[1]:
            raise ValueError("dispense_gummy requires left_stack and right_stack to have the same length")
        if lengths[0] > self.tube_capacity:
            raise ValueError("dispense_gummy stack length exceeds tube_capacity")
        for depth in range(lengths[0]):
            left_color = self._tube_stack_colors["left"][depth]
            right_color = self._tube_stack_colors["right"][depth]
            if left_color not in allowed or right_color not in allowed:
                raise ValueError("dispense_gummy supports only yellow and blue gummy colors")
            if {left_color, right_color} != allowed:
                raise ValueError(
                    "dispense_gummy requires exactly one yellow and one blue bottom ball on each press"
                )

    def _request_dispense(self):
        if self._active_drops:
            return

        target_side = self._current_target_side()
        bowl_side = self._bowl_aligned_side()
        event = {
            "press_index": int(len(self.press_history)),
            "target_side": target_side,
            "bowl_side": bowl_side,
            "bottom_colors": {side: self._current_bottom_color(side) for side in self._tube_order},
        }
        self.press_history.append(event)

        if target_side is None:
            self.invalid_pattern = True
            return

        for side in self._tube_order:
            idx = self._dispensed_count[side]
            if idx >= len(self._tube_records[side]):
                continue
            record = self._tube_records[side][idx]
            caught = bowl_side == side
            if caught:
                end_pose = sapien.Pose(
                    [*self._bowl_center_world()[:2], self.bowl_catch_z + 0.01],
                    [1, 0, 0, 0],
                )
            else:
                end_pose = self._discard_pose(side, self._discard_counts[side])

            record["state"] = "dropping"
            self._active_drops.append(
                {
                    "side": side,
                    "color": record["color"],
                    "record": record,
                    "step": 0,
                    "caught": caught,
                    "start_pose": record["actor"].get_pose(),
                    "end_pose": end_pose,
                }
            )
            self._dispensed_count[side] += 1

        self._pending_restack = True

    def _finish_drop(self, drop):
        side = drop["side"]
        color = drop["color"]
        record = drop["record"]

        if drop["caught"]:
            record["state"] = "caught"
            self._caught_ball_records.append(record)
            if color == "yellow":
                self.yellow_caught += 1
            else:
                self.blue_caught += 1
            return

        record["state"] = "discarded"
        self._set_entity_pose(record["actor"], self._discard_pose(side, self._discard_counts[side]))
        self._discard_counts[side] += 1
        if color == "yellow":
            self.yellow_missed += 1
        else:
            self.blue_dropped += 1

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = 0.74 + self.table_z_bias

        self.ball_diameter = float(cfg.get("ball_diameter", self.BALL_DIAMETER_DEFAULT))
        self.ball_radius = 0.5 * self.ball_diameter
        self.ball_slot_gap = float(cfg.get("ball_slot_gap", self.BALL_SLOT_GAP_DEFAULT))
        self.tube_capacity = int(cfg.get("tube_capacity", self.TUBE_CAPACITY_DEFAULT))
        self.tube_inner_radius = float(cfg.get("tube_inner_radius", self.TUBE_INNER_RADIUS_DEFAULT))
        self.tube_wall_thickness = float(cfg.get("tube_wall_thickness", self.TUBE_WALL_THICKNESS_DEFAULT))
        self.tube_bottom_z_offset = float(cfg.get("tube_bottom_z_offset", self.TUBE_BOTTOM_Z_OFFSET_DEFAULT))
        self.tube_center_y = float(cfg.get("tube_center_y", self.TUBE_CENTER_Y_DEFAULT))
        self.key_x = float(cfg.get("key_x", self.KEY_X_DEFAULT))
        self.key_y = float(cfg.get("key_y", self.KEY_Y_DEFAULT))
        self.key_half = list(cfg.get("key_half", self.KEY_HALF_DEFAULT))
        self.key_hover_dis = float(cfg.get("key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        self.key_press_depth = float(cfg.get("key_press_depth", self.KEY_PRESS_DEPTH_DEFAULT))
        self.dispense_steps = int(cfg.get("dispense_steps", self.DISPENSE_STEPS_DEFAULT))
        self.press_hold_steps = int(cfg.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.post_press_dwell = int(cfg.get("post_press_dwell", self.POST_PRESS_DWELL_DEFAULT))
        self.bowl_align_tol = float(cfg.get("bowl_align_tol", self.BOWL_ALIGN_TOL_DEFAULT))
        self.bowl_catch_z_tol = float(cfg.get("bowl_catch_z_tol", self.BOWL_CATCH_Z_TOL_DEFAULT))
        self.bowl_radius = float(cfg.get("bowl_radius", self.BOWL_RADIUS_DEFAULT))
        self.belt_thickness = float(cfg.get("belt_thickness", self.BELT_THICKNESS_DEFAULT))
        self.belt_move_steps = int(cfg.get("belt_move_steps", self.BELT_MOVE_STEPS_DEFAULT))
        self.belt_key_press_xy = float(cfg.get("belt_key_press_xy", self.BELT_KEY_PRESS_XY_DEFAULT))
        self.belt_key_press_dz = float(cfg.get("belt_key_press_dz", self.BELT_KEY_PRESS_DZ_DEFAULT))
        self.target_color = str(cfg.get("target_color", "yellow")).strip().lower()
        if self.target_color not in self.COLORS:
            raise ValueError("dispense_gummy target_color must be yellow or blue")
        if self.tube_inner_radius <= self.ball_radius:
            raise ValueError("dispense_gummy tube_inner_radius must exceed the gummy radius")

        self.tube_centers = {
            "left": np.array(
                [
                    float(cfg.get("tube_x_left", self.TUBE_X_LEFT_DEFAULT)),
                    self.tube_center_y,
                ],
                dtype=np.float64,
            ),
            "right": np.array(
                [
                    float(cfg.get("tube_x_right", self.TUBE_X_RIGHT_DEFAULT)),
                    self.tube_center_y,
                ],
                dtype=np.float64,
            ),
        }
        self.discard_anchors = {
            side: np.array(
                [
                    self.tube_centers[side][0],
                    self.tube_centers[side][1] + 0.14,
                    self.table_top + self.ball_radius,
                ],
                dtype=np.float64,
            )
            for side in self._tube_order
        }

        self._tube_stack_colors = {
            "left": [str(c).strip().lower() for c in cfg.get("left_stack", ["yellow", "blue", "yellow", "blue"])],
            "right": [str(c).strip().lower() for c in cfg.get("right_stack", ["blue", "yellow", "blue", "yellow"])],
        }
        self._validate_stack_pattern()
        self.num_presses = len(self._tube_stack_colors["left"])
        self.total_yellow = self.num_presses
        self.total_target = self.num_presses

        stack_span = self.tube_capacity * self.ball_diameter + max(0, self.tube_capacity - 1) * self.ball_slot_gap
        self.tube_half_length = 0.5 * stack_span + 0.02
        tube_outer_radius = self.tube_inner_radius + self.tube_wall_thickness
        bowl_diameter = 2.0 * self.bowl_radius
        self.belt_x_min = float(min(center[0] for center in self.tube_centers.values()) - tube_outer_radius - bowl_diameter)
        self.belt_x_max = float(max(center[0] for center in self.tube_centers.values()) + tube_outer_radius + bowl_diameter)
        self.bowl_x_min = self.belt_x_min + self.bowl_radius
        self.bowl_x_max = self.belt_x_max - self.bowl_radius
        self.belt_surface_z = self.table_top + self.belt_thickness
        self.bowl_catch_z = self.belt_surface_z
        self.bowl_stations = [
            self.bowl_x_min,
            float(self.tube_centers["left"][0]),
            float(self.tube_centers["right"][0]),
            self.bowl_x_max,
        ]
        station_gaps = np.diff(self.bowl_stations)
        self.belt_step_per_sim = float(max(station_gaps) / max(1, self.belt_move_steps))
        self._bowl_station_idx = 0
        self._bowl_target_x = self.bowl_stations[0]
        self.tube_bottom_z = self.belt_surface_z + self.tube_bottom_z_offset
        self.tube_slot_z = [
            self.tube_bottom_z + self.ball_radius + 0.008 + i * (self.ball_diameter + self.ball_slot_gap)
            for i in range(self.tube_capacity)
        ]

        belt_center_x = 0.5 * (self.belt_x_min + self.belt_x_max)
        belt_half_x = 0.5 * (self.belt_x_max - self.belt_x_min)
        self.belt = create_box(
            self,
            pose=sapien.Pose(
                [belt_center_x, self.tube_center_y, self.table_top + 0.5 * self.belt_thickness],
                [1, 0, 0, 0],
            ),
            half_size=[belt_half_x, tube_outer_radius, 0.5 * self.belt_thickness],
            color=[0.10, 0.10, 0.12],
            is_static=True,
            name="gummy_bowl_belt",
        )

        frame_top_z = self.tube_bottom_z + 2.0 * self.tube_half_length + 0.04
        frame_center_x = 0.5 * (self.tube_centers["left"][0] + self.tube_centers["right"][0])
        frame_half_x = 0.5 * abs(self.tube_centers["right"][0] - self.tube_centers["left"][0]) + 0.05
        create_box(
            self,
            pose=sapien.Pose([frame_center_x, self.tube_center_y, frame_top_z], [1, 0, 0, 0]),
            half_size=[frame_half_x, 0.014, 0.014],
            color=self.FRAME_COLOR,
            is_static=True,
            name="gummy_frame_bar",
        )
        for side in self._tube_order:
            tube_top_z = self.tube_bottom_z + 2.0 * self.tube_half_length
            hanger_center_z = 0.5 * (frame_top_z + tube_top_z)
            hanger_half_z = 0.5 * abs(frame_top_z - tube_top_z)
            create_box(
                self,
                pose=sapien.Pose([self.tube_centers[side][0], self.tube_centers[side][1], hanger_center_z], [1, 0, 0, 0]),
                half_size=[0.008, 0.008, hanger_half_z],
                color=self.FRAME_COLOR,
                is_static=True,
                name=f"gummy_hanger_{side}",
            )
            self._build_transparent_tube(
                name=f"gummy_tube_{side}",
                center_xyz=[
                    float(self.tube_centers[side][0]),
                    float(self.tube_centers[side][1]),
                    float(self.tube_bottom_z + self.tube_half_length),
                ],
                radius=self.tube_inner_radius + self.tube_wall_thickness,
                half_length=self.tube_half_length,
            )

        bowl_pose = sapien.Pose(
            [self.bowl_x_min, self.tube_center_y, self.belt_surface_z],
            [0.5, 0.5, 0.5, 0.5],
        )
        self.bowl_id = int(cfg.get("bowl_id", 1))
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=True,
            is_static=False,
        )
        self.bowl.set_mass(0.06)
        self._make_kinematic(self.bowl)
        self.add_prohibit_area(self.bowl, padding=0.05)
        self.add_prohibit_area(self.belt, padding=0.03)

        # The red key on the left dispenses one gummy from each tube.
        create_box(
            self,
            pose=sapien.Pose([self.key_x, self.key_y, self.table_top + 0.010], [1, 0, 0, 0]),
            half_size=[0.040, 0.040, 0.010],
            color=[0.36, 0.36, 0.40],
            is_static=True,
            name="dispense_key_base",
        )
        self.dispense_key = create_box(
            self,
            pose=sapien.Pose([self.key_x, self.key_y, self.table_top + self.key_half[2]], [1, 0, 0, 0]),
            half_size=self.key_half,
            color=[0.86, 0.22, 0.18],
            is_static=True,
            name="dispense_key",
        )

        # Two arrow-labeled keys on the right move the bowl by one bounded belt station.
        belt_key_x = float(cfg.get("belt_key_x", self.BELT_KEY_X_DEFAULT))
        self.belt_key_xy = {
            "left": (belt_key_x, float(cfg.get("belt_key_y_left", self.BELT_KEY_Y_LEFT_DEFAULT))),
            "right": (belt_key_x, float(cfg.get("belt_key_y_right", self.BELT_KEY_Y_RIGHT_DEFAULT))),
        }
        self.belt_key_top_z = self.table_top + 2.0 * self.key_half[2]
        for side, (key_x, key_y) in self.belt_key_xy.items():
            create_box(
                self,
                pose=sapien.Pose([key_x, key_y, self.table_top + 0.010]),
                half_size=[0.040, 0.034, 0.010],
                color=[0.28, 0.28, 0.31],
                is_static=True,
                name=f"belt_key_base_{side}",
            )
            self.belt_keys[side] = create_box(
                self,
                pose=sapien.Pose([key_x, key_y, self.table_top + self.key_half[2]]),
                half_size=self.key_half,
                color=[0.18, 0.48, 0.82],
                is_static=True,
                name=f"belt_key_{side}",
            )
            self._draw_arrow(side, key_x, key_y, self.belt_key_top_z + 0.0015)
            self.add_prohibit_area(self.belt_keys[side], padding=0.04)

        for side in self._tube_order:
            for depth, color in enumerate(self._tube_stack_colors[side]):
                actor = create_sphere(
                    self,
                    pose=self._slot_pose(side, depth),
                    radius=self.ball_radius,
                    color=self.COLORS[color],
                    is_static=False,
                    name=f"{side}_gummy_{depth}_{color}",
                )
                rigid = self._make_kinematic(actor)
                self._tube_records[side].append(
                    {
                        "actor": actor,
                        "rigid": rigid,
                        "color": color,
                        "state": "tube",
                    }
                )

        self._reposition_tube_balls()

    # ---------------------------------------------------------------- scene motion
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if self.bowl is None:
            return
        self._detect_belt_key_presses()
        self._advance_bowl_on_belt()
        self._update_caught_balls()

        if not self._active_drops:
            return

        remaining_drops = []
        for drop in self._active_drops:
            drop["step"] += 1
            frac = min(1.0, drop["step"] / max(1, self.dispense_steps))
            smooth = frac * frac * (3.0 - 2.0 * frac)

            start_p = np.asarray(drop["start_pose"].p, dtype=np.float64)
            end_p = np.asarray(drop["end_pose"].p, dtype=np.float64)
            target_p = start_p + smooth * (end_p - start_p)
            self._set_entity_pose(drop["record"]["actor"], sapien.Pose(target_p.tolist(), [1, 0, 0, 0]))

            if frac >= 1.0:
                self._finish_drop(drop)
            else:
                remaining_drops.append(drop)

        self._active_drops = remaining_drops
        if self._pending_restack and not self._active_drops:
            self._reposition_tube_balls()
            self._pending_restack = False

    # ------------------------------------------------------------------ policy
    def play_once(self):
        left = ArmTag("left")

        self.move(self.close_gripper(left))
        self.move(self._hover_key())

        for _ in range(self.num_presses):
            target_side = self._current_target_side()
            if target_side is None:
                self.invalid_pattern = True
                break

            target_station = 1 if target_side == "left" else 2
            while self._bowl_station_idx < target_station:
                self._press_belt_key("right")
            while self._bowl_station_idx > target_station:
                self._press_belt_key("left")

            self.move(self._hover_key())
            self.move(self.move_by_displacement(left, z=-self.key_press_depth))
            self._request_dispense()
            self._dwell(self.press_hold_steps)
            self.move(self.move_by_displacement(left, z=self.key_press_depth))
            self._dwell(self.dispense_steps + self.post_press_dwell)

        self._dwell(max(12, self.save_freq or 12))
        self.info["info"] = {
            "{A}": f"{self.target_color} gummy balls",
            "{B}": f"002_bowl/base{self.bowl_id}",
            "{C}": "dual transparent gummy tubes",
            "{a}": "both arms",
        }
        return self.info

    # ------------------------------------------------------------------ success
    def check_success(self):
        target_caught = self.yellow_caught if self.target_color == "yellow" else self.blue_caught
        target_missed = self.yellow_missed if self.target_color == "yellow" else self.blue_dropped
        distractor_caught = self.blue_caught if self.target_color == "yellow" else self.yellow_caught
        return bool(
            (not self.invalid_pattern)
            and target_caught == self.total_target
            and target_missed == 0
            and distractor_caught == 0
        )

    def get_obs(self):
        obs = super().get_obs()
        obs["dispense_gummy"] = {
            "yellow_caught": int(self.yellow_caught),
            "yellow_missed": int(self.yellow_missed),
            "blue_caught": int(self.blue_caught),
            "blue_dropped": int(self.blue_dropped),
            "target_color": self.target_color,
            "invalid_pattern": bool(self.invalid_pattern),
            "press_count": int(len(self.press_history)),
            "bowl_station": int(self._bowl_station_idx),
            "bowl_x": float(self._bowl_center_world()[0]),
            "current_bottom_colors": {
                side: self._current_bottom_color(side) for side in self._tube_order
            },
            "remaining_left": list(self._tube_stack_colors["left"][self._dispensed_count["left"]:]),
            "remaining_right": list(self._tube_stack_colors["right"][self._dispensed_count["right"]:]),
        }
        return obs
