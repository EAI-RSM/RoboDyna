from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
from scene_utils import print_c, get_quat_euler


class place_block_belt(Base_Task):
    """Set a tall, top-heavy block onto a moving conveyor belt so it drops into a
    bowl placed at the belt exit.

    A flat conveyor belt spans the mid zone of the RIGHT half of the table and
    carries anything resting on it laterally (along +x) at a constant belt speed.
    A tall block with a high center of mass spawns in the near zone within the
    right arm's reach. The scripted expert:
      1) grasps the block in the near zone,
      2) lifts it and carries it above the belt surface,
      3) MATCHES the gripper's horizontal (x) velocity to the belt speed by a
         short move_by_displacement along +x just before release, so the block is
         not sheared/tipped by a velocity mismatch on release,
      4) opens the gripper; the belt then carries the upright block to the exit,
         where it falls into the bowl on the table.

    The belt motion is driven step-by-step in the overridden _update_kinematic_tasks:
    once the block is released and resting on the belt, the block is set kinematic
    and advanced along +x by belt_speed * dt every physics step. Tilt of the block
    away from vertical is recorded each step.

    Single-arm (right). Key mechanic: release at belt speed to keep the top-heavy
    block upright. Metric: tilt_score = clamp(1 - max_tilt_angle/theta_max, 0, 1).
    """

    # ----------------------------------------------------------------- params (class defaults)
    BELT_SPEED_DEFAULT = 0.4          # m/s lateral (+x) belt surface speed
    BELT_X_START_DEFAULT = -0.2        # near-center x where the belt begins (right half)
    BELT_X_END_DEFAULT = 0.2         # outer x where the belt ends (success line)
    BELT_Y_DEFAULT = -0.02             # y of the belt centerline (mid zone)
    BELT_HALF_W_DEFAULT = 0.15         # belt half-width in y
    BELT_RIDE_STEPS_DEFAULT = 600      # max physics steps to let the block ride the belt
    BELT_RELEASE_DELAY_STEPS_DEFAULT = 12  # wait after opening so the block fully drops first
    BELT_BAND_SPACING_DEFAULT = 0.055  # spacing between visible tread bands
    BELT_BAND_HALF_X_DEFAULT = 0.012   # half-length of each moving tread band along x
    BELT_BAND_HALF_Z_DEFAULT = 0.0006  # thin render-only band height
    BELT_BAND_Z_OFFSET_DEFAULT = 0.0008  # lift above the top face to avoid z-fighting
    BOWL_X_OFFSET_DEFAULT = 0.04       # bowl center offset beyond the physical belt exit
    BOWL_Y_OFFSET_DEFAULT = 0.0        # bowl shares the belt lane by default
    BOWL_Y_RANDOM_RANGE_DEFAULT = 0.08 # per-episode y randomization around bowl_y_offset
    BOWL_MOVE_ENABLED_DEFAULT = True  # optionally oscillate the bowl along the belt y axis
    BOWL_MOVE_SPEED_DEFAULT = 0.30     # m/s signed speed used by the y-oscillation
    BOWL_MOVE_RANGE_DEFAULT = 0.12     # half-range of the bowl motion around its start y
    BOWL_INNER_RADIUS_DEFAULT = 0.065  # geometric success radius around bowl center
    BOWL_SUCCESS_HEIGHT_DEFAULT = 0.10 # block center must settle below this height above bowl base
    BOWL_SETTLE_STEPS_DEFAULT = 140    # extra physics steps after the block leaves the belt

    BLOCK_HALF_W_DEFAULT = 0.02       # block half-width (x,y) -> narrow base (top-heavy)
    BLOCK_HALF_H_DEFAULT = 0.02      # block half-height (z) -> 12 cm tall
    BLOCK_MASS_DEFAULT = 0.05          # block mass (kg)
    BLOCK_COM_FRAC_DEFAULT = 0.55      # COM height as fraction of half-height above center (top-heavy)

    THETA_MAX_DEG_DEFAULT = 30.0       # tilt threshold for the metric / success

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's task_args block BEFORE init
        self._cfg = kwags.get("task_args", {}).get("place_block_belt", {})
        # reset per-episode belt/tilt state up front so a reused instance can't leak the
        # belt drive into the next episode's load_camera() (which steps the sim before
        # load_actors() rebinds self.block).
        self._belt_active = False
        self._released = False
        self._block_kinematic = False
        self.max_tilt_deg = 0.0
        self.tilt_score = 0.0
        self.reached_end = False
        self._block_dyn = None
        self._release_q = [1.0, 0.0, 0.0, 0.0]
        self._belt_q = [1.0, 0.0, 0.0, 0.0]
        self._belt_bands = []
        self._belt_band_offsets = np.zeros(0, dtype=np.float64)
        self._belt_band_phase = 0.0
        self._belt_band_start_x = 0.0
        self._belt_band_span = 0.0
        self._belt_band_z = 0.0
        self._belt_band_half_x = 0.0
        self._belt_band_min_x = 0.0
        self._belt_band_max_x = 0.0
        self._belt_drop_x = 0.0
        self._block_dropped = False
        self._drop_settle_steps = 0
        self.placed_on_belt = False
        self.dropped_at_start_left = False
        self.in_bowl = False
        self.bowl_id = None
        self._bowl_dyn = None
        self.bowl_start_z = 0.0
        self._bowl_base_x = 0.0
        self._bowl_base_y = 0.0
        self._bowl_move_dir = 1.0
        self._belt_start_zone_max_x = 0.0
        self._belt_contact_latched = False
        self._belt_contact_x = None
        self._belt_contact_y = None
        super()._init_task_env_(**kwags)

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.belt_speed = float(cfg.get("belt_speed", self.BELT_SPEED_DEFAULT))
        self.belt_x_start = float(cfg.get("belt_x_start", self.BELT_X_START_DEFAULT))
        self.belt_x_end = float(cfg.get("belt_x_end", self.BELT_X_END_DEFAULT))
        self.belt_y = float(cfg.get("belt_y", self.BELT_Y_DEFAULT))
        self.belt_half_w = float(cfg.get("belt_half_w", self.BELT_HALF_W_DEFAULT))
        self.belt_ride_steps = int(cfg.get("belt_ride_steps", self.BELT_RIDE_STEPS_DEFAULT))
        self.belt_release_delay_steps = int(cfg.get("belt_release_delay_steps", self.BELT_RELEASE_DELAY_STEPS_DEFAULT))
        self.block_half_w = float(cfg.get("block_half_w", self.BLOCK_HALF_W_DEFAULT))
        self.block_half_h = float(cfg.get("block_half_h", self.BLOCK_HALF_H_DEFAULT))
        self.block_mass = float(cfg.get("block_mass", self.BLOCK_MASS_DEFAULT))
        self.block_com_frac = float(cfg.get("block_com_frac", self.BLOCK_COM_FRAC_DEFAULT))
        self.theta_max_deg = float(cfg.get("theta_max_deg", self.THETA_MAX_DEG_DEFAULT))
      
        self.bowl_x_offset = float(cfg.get("bowl_x_offset", self.BOWL_X_OFFSET_DEFAULT))
        self.bowl_y_offset = float(cfg.get("bowl_y_offset", self.BOWL_Y_OFFSET_DEFAULT))
        self.bowl_y_random_range = float(cfg.get("bowl_y_random_range", self.BOWL_Y_RANDOM_RANGE_DEFAULT))
        self.bowl_move_enabled = bool(cfg.get("bowl_move_enabled", self.BOWL_MOVE_ENABLED_DEFAULT))
        self.bowl_move_speed = float(cfg.get("bowl_move_speed", self.BOWL_MOVE_SPEED_DEFAULT))
        self.bowl_move_range = float(cfg.get("bowl_move_range", self.BOWL_MOVE_RANGE_DEFAULT))
        self.bowl_inner_radius = float(cfg.get("bowl_inner_radius", self.BOWL_INNER_RADIUS_DEFAULT))
        self.bowl_success_height = float(cfg.get("bowl_success_height", self.BOWL_SUCCESS_HEIGHT_DEFAULT))
        self.bowl_settle_steps = int(cfg.get("bowl_settle_steps", self.BOWL_SETTLE_STEPS_DEFAULT))

        # randomize within the configured envelope (seed-driven) for variety
        self.belt_speed = float(np.random.uniform(self.belt_speed * 0.8, self.belt_speed * 1.2))
        self.block_half_h = float(np.random.uniform(self.block_half_h * 0.9, self.block_half_h * 1.1))
        self.block_mass = float(np.random.uniform(self.block_mass * 0.8, self.block_mass * 1.2))

        belt_step_dx = max(self.belt_speed * self.scene.get_timestep(), 1e-6)
        # min_ride_steps = int(np.ceil((self.belt_x_end - self.belt_x_start + 0.04) / belt_step_dx))
        # self.belt_ride_steps = max(self.belt_ride_steps, min_ride_steps)

        table_z = 0.74 + self.table_z_bias
        self.table_top_z = table_z

        # ------- support box to elevate the belt 10 cm above the table
        support_box_height = 0.10  # 10 cm elevation
        support_box_half_h = support_box_height * 0.5
        belt_cx = 0.5 * (self.belt_x_start + self.belt_x_end)
        belt_half_len_x = 0.5 * (self.belt_x_end - self.belt_x_start)
        belt_xy_padding = 0.02
        self._belt_cx = belt_cx
        self._belt_half_len_x = belt_half_len_x + belt_xy_padding
        self._belt_half_w = self.belt_half_w + belt_xy_padding
        self._belt_start_zone_max_x = float(
            self.belt_x_start + max(0.05, 0.18 * (self.belt_x_end - self.belt_x_start))
        )

        support_box_pose = sapien.Pose(
            p=[belt_cx, self.belt_y, table_z + support_box_half_h],
            q=[1, 0, 0, 0],
        )
        self.support_box = create_box(
            scene=self,
            pose=support_box_pose,
            half_size=(self._belt_half_len_x, self._belt_half_w, support_box_half_h),
            color=(0.25, 0.25, 0.28),
            name="belt_support",
            is_static=True,
        )

        # ------- belt (match the holding/support box footprint)
        self.belt_thickness = 0.012
        belt_pose = sapien.Pose(
            p=[belt_cx, self.belt_y, table_z + support_box_height + self.belt_thickness * 0.5],
            q=[1, 0, 0, 0],
        )
        self.belt = create_box(
            scene=self,
            pose=belt_pose,
            half_size=(self._belt_half_len_x, self._belt_half_w, self.belt_thickness * 0.5),
            color=(0.10, 0.10, 0.12),
            name="conveyor_belt",
            is_static=True,
        )
        self.belt_surface_z = table_z + support_box_height + self.belt_thickness  # top face of the belt
        self._init_belt_visual_motion()

        # ------- receiving bowl on the table at the RIGHT / exit side of the belt
        bowl_x_clearance = max(self.bowl_x_offset, self.bowl_inner_radius + 0.005)
        bowl_x = self._belt_cx + self._belt_half_len_x + bowl_x_clearance
        bowl_lane_low = float(self.belt_y - self.belt_half_w)
        bowl_lane_high = float(self.belt_y + self.belt_half_w)
        if self.bowl_move_enabled:
            bowl_y = bowl_lane_low
        else:
            bowl_y = float(np.random.uniform(bowl_lane_low, bowl_lane_high))
        self.bowl_id = int(np.random.choice([1, 2, 3, 4, 5, 6, 7]))
        self.bowl = create_actor(
            self,
            pose=sapien.Pose([bowl_x, bowl_y, table_z], [0.5, 0.5, 0.5, 0.5]),
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=False,
            is_static=not self.bowl_move_enabled,
        )
        self._bowl_dyn = None
        for c in self.bowl.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._bowl_dyn = c
                if self.bowl_move_enabled:
                    try:
                        c.set_disable_gravity(True)
                        c.set_kinematic(True)
                    except Exception:
                        pass
        self.bowl_start_z = float(self.bowl.get_pose().p[2])
        self._bowl_base_x = float(bowl_x)
        self._bowl_base_y = float(bowl_y)
        self._bowl_move_dir = 1.0 if self.bowl_move_enabled else float(np.random.choice([-1.0, 1.0]))
        # Release only after the whole block has cleared the physical belt edge;
        # otherwise the dynamic block can remain supported and appear to stop there.
        self._belt_drop_x = self._belt_cx + self._belt_half_len_x + self.block_half_w + 0.005
        min_ride_steps = int(np.ceil((self._belt_drop_x - self.belt_x_start + 0.02) / belt_step_dx)) + self.bowl_settle_steps
        self.belt_ride_steps = max(self.belt_ride_steps, min_ride_steps)

        # ------- tall top-heavy block, spawned on the LEFT/start side of the belt within the
        # right arm's reach (spawned on table surface, will be lifted above the elevated belt)
        block_x_min = max(0.015, self.belt_x_start - 0.06)
        block_x_max = max(block_x_min + 0.01, self.belt_x_start - 0.01)
        block_x = float(np.random.uniform(block_x_min, block_x_max))
        block_y = float(np.random.uniform(self.belt_y + 0.05, self.belt_y + 0.12))
        block_pose = rand_pose(
            xlim=[self.belt_x_start-0.3, self.belt_x_start-0.05],
            ylim=[block_y, block_y],
            zlim=[table_z + self.block_half_h],
            qpos=get_quat_euler([0,90,0]), #[1, 0, 0, 0],
            rotate_rand=False,
            rotate_lim=[0, 0, np.pi / 18],
        )
        self.block = create_box(
            scene=self,
            pose=block_pose,
            half_size=(self.block_half_w, self.block_half_w, self.block_half_h),
            color=(0.85, 0.35, 0.10),
            name="tall_block",
            boxtype="long",
            is_static=False,
        )
        self.block.set_mass(self.block_mass)

        # make the block TOP-HEAVY: raise its center of mass well above the geometric center
        self._block_dyn = None
        for c in self.block.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                self._block_dyn = c
                try:
                    cm = c.get_cmass_local_pose()
                    com_z = self.block_com_frac * self.block_half_h
                    c.set_cmass_local_pose(sapien.Pose(p=[cm.p[0], cm.p[1], com_z], q=cm.q))
                except Exception:
                    pass

        self.add_prohibit_area(self.belt, padding=0.03)
        self.add_prohibit_area(self.bowl, padding=0.04)
        self.add_prohibit_area(self.block, padding=0.05)

        # ------- belt / tilt bookkeeping
        self._belt_active = False
        self._released = False
        self._block_kinematic = False
        self.max_tilt_deg = 0.0
        self.tilt_score = 0.0
        self.reached_end = False
        self._ride_steps_done = 0
        self._block_dropped = False
        self._drop_settle_steps = 0
        self.placed_on_belt = False
        self.dropped_at_start_left = False
        self.in_bowl = False
        self._bowl_move_dir = float(np.random.choice([-1.0, 1.0]))
        self._belt_contact_latched = False
        self._belt_contact_x = None
        self._belt_contact_y = None

    # ----------------------------------------------------------------- tilt helper
    def _current_tilt_deg(self):
        # angle between the block's local +z (its long axis) and the world +z axis
        m = self.block.get_pose().to_transformation_matrix()
        up = m[:3, 2]
        up = up / (np.linalg.norm(up) + 1e-9)
        cos_t = float(np.clip(abs(up[2]), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_t)))

    def _on_belt(self):
        p = self.block.get_pose().p
        near_belt = (
            abs(p[0] - self._belt_cx) <= (self._belt_half_len_x + 0.03)
            and abs(p[1] - self.belt_y) <= (self._belt_half_w + 0.03)
            and p[2] < (self.belt_surface_z + self.block_half_h + 0.03)
        )
        return bool(near_belt)

    def _block_in_bowl(self):
        if getattr(self, "bowl", None) is None or getattr(self, "block", None) is None:
            return False
        block_p = np.array(self.block.get_pose().p, dtype=np.float64)
        bowl_p = np.array(self.bowl.get_pose().p, dtype=np.float64)
        xy_close = float(np.linalg.norm(block_p[:2] - bowl_p[:2])) <= self.bowl_inner_radius
        below_rim = block_p[2] <= (self.bowl_start_z + self.bowl_success_height)
        above_table = block_p[2] >= (self.table_top_z - 0.02)
        return bool(xy_close and below_rim and above_table)

    def _is_start_left_drop_x(self, x: float) -> bool:
        return bool(float(x) <= self._belt_start_zone_max_x)

    def _latch_belt_contact(self):
        if getattr(self, "_belt_contact_latched", False) or getattr(self, "block", None) is None:
            return
        if not self._on_belt():
            return
        p = np.array(self.block.get_pose().p, dtype=np.float64)
        self._belt_contact_x = float(p[0])
        self._belt_contact_y = float(p[1])
        self.dropped_at_start_left = self._is_start_left_drop_x(self._belt_contact_x)
        self._belt_contact_latched = True

    def _advance_bowl_motion(self):
        if not getattr(self, "bowl_move_enabled", False) or getattr(self, "bowl", None) is None:
            return
        pose = self.bowl.get_pose()
        next_y = float(pose.p[1] + self._bowl_move_dir * abs(self.bowl_move_speed) * self.scene.get_timestep())
        low = float(self.belt_y - self.belt_half_w)
        high = float(self.belt_y + self.belt_half_w)
        if high <= low:
            return
        if next_y > high:
            next_y = high
            self._bowl_move_dir = -1.0
        elif next_y < low:
            next_y = low
            self._bowl_move_dir = 1.0
        self.bowl.actor.set_pose(sapien.Pose([self._bowl_base_x, next_y, pose.p[2]], pose.q))

    def _predict_bowl_y(self, future_time: float) -> float:
        if getattr(self, "bowl", None) is None:
            return float(self.belt_y)
        pose = self.bowl.get_pose()
        y_now = float(pose.p[1])
        if (not getattr(self, "bowl_move_enabled", False)) or future_time <= 0.0:
            return y_now

        speed = abs(float(self.bowl_move_speed))
        if speed <= 1e-8:
            return y_now

        low = float(self.belt_y - self.belt_half_w)
        high = float(self.belt_y + self.belt_half_w)
        if high <= low:
            return float(np.clip(y_now, low, high))

        y = float(np.clip(y_now, low, high))
        direction = 1.0 if self._bowl_move_dir >= 0 else -1.0
        remaining = float(future_time)

        # Piecewise-linear reflection between the belt-lane endpoints.
        for _ in range(32):
            if remaining <= 0.0:
                break
            boundary = high if direction > 0 else low
            dist = abs(boundary - y)
            t_to_boundary = dist / speed if dist > 1e-9 else 0.0
            if remaining <= t_to_boundary:
                y += direction * speed * remaining
                return float(np.clip(y, low, high))
            y = boundary
            remaining -= t_to_boundary
            direction *= -1.0

        return float(np.clip(y, low, high))

    def _estimate_drop_lane_y(self, release_x: float, release_time: float = 0.0) -> float:
        if getattr(self, "bowl", None) is None:
            return float(self.belt_y)
        if not getattr(self, "bowl_move_enabled", False):
            return float(self.bowl.get_pose().p[1])
        belt_travel_time = max(float(self._belt_drop_x - release_x), 0.0) / max(self.belt_speed, 1e-8)
        return self._predict_bowl_y(release_time + belt_travel_time)

    def _move_block_to_belt_clearance(self, arm_tag: ArmTag, clearance: float):
        block_bottom_z = float(self.block.get_pose().p[2]) - self.block_half_h
        dz = (self.belt_surface_z + float(clearance)) - block_bottom_z
        if abs(dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz))

    def _dwell(self, n_steps: int):
        for i in range(max(0, int(n_steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _steps_until_drop_alignment(
        self,
        release_x: float,
        release_y: float,
        lead_time: float = 0.0,
        max_steps: int = 1500,
        tol: float | None = None,
    ) -> int:
        if not getattr(self, "bowl_move_enabled", False):
            return 0

        dt = float(self.scene.get_timestep())
        tol = float(tol if tol is not None else min(self.bowl_inner_radius * 0.35, 0.02))
        best_steps = 0
        best_err = float("inf")
        for step in range(max(0, int(max_steps)) + 1):
            future_t = lead_time + step * dt
            err = abs(self._estimate_drop_lane_y(release_x=release_x, release_time=future_t) - float(release_y))
            if err < best_err:
                best_err = err
                best_steps = step
            if err <= tol:
                return step
        return best_steps

    # ----------------------------------------------------------------- belt visuals
    def _init_belt_visual_motion(self):
        # Render-only tread bands travel with the top surface so the conveyor visibly moves.
        self._belt_bands = []
        self._belt_band_phase = 0.0

        band_half_x = float(self.BELT_BAND_HALF_X_DEFAULT)
        self._belt_band_half_x = band_half_x
        band_half_y = 0.0
        band_half_z = float(self.BELT_BAND_HALF_Z_DEFAULT)
        edge_margin = 0.002
        band_half_y = max(self._belt_half_w - edge_margin, 0.001)

        self._belt_band_min_x = self._belt_cx - self._belt_half_len_x + band_half_x + edge_margin
        self._belt_band_max_x = self._belt_cx + self._belt_half_len_x - band_half_x - edge_margin
        self._belt_band_start_x = self._belt_band_min_x
        self._belt_band_span = max(self._belt_band_max_x - self._belt_band_min_x, 1e-6)
        self._belt_band_z = self.belt_surface_z + self.BELT_BAND_Z_OFFSET_DEFAULT + band_half_z

        spacing = max(float(self.BELT_BAND_SPACING_DEFAULT), band_half_x * 2.5)
        band_count = max(4, int(np.ceil(max(self._belt_band_span, spacing) / spacing)))
        self._belt_band_offsets = np.linspace(0.0, self._belt_band_span, band_count, endpoint=False)

        band_colors = (
            (0.22, 0.22, 0.24),
            (0.32, 0.32, 0.35),
        )
        for i, offset in enumerate(self._belt_band_offsets):
            x = self._belt_band_start_x + float(offset)
            band = create_visual_box(
                scene=self,
                pose=sapien.Pose([x, self.belt_y, self._belt_band_z], [1, 0, 0, 0]),
                half_size=(band_half_x, band_half_y, band_half_z),
                color=band_colors[i % len(band_colors)],
                name=f"belt_band_{i}",
            )
            self._belt_bands.append(band)

    def _advance_belt_visual_motion(self):
        if not getattr(self, "_belt_bands", None) or self._belt_band_span <= 0.0:
            return

        self._belt_band_phase = (self._belt_band_phase + self.belt_speed * self.scene.get_timestep()) % self._belt_band_span
        for band, offset in zip(self._belt_bands, self._belt_band_offsets):
            x = self._belt_band_start_x + float((offset + self._belt_band_phase) % self._belt_band_span)
            x = float(np.clip(x, self._belt_band_min_x, self._belt_band_max_x))
            band.set_pose(sapien.Pose([x, self.belt_y, self._belt_band_z], [1, 0, 0, 0]))

    # ----------------------------------------------------------------- per-step hook (belt drive)
    def _update_kinematic_tasks(self):
        # base hook first (drives DOMINO dynamic-object motion, if any)
        super()._update_kinematic_tasks()
        self._advance_belt_visual_motion()
        self._advance_bowl_motion()

        if getattr(self, "_released", False) and getattr(self, "block", None) is not None:
            t = self._current_tilt_deg()
            self.max_tilt_deg = max(self.max_tilt_deg, t)
            if self._on_belt():
                self.placed_on_belt = True
                self._latch_belt_contact()
            self.in_bowl = self._block_in_bowl()

        # only drive the block while it is still being carried by the belt
        if getattr(self, "_belt_active", False) and getattr(self, "block", None) is not None:
            if self._block_dyn is not None and getattr(self, "_released", False) and self._on_belt():
                # The belt grabs the block. On the first captured step, hand the block over to the
                # belt's kinematic drive: snap it onto the belt surface in its (released, upright)
                # orientation so the moving belt carries it cleanly. A grossly mismatched release
                # is what produces a large pre-capture tilt (recorded above into max_tilt_deg).
                if not self._block_kinematic:
                    try:
                        self._block_dyn.set_linear_velocity(np.zeros(3))
                        self._block_dyn.set_angular_velocity(np.zeros(3))
                        self._block_dyn.set_kinematic(True)
                    except Exception:
                        pass
                    self._block_kinematic = True
                    self._belt_q = self._release_q  # ride upright, as released
                dt = self.scene.get_timestep()
                pose = self.block.get_pose()
                new_p = np.array(pose.p, dtype=np.float64)
                new_p[0] += self.belt_speed * dt
                # keep the block seated on the belt surface
                new_p[2] = self.belt_surface_z + self.block_half_h
                self.block.actor.set_pose(sapien.Pose(p=new_p.tolist(), q=self._belt_q))
                if new_p[0] >= self._belt_drop_x:
                    self.reached_end = True
                    self._belt_active = False
                    self._block_dropped = True
                    self._drop_settle_steps = 0
                    if self._block_dyn is not None:
                        try:
                            self._block_dyn.set_kinematic(False)
                            self._block_dyn.set_linear_velocity(np.array([self.belt_speed, 0.0, 0.0]))
                            self._block_dyn.set_angular_velocity(np.zeros(3))
                        except Exception:
                            pass
                    self._block_kinematic = False
                elif new_p[0] >= self.belt_x_end:
                    self.reached_end = True

    def _ride_belt(self):
        # Dwell while the belt carries the block to the end, then keep recording through the
        # configured post-drop settle window so the final fall into the bowl appears in saved video.
        for i in range(self.belt_ride_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            self._ride_steps_done += 1
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self._block_dropped:
                self._drop_settle_steps += 1
                if self._drop_settle_steps >= self.bowl_settle_steps:
                    for j in range(20):
                        self._update_kinematic_tasks()
                        self.scene.step()
                        if self.save_freq and (j % self.save_freq == 0):
                            self._take_picture()
                    self.in_bowl = self._block_in_bowl()
                    if self.save_freq:
                        self._take_picture()
                    break

    # ----------------------------------------------------------------- policy
    def play_once(self):
        arm_tag = ArmTag("right" if self.block.get_pose().p[0] > 0 else "left")
        moving_bowl = bool(getattr(self, "bowl_move_enabled", False))

        # 1) grasp the tall block in the near zone. Grasp near the UPPER body so the gripper
        #    stays well above the belt surface during the lower / release steps (a low grasp
        #    puts the fingers right at the belt and the open/lift plan collides with the slab).
        self.move(self.grasp_actor(self.block, arm_tag=arm_tag, pre_grasp_dis=0.1))
        #,            contact_point_id=[0, 1, 2, 3]))
        # 2) lift well clear of the table
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="arm"))

        # Release at the LEFT/start of the belt; for the moving bowl case we still time the
        # release so the bowl will line up with the lane by the time the block reaches the exit.
        match_dist = max(0.02, self.belt_speed * 0.5)
        hover_x = self.belt_x_start
        release_x = hover_x + match_dist
        if moving_bowl:
            target_lane_y = float(self.belt_y)
        else:
            target_lane_y = float(self.bowl.get_pose().p[1]) if getattr(self, "bowl", None) is not None else self.belt_y

        # carry it over above the chosen belt lane.
        dx = hover_x - float(self.block.get_pose().p[0])
        dy = target_lane_y - float(self.block.get_pose().p[1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))
        hover_clearance = 0.015 if moving_bowl else 0.005
        self._move_block_to_belt_clearance(arm_tag=arm_tag, clearance=hover_clearance)

        # 3) KEY MECHANIC: match the gripper's horizontal velocity to the belt before releasing.
        #    A short, smooth +x move at (approximately) belt speed gives the block forward momentum
        #    so it is not sheared/tipped by a sudden velocity mismatch when the belt grabs it.
        if moving_bowl:
            release_time = (
                match_dist / max(self.belt_speed, 1e-8)
                + self.belt_release_delay_steps * self.scene.get_timestep()
            )
            wait_steps = self._steps_until_drop_alignment(
                release_x=release_x,
                release_y=float(self.block.get_pose().p[1]),
                lead_time=release_time,
            )
            self._dwell(wait_steps)
            target_lane_y = self._estimate_drop_lane_y(release_x=release_x, release_time=release_time)
            dy = target_lane_y - float(self.block.get_pose().p[1])
            if abs(dy) > 1e-3:
                self.move(self.move_by_displacement(arm_tag=arm_tag, y=dy))
            self._move_block_to_belt_clearance(arm_tag=arm_tag, clearance=0.005)
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=match_dist))

        # 4) release. Open the gripper and let a few dynamic steps pass so the (top-heavy) block's
        #    response to the release is what the tilt metric captures, THEN hand it to the belt's
        #    kinematic drive. Tracking begins here (after the match stroke) so the metric reflects
        #    only the release + ride, not the carry.
        self._release_q = [1.0, 0.0, 0.0, 0.0]
        self._released = True
        self.move(self.open_gripper(arm_tag))
        # Keep the belt paused briefly so the block clears the fingers and settles onto the belt
        # before the kinematic conveyor hand-off begins.
        self._dwell(self.belt_release_delay_steps)
        self._belt_active = True
        # ride-tilt metric measures stability AFTER the hand-off to the belt
        # self.max_tilt_deg = 0.0
        # retract the arm clear of the (now belt-borne) block
        # self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        # self.move(self.back_to_origin(arm_tag))

        # let the belt carry the block to the end
        self._ride_belt()

        # finalize the metric
        self.tilt_score = float(np.clip(1.0 - self.max_tilt_deg / self.theta_max_deg, 0.0, 1.0))

        self.info["info"] = {
            "{A}": "tall_block",
            "{B}": f"002_bowl/base{self.bowl_id}",
            "{C}": "conveyor_belt",
            "{a}": str(arm_tag),
        }
        return self.info

    # ----------------------------------------------------------------- success + metric
    def check_success(self):
        # success = the block was dropped onto the left/start of the belt and ends inside the bowl
        self._latch_belt_contact()
        self.placed_on_belt = bool(self.placed_on_belt or self._on_belt())
        self.in_bowl = bool(self._block_in_bowl())
        success = bool(self.dropped_at_start_left and self.placed_on_belt and self.in_bowl)
        self.info["placed_on_belt"] = self.placed_on_belt
        self.info["dropped_at_start_left"] = self.dropped_at_start_left
        self.info["in_bowl"] = success
        self.info["tilt_score"] = float(self.tilt_score)
        return success

    # ----------------------------------------------------------------- record state per-frame
    def get_obs(self):
        obs = super().get_obs()
        cur_tilt = self._current_tilt_deg() if hasattr(self, "block") else 0.0
        obs["belt"] = {
            "tilt_deg": float(cur_tilt),
            "max_tilt_deg": float(getattr(self, "max_tilt_deg", 0.0)),
            "theta_max_deg": float(getattr(self, "theta_max_deg", self.THETA_MAX_DEG_DEFAULT)),
            "tilt_score": float(np.clip(1.0 - getattr(self, "max_tilt_deg", 0.0) /
                                        getattr(self, "theta_max_deg", self.THETA_MAX_DEG_DEFAULT), 0.0, 1.0)),
            "belt_speed": float(getattr(self, "belt_speed", self.BELT_SPEED_DEFAULT)),
            "reached_end": bool(getattr(self, "reached_end", False)),
            "placed_on_belt": bool(getattr(self, "placed_on_belt", False)),
            "dropped_at_start_left": bool(getattr(self, "dropped_at_start_left", False)),
            "in_bowl": bool(getattr(self, "in_bowl", False)),
            "belt_contact_x": float(getattr(self, "_belt_contact_x", 0.0) or 0.0),
            "bowl_center": self.bowl.get_pose().p.tolist() if getattr(self, "bowl", None) is not None else [0.0, 0.0, 0.0],
        }
        return obs
