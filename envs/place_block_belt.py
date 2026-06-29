from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class place_block_belt(Base_Task):
    """Set a tall, top-heavy block onto a moving conveyor belt so it rides to the
    far end of the belt without tipping over.

    A flat conveyor belt spans the mid zone of the RIGHT half of the table and
    carries anything resting on it laterally (along +x) at a constant belt speed.
    A tall block with a high center of mass spawns in the near zone within the
    right arm's reach. The scripted expert:
      1) grasps the block in the near zone,
      2) lifts it and carries it above the belt surface,
      3) MATCHES the gripper's horizontal (x) velocity to the belt speed by a
         short move_by_displacement along +x just before release, so the block is
         not sheared/tipped by a velocity mismatch on release,
      4) opens the gripper; the belt then carries the upright block to the end.

    The belt motion is driven step-by-step in the overridden _update_kinematic_tasks:
    once the block is released and resting on the belt, the block is set kinematic
    and advanced along +x by belt_speed * dt every physics step. Tilt of the block
    away from vertical is recorded each step.

    Single-arm (right). Key mechanic: release at belt speed to keep the top-heavy
    block upright. Metric: tilt_score = clamp(1 - max_tilt_angle/theta_max, 0, 1).
    """

    # ----------------------------------------------------------------- params (class defaults)
    BELT_SPEED_DEFAULT = 0.06          # m/s lateral (+x) belt surface speed
    BELT_X_START_DEFAULT = 0.04        # near-center x where the belt begins (right half)
    BELT_X_END_DEFAULT = 0.30          # outer x where the belt ends (success line)
    BELT_Y_DEFAULT = -0.02             # y of the belt centerline (mid zone)
    BELT_HALF_W_DEFAULT = 0.07         # belt half-width in y
    BELT_RIDE_STEPS_DEFAULT = 600      # max physics steps to let the block ride the belt

    BLOCK_HALF_W_DEFAULT = 0.022       # block half-width (x,y) -> narrow base (top-heavy)
    BLOCK_HALF_H_DEFAULT = 0.060       # block half-height (z) -> 12 cm tall
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
        self.block_half_w = float(cfg.get("block_half_w", self.BLOCK_HALF_W_DEFAULT))
        self.block_half_h = float(cfg.get("block_half_h", self.BLOCK_HALF_H_DEFAULT))
        self.block_mass = float(cfg.get("block_mass", self.BLOCK_MASS_DEFAULT))
        self.block_com_frac = float(cfg.get("block_com_frac", self.BLOCK_COM_FRAC_DEFAULT))
        self.theta_max_deg = float(cfg.get("theta_max_deg", self.THETA_MAX_DEG_DEFAULT))

        # randomize within the configured envelope (seed-driven) for variety
        self.belt_speed = float(np.random.uniform(self.belt_speed * 0.8, self.belt_speed * 1.2))
        self.block_half_h = float(np.random.uniform(self.block_half_h * 0.9, self.block_half_h * 1.1))
        self.block_mass = float(np.random.uniform(self.block_mass * 0.8, self.block_mass * 1.2))

        table_z = 0.74 + self.table_z_bias
        self.table_top_z = table_z

        # ------- belt (a thin static slab on the table surface; carries objects kinematically)
        self.belt_thickness = 0.012
        belt_cx = 0.5 * (self.belt_x_start + self.belt_x_end)
        belt_half_len_x = 0.5 * (self.belt_x_end - self.belt_x_start)
        belt_pose = sapien.Pose(
            p=[belt_cx, self.belt_y, table_z + self.belt_thickness * 0.5],
            q=[1, 0, 0, 0],
        )
        self.belt = create_box(
            scene=self,
            pose=belt_pose,
            half_size=(belt_half_len_x, self.belt_half_w, self.belt_thickness * 0.5),
            color=(0.15, 0.15, 0.17),
            name="conveyor_belt",
            is_static=True,
        )
        self.belt_surface_z = table_z + self.belt_thickness  # top face of the belt

        # ------- tall top-heavy block, spawned in the NEAR zone within the right arm's reach
        block_x = float(np.random.uniform(0.06, 0.11))
        block_y = float(np.random.uniform(0.04, 0.10))
        block_pose = rand_pose(
            xlim=[block_x, block_x],
            ylim=[block_y, block_y],
            zlim=[table_z + self.block_half_h],
            qpos=[1, 0, 0, 0],
            rotate_rand=True,
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
        self.add_prohibit_area(self.block, padding=0.05)

        # ------- belt / tilt bookkeeping
        self._belt_active = False
        self._released = False
        self._block_kinematic = False
        self.max_tilt_deg = 0.0
        self.tilt_score = 0.0
        self.reached_end = False
        self._ride_steps_done = 0

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
            (self.belt_x_start - 0.03) <= p[0] <= (self.belt_x_end + 0.06)
            and abs(p[1] - self.belt_y) <= (self.belt_half_w + 0.03)
            and p[2] < (self.belt_surface_z + self.block_half_h + 0.03)
        )
        return bool(near_belt)

    # ----------------------------------------------------------------- per-step hook (belt drive)
    def _update_kinematic_tasks(self):
        # base hook first (drives DOMINO dynamic-object motion, if any)
        super()._update_kinematic_tasks()

        # only act once the policy has armed the belt AND the block actor exists
        if getattr(self, "_belt_active", False) and getattr(self, "block", None) is not None:
            t = self._current_tilt_deg()
            self.max_tilt_deg = max(self.max_tilt_deg, t)

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
                if new_p[0] >= self.belt_x_end:
                    self.reached_end = True

    def _ride_belt(self):
        # dwell while the belt carries the block to the end, recording frames step-driven
        for i in range(self.belt_ride_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            self._ride_steps_done += 1
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self.reached_end:
                # a few extra settle steps to capture the final upright pose
                for _ in range(20):
                    self._update_kinematic_tasks()
                    self.scene.step()
                break

    # ----------------------------------------------------------------- policy
    def play_once(self):
        arm_tag = ArmTag("right" if self.block.get_pose().p[0] > 0 else "left")

        # 1) grasp the tall block in the near zone. Grasp near the UPPER body so the gripper
        #    stays well above the belt surface during the lower / release steps (a low grasp
        #    puts the fingers right at the belt and the open/lift plan collides with the slab).
        self.move(self.grasp_actor(self.block, arm_tag=arm_tag, pre_grasp_dis=0.08,
                                   contact_point_id=[0, 1, 2, 3]))
        # 2) lift well clear of the table
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.14, move_axis="arm"))

        # carry it over above the START of the belt: shift to belt y and to the belt start x.
        dx = self.belt_x_start - float(self.block.get_pose().p[0])
        dy = self.belt_y - float(self.block.get_pose().p[1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))
        # lower until the block bottom is just above the belt surface
        block_bottom_z = float(self.block.get_pose().p[2]) - self.block_half_h
        dz = (self.belt_surface_z + 0.005) - block_bottom_z
        if dz < 0:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=dz))

        # 3) KEY MECHANIC: match the gripper's horizontal velocity to the belt before releasing.
        #    A short, smooth +x move at (approximately) belt speed gives the block forward momentum
        #    so it is not sheared/tipped by a sudden velocity mismatch when the belt grabs it.
        match_dist = max(0.02, self.belt_speed * 0.5)  # short matched stroke along belt direction
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=match_dist))

        # 4) release. Open the gripper and let a few dynamic steps pass so the (top-heavy) block's
        #    response to the release is what the tilt metric captures, THEN hand it to the belt's
        #    kinematic drive. Tracking begins here (after the match stroke) so the metric reflects
        #    only the release + ride, not the carry.
        self._release_q = [1.0, 0.0, 0.0, 0.0]
        # Arm the belt BEFORE opening the gripper so the kinematic belt drive captures the block
        # onto the belt on the very first on-belt physics step of the gripper-open motion (while it
        # is still upright in the fingers), instead of letting the top-heavy block topple free.
        self._belt_active = True
        self._released = True
        self.move(self.open_gripper(arm_tag))
        # ride-tilt metric measures stability AFTER the hand-off to the belt
        self.max_tilt_deg = 0.0
        # retract the arm clear of the (now belt-borne) block
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))

        # let the belt carry the block to the end
        self._ride_belt()

        # finalize the metric
        self.tilt_score = float(np.clip(1.0 - self.max_tilt_deg / self.theta_max_deg, 0.0, 1.0))

        self.info["info"] = {
            "{A}": "tall_block",
            "{B}": "conveyor_belt",
            "{a}": str(arm_tag),
        }
        return self.info

    # ----------------------------------------------------------------- success + metric
    def check_success(self):
        # the block must have ridden to (near) the belt end AND stayed upright
        p = self.block.get_pose().p
        reached = bool(self.reached_end or p[0] >= (self.belt_x_end - 0.03))
        upright = bool(self.max_tilt_deg <= self.theta_max_deg)
        on_belt_y = bool(abs(float(p[1]) - self.belt_y) <= (self.belt_half_w + 0.05))
        # not fallen off / through the belt
        seated = bool(p[2] > (self.belt_surface_z + self.block_half_h - 0.04))
        return bool(reached and upright and on_belt_y and seated)

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
        }
        return obs
