from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np


class pick_cup_behind_fan(Base_Task):
    """Retrieve a water-filled cup from behind a spinning 3-blade fan without hitting the
    blades, and bring it back to the near zone without spilling.

    The fan is reused asset 099_fan, spun about its vertical (table-normal) axis every physics
    step by an overridden _update_kinematic_tasks(). The spin angle is driven purely by a step
    counter so the plan pass and the render pass stay in lock-step (two-pass determinism).

    A 3-blade fan has a 120 deg rotational period; for a fraction GAP_FRAC of each period a
    blade-gap is aligned with the reach corridor and the gripper may pass safely. The reach is
    timed to land inside a gap window (the policy dwells, advancing the fan, until the predicted
    phase at reach time is safe). Passing through while a blade occludes the corridor is a blade
    collision (hard fail). On the way out the cup must stay upright; tilt beyond THETA_SPILL is a
    spill (hard fail). stability_score = clamp(1 - max_tilt/THETA_SPILL, 0, 1), zeroed on any
    blade collision or spill.
    """

    # ----- class-default params (overridable via task_args.pick_cup_behind_fan in the yml) -----
    FAN_OMEGA_DEG_DEFAULT = 2.0        # fan angular speed, degrees per physics step
    GAP_FRAC_DEFAULT = 0.40            # fraction of each 120-deg period the corridor is a safe gap
    THETA_SPILL_DEFAULT = 35.0         # cup tilt (deg) from vertical beyond which it spills
    CUP_Y_FAR_DEFAULT = 0.12           # cup spawns this far out (behind the fan)
    DWELL_MAX_STEPS = 1500             # cap on how long we wait for a safe approach window
    REACH_STEPS_EST = 120             # rough steps the reach-through motion consumes (for phasing)

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags is not retained on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("pick_cup_behind_fan", {})
        # reset transient state so a reused instance can't fire the fan-spin during load_camera
        self._spinning = False
        self._tracking_tilt = False
        self._fan_base_pose = None
        super()._init_task_env_(**kwags)

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        self.fan_omega = float(self._cfg.get("fan_omega_deg", self.FAN_OMEGA_DEG_DEFAULT))
        self.gap_frac = float(self._cfg.get("gap_frac", self.GAP_FRAC_DEFAULT))
        self.theta_spill = float(self._cfg.get("theta_spill", self.THETA_SPILL_DEFAULT))

        # randomize fan speed and cup distance a little per episode. The cup sits at "far-center"
        # but must stay inside the arm's reach (a right-arm grasp fails past y~+0.12), so the far
        # band is capped accordingly.
        self.fan_omega *= float(np.random.uniform(0.8, 1.3))
        cup_y = float(np.random.uniform(0.04, 0.09))
        self._cup_spawn_y = cup_y

        # fan-spin bookkeeping (step-driven so the two collector passes match exactly)
        self._fan_step = 0
        self._fan_angle0 = float(np.random.uniform(0.0, 2 * np.pi))  # random start phase
        self._spinning = False
        self._water_level = float(np.random.uniform(0.4, 0.9))       # randomization: water level
        self.max_tilt = 0.0
        self.blade_collision = False
        self.spilled = False
        self._reach_phase_deg = None  # phase recorded at the moment of the reach-through

        # Single-arm task -> everything on the RIGHT half (right arm only reaches x>0).
        side = 1.0
        self._side = side

        # Cup: far-center on the right side, behind the fan. 021_cup has grasp contact points
        # and a bottom functional point for placing.
        cup_x = float(np.random.uniform(0.10, 0.18))
        cup_pose = rand_pose(
            xlim=[cup_x, cup_x], ylim=[cup_y, cup_y],
            zlim=[0.74 + self.table_z_bias], qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=False,
        )
        self.cup = create_actor(
            self, pose=cup_pose, modelname="021_cup", model_id=0, convex=True, is_static=False,
        )
        self.cup.set_mass(0.1)
        # moderate damping steadies the cup while carried without the violent fling that a too-light
        # body produced.
        for c in self.cup.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(3.0)
                    c.set_angular_damping(8.0)
                except Exception:
                    pass

        # Fan: center-mid, between the cup (far) and the near zone, on the same (right) side so it
        # actually sits in the reach corridor. Stand it up with the place_fan qpos.
        self.fan_id = int(np.random.choice([4, 5]))
        fan_x = cup_x  # in line with the cup so it shields the approach
        fan_pose = rand_pose(
            xlim=[fan_x, fan_x], ylim=[cup_y - 0.13, cup_y - 0.13],
            qpos=[0.0, 0.0, 0.707, 0.707], rotate_rand=False,
        )
        self.fan = create_actor(
            self, pose=fan_pose, modelname="099_fan", model_id=self.fan_id,
            convex=True, is_static=False,
        )
        # Make the fan KINEMATIC so we can drive its spin every step with set_pose (a static body
        # cannot be repositioned; a free dynamic body would topple). Collision with the blades is
        # judged from the step-driven phase, not physics, so a kinematic fan is exactly right.
        for c in self.fan.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                c.set_kinematic(True)
        self._fan_base_pose = self.fan.get_pose()

        # tint the "water" in the cup blue so the fill is visible
        self._set_cup_water_color([0.10, 0.35, 0.95, 1.0])

        self.add_prohibit_area(self.cup, padding=0.03)
        self.add_prohibit_area(self.fan, padding=0.05)

    # -------------------------------------------------------------- helpers
    def _set_cup_water_color(self, col):
        for c in self.cup.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color(col)
                    except Exception:
                        pass

    def _fan_phase_deg(self, step=None):
        """Step-driven fan rotation angle, wrapped into a 0..120 blade period (3 blades)."""
        if step is None:
            step = self._fan_step
        ang = self._fan_angle0 + np.deg2rad(self.fan_omega) * step
        deg = np.rad2deg(ang) % 120.0
        return deg

    def _gap_aligned(self, phase_deg):
        """True if the blade gap is aligned with the reach corridor at this phase.
        Gap occupies the first GAP_FRAC of each 120-deg period (centered logic kept simple)."""
        return phase_deg < (120.0 * self.gap_frac)

    @staticmethod
    def _quat_mul(a, b):
        """Hamilton product of two (w,x,y,z) quaternions."""
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])

    def _apply_fan_pose(self):
        """Rotate the fan actor about the table-normal (world z) axis to the current step angle."""
        base = getattr(self, "_fan_base_pose", None)
        if base is None:
            return
        ang = self._fan_angle0 + np.deg2rad(self.fan_omega) * self._fan_step
        half = ang / 2.0
        spin_q = np.array([np.cos(half), 0.0, 0.0, np.sin(half)])  # (w,x,y,z) about z
        base_q = np.array([base.q[0], base.q[1], base.q[2], base.q[3]])
        new_q = self._quat_mul(spin_q, base_q)
        self.fan.actor.set_pose(sapien.Pose(p=base.p, q=new_q.tolist()))

    def _cup_tilt_deg(self):
        """Angle (deg) between the cup's local up-axis and world up."""
        q = self.cup.get_pose().q  # (w,x,y,z)
        w, x, y, z = q
        # The cup is spawned with qpos [0.707,0.707,0,0] so its world-up maps to local +y.
        # Rotate local +y by q and measure deviation from world +z.
        # Rotation matrix column for the axis that is initially vertical:
        up_local = np.array([0.0, 1.0, 0.0])
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])
        v = R @ up_local
        v = v / (np.linalg.norm(v) + 1e-9)
        cosang = float(np.clip(abs(v[2]), 0.0, 1.0))  # |z-component| -> tilt from vertical
        return float(np.rad2deg(np.arccos(cosang)))

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic-object motion; runs every physics step
        super()._update_kinematic_tasks()
        if getattr(self, "_spinning", False):
            self._fan_step += 1
            self._apply_fan_pose()
        # continuously track the worst cup tilt once we are holding it
        if getattr(self, "_tracking_tilt", False):
            t = self._cup_tilt_deg()
            self.max_tilt = max(self.max_tilt, t)
            if t > self.theta_spill:
                self.spilled = True

    def _dwell(self, n_steps):
        """Advance physics (and the spinning fan) for n_steps, recording frames periodically."""
        for i in range(int(n_steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _wait_for_gap(self):
        """Dwell, spinning the fan, until the predicted phase at reach-completion is inside a gap
        window. Returns the phase (deg) recorded at the reach moment."""
        for _ in range(self.DWELL_MAX_STEPS):
            future = self._fan_phase_deg(self._fan_step + self.REACH_STEPS_EST)
            if self._gap_aligned(future):
                break
            self._dwell(5)
        return self._fan_phase_deg(self._fan_step + self.REACH_STEPS_EST)

    def _dbg(self, label):
        import os
        if os.environ.get("PCBF_DEBUG"):
            print(f"  [pcbf] after {label}: plan_success={self.plan_success} "
                  f"cup_p={np.round(self.cup.get_pose().p, 3)} tilt={self._cup_tilt_deg():.1f}", flush=True)

    # ------------------------------------------------------------- policy
    def play_once(self):
        arm_tag = ArmTag("right" if self.cup.get_pose().p[0] > 0 else "left")

        # 1) the fan is already spinning; observe its phase and dwell until a gap aligns with the
        #    corridor before reaching through.
        self._spinning = True
        self._dwell(20)                       # let the fan spin a moment (arm "observes")
        self._reach_phase_deg = self._wait_for_gap()

        # record whether the corridor was actually clear at the reach moment (blade collision)
        if not self._gap_aligned(self._reach_phase_deg):
            self.blade_collision = True

        # 2) reach through the gap and grasp the cup. Guard against an unreachable grasp pose
        #    (choose_grasp_pose can return None at some cup orientations, which would make
        #    grasp_actor build a move-action with target_pose=None) -> treat as plan failure.
        pre_g, g = self.choose_grasp_pose(self.cup, arm_tag=arm_tag, pre_dis=0.1, target_dis=0.0)
        if pre_g is None or g is None:
            self.plan_success = False
            self.info["info"] = {"{A}": "021_cup/base0",
                                 "{B}": f"099_fan/base{self.fan_id}", "{a}": str(arm_tag)}
            return self.info
        # grip firmly (gripper_pos=0 fully closes on the thin cup wall) so the cup stays held
        # through the lift/retract instead of being knocked loose.
        self.move(self.grasp_actor(self.cup, arm_tag=arm_tag, pre_grasp_dis=0.1, gripper_pos=0.0))
        self._dbg("grasp")
        # let the grasp settle, then START tracking tilt for the CARRY phase only (the brief
        # gripper-close transient is not a spill -- spill is about tilting while carrying it back).
        self._dwell(10)
        self.max_tilt = 0.0
        self.spilled = False
        self._tracking_tilt = True
        # lift along the gripper (arm) axis -- the reliable lift pattern used across the codebase.
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08, move_axis="arm"))
        self._dwell(6)                        # let it stabilize before translating
        self._dbg("lift")

        # 3) retract toward the near zone KEEPING THE GRASP ORIENTATION FIXED. Translate the
        #    *current* EE pose position only (same quaternion) via absolute move_to_pose; a relative
        #    move_by_displacement let the wrist re-orient and flung the cup. Keep the pull modest
        #    (the near zone only needs ~12 cm inboard) and dwell so the held cup stays upright.
        ee = list(self.get_arm_pose(arm_tag))   # [x,y,z, qw,qx,qy,qz]
        quat = ee[3:]
        for dy in (-0.06, -0.06):
            ee[1] += dy
            self.move(self.move_to_pose(arm_tag, ee[:3] + quat))
            self._dwell(6)
        self._dbg("retract")

        # 4) set it down in the near-center (lower position only, same orientation) and release
        ee[2] -= 0.06
        self.move(self.move_to_pose(arm_tag, ee[:3] + quat))
        self._dbg("lower")
        self.move(self.open_gripper(arm_tag))
        self._dwell(15)                       # let it settle (and keep tracking tilt)
        self._tracking_tilt = False
        self._spinning = False

        self.info["info"] = {
            "{A}": "021_cup/base0",
            "{B}": f"099_fan/base{self.fan_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    # ------------------------------------------------------------- success
    def stability_score(self):
        if self.blade_collision or self.spilled:
            return 0.0
        return float(np.clip(1.0 - self.max_tilt / max(1e-6, self.theta_spill), 0.0, 1.0))

    def check_success(self):
        # no blade hit, no spill
        if self.blade_collision or self.spilled:
            return False
        cup_p = self.cup.get_pose().p
        cup_z = float(cup_p[2])
        # cup resting on the table (not dropped, not still aloft)
        on_table = (0.70 + self.table_z_bias) < cup_z < (0.84 + self.table_z_bias)
        # brought back to the NEAR zone: y pulled at least 10 cm in from the actual far spawn y
        # toward the robot (a fixed threshold could overlap the spawn band and pass trivially).
        in_near_zone = float(cup_p[1]) < (self._cup_spawn_y - 0.10)
        # still on the correct (right) half
        same_side = (cup_p[0] > 0.0) if self._side > 0 else (cup_p[0] < 0.0)
        # upright enough (tilt never exceeded spill threshold -> also implied by not self.spilled)
        upright = self.max_tilt <= self.theta_spill
        return bool(on_table and in_near_zone and same_side and upright)

    # record fan phase / cup tilt into the trajectory (per frame)
    def get_obs(self):
        obs = super().get_obs()
        obs["pick_cup_behind_fan"] = {
            "fan_phase_deg": float(self._fan_phase_deg()),
            "fan_omega_deg": float(self.fan_omega),
            "cup_tilt_deg": float(self._cup_tilt_deg()),
            "max_tilt_deg": float(self.max_tilt),
            "theta_spill_deg": float(self.theta_spill),
            "water_level": float(self._water_level),
            "blade_collision": bool(self.blade_collision),
            "spilled": bool(self.spilled),
            "stability_score": float(self.stability_score()),
        }
        return obs
