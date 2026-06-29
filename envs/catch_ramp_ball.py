from ._base_task import Base_Task
from .utils import *
import sapien
import numpy as np
import transforms3d as t3d
import os


class catch_ramp_ball(Base_Task):
    """A ball rolls down an inclined ramp (far edge high, near edge low+elevated to ~cup height),
    leaves the FRONT (near, -y) edge and falls; the robot pre-positions a cup at the predicted landing
    to catch it. Hybrid dynamics: the roll is scripted/kinematic (deterministic, exact edge state),
    then the ball is RELEASED to a real-physics projectile that drops into the cup. The landing is
    computed analytically from the edge state (gravity parabola), so the expert places the cup exactly.
    Frame: -y = NEAR (robot), +y = FAR. Only the cup is manipulated; ramp+ball are scenery."""

    SIM_HZ = 250.0                     # sim runs at 250 Hz (scene timestep 1/250 s) -> steps = sec * 250

    RAMP_ANGLE_DEFAULT = 0.24          # rad (~14 deg) -- flatter: more forward(-y) carry, less drop
    RAMP_CENTER_Y_DEFAULT = 0.12       # ramp center depth (FAR = +y)
    RAMP_HALF = (0.20, 0.11, 0.008)    # half-size x(width) y(length) z(thickness) -- wide board so the
                                       # +-0.15 x spread stays clear of the rails (ball radius ~0.018)
    RAIL_H = 0.025
    LOW_EDGE_Z = 0.10                  # near/low edge ~10cm up (lowered) -> support legs auto-scale to ~half
    BALL_RADIUS_DEFAULT = 0.018
    CUP_MOUTH_Z = 0.086                # 021_cup mouth height above table (ball enters here)
    CUP_CENTER_Z = 0.043               # cup origin height when base is on the table
    CUP_FWD_CLEARANCE_DEFAULT = 0.02   # land the ball (and place the cup) this far in FRONT (-y) of the
                                       # ramp's front edge, so the cup/gripper clear the ramp on placement
    BALL_REST_Z = 0.035                # ball-center height at rest inside the cup (for landing prediction)
    X_SPAN_DEFAULT = 0.15              # ball top/edge x sampled in [-X_SPAN, +X_SPAN] (diagonal roll)
    # all motion is TIME based (seconds) -> the kinematic roll lasts a random ROLL_TIME, then a short
    # accelerating fall; the robot idles a random IDLE_TIME before acting.
    ROLL_TIME_MIN_DEFAULT = 5.0        # s; random roll duration (t1)
    ROLL_TIME_MAX_DEFAULT = 20.0       # s
    FALL_TIME_DEFAULT = 0.5            # s; short accelerating drop off the edge into the cup
    IDLE_TIME_MIN_DEFAULT = 2.0        # s; robot observes this long before grasping the cup
    IDLE_TIME_MAX_DEFAULT = 4.0        # s
    SETTLE_STEPS_DEFAULT = 70          # physics steps for the final settle inside the cup
    RIM_RADIUS_DEFAULT = 0.040         # 021_cup inner catch radius
    GRAVITY = 9.81

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_ramp_ball", {})
        # The collector REUSES this env across episodes. _init_task_env_ runs load_camera
        # (which calls _update_kinematic_tasks) BEFORE the new load_actors rebuilds the
        # actors. Clear all per-episode state here so the guards in _update_kinematic_tasks
        # block until load_actors runs -- otherwise we touch the previous episode's
        # destroyed cup/ball entities and SAPIEN segfaults.
        self._loaded = False
        self._cup_welded = False
        self._ball_phase = None
        self._cup_ready = False
        super()._init_task_env_(**kwags)
        # Eval-readiness: start the ball rolling here so the ball motion is SELF-CONTAINED and does
        # not depend on the expert play_once. At evaluation (no play_once) the ball rolls from now and
        # FALLS after roll_steps (see _update_kinematic_tasks). For demo collection, play_once re-inits
        # these and gates the fall on the cup being placed (so demos stay clean -- behavior unchanged).
        self._ball_phase = "rolling"
        self._roll_i = 0
        self._expert_demo = False

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.table_top = 0.74 + self.table_z_bias
        ang = float(c.get("ramp_angle", self.RAMP_ANGLE_DEFAULT))
        cy = float(c.get("ramp_center_y", self.RAMP_CENTER_Y_DEFAULT))
        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.settle_steps = int(c.get("settle_steps", self.SETTLE_STEPS_DEFAULT))
        self.rim_radius = float(c.get("rim_radius", self.RIM_RADIUS_DEFAULT))
        self.ramp_angle = ang
        hx, hy, hz = self.RAMP_HALF

        # ---- time-based motion sampling (deterministic per seed) ----
        x_span = float(c.get("x_span", self.X_SPAN_DEFAULT))
        roll_t = float(np.random.uniform(c.get("roll_time_min", self.ROLL_TIME_MIN_DEFAULT),
                                         c.get("roll_time_max", self.ROLL_TIME_MAX_DEFAULT)))
        fall_t = float(c.get("fall_time", self.FALL_TIME_DEFAULT))
        idle_t = float(np.random.uniform(c.get("idle_time_min", self.IDLE_TIME_MIN_DEFAULT),
                                         c.get("idle_time_max", self.IDLE_TIME_MAX_DEFAULT)))
        self.roll_time, self.fall_time, self.idle_time = roll_t, fall_t, idle_t
        self.roll_steps = int(round(roll_t * self.SIM_HZ))
        self.fall_steps = int(round(fall_t * self.SIM_HZ))
        self.idle_steps = int(round(idle_t * self.SIM_HZ))
        # random diagonal: x1 at the top, x2 at the bottom edge -> x-speed = (x2-x1)/roll_t
        self.x1 = float(np.random.uniform(-x_span, x_span))
        self.x2 = float(np.random.uniform(-x_span, x_span))

        q = t3d.euler.euler2quat(ang, 0, 0)            # far/+y edge UP
        self._R = t3d.quaternions.quat2mat(q)
        rise = hy * np.sin(ang)
        cz = self.table_top + self.LOW_EDGE_Z + rise
        self.ramp_center = np.array([0.0, cy, cz])

        self.ramp = create_box(
            self.scene, sapien.Pose(self.ramp_center.tolist(), q.tolist()),
            half_size=list(self.RAMP_HALF), color=[0.72, 0.64, 0.52], is_static=True, name="ramp",
        )
        for sx in (-1.0, 1.0):
            off = np.array([sx * hx, 0.0, self.RAIL_H * 0.5 + hz])
            rail_p = self.ramp_center + self._R @ off
            create_box(
                self.scene, sapien.Pose(rail_p.tolist(), q.tolist()),
                half_size=[0.006, hy, self.RAIL_H * 0.5], color=[0.5, 0.42, 0.32],
                is_static=True, name=f"rail_{'L' if sx < 0 else 'R'}",
            )
        # two support legs under the elevated board (front leg shorter, back leg taller -> follows
        # the incline) so it visibly holds the ramp up off the table
        for tag, ly in (("back", hy * 0.6), ("front", -hy * 0.6)):
            top = self.ramp_center + self._R @ np.array([0.0, ly, -hz])  # underside of board at this y
            leg_h = max(0.02, top[2] - self.table_top)
            create_box(
                self.scene, sapien.Pose([0.0, cy + ly, self.table_top + leg_h * 0.5]),
                half_size=[0.05, 0.025, leg_h * 0.5], color=[0.45, 0.4, 0.36],
                is_static=True, name=f"ramp_support_{tag}",
            )

        # ball starts at (x1, top) and rolls DIAGONALLY to (x2, edge): the x offset between top and
        # edge gives the lateral (x) speed = (x2-x1)/roll_t, so the ball travels diagonally.
        top_off = np.array([self.x1, hy * 0.85, hz + self.ball_radius])
        self.ball_top = self.ramp_center + self._R @ top_off
        edge_off = np.array([self.x2, -hy * 0.95, hz + self.ball_radius])
        self.ball_edge = self.ramp_center + self._R @ edge_off
        # lateral (x) velocity carried off the edge from the diagonal roll (keeps the diagonal look)
        self.roll_vel_3d = (self.ball_edge - self.ball_top) / max(roll_t, 1e-6)
        # REAL-physics fall: give a forward (-y) exit velocity so the ball clears the ramp's front
        # edge and lands CUP_FWD_CLEARANCE in front of it; gravity does the falling. Solve the exit
        # vy so the ball reaches its rest height (BALL_REST_Z) exactly CUP_FWD_CLEARANCE forward.
        self.cup_fwd_clearance = float(self._cfg.get("cup_fwd_clearance", self.CUP_FWD_CLEARANCE_DEFAULT))
        z_rest = self.table_top + self.BALL_REST_Z
        t_fall = float(np.sqrt(2.0 * max(1e-3, self.ball_edge[2] - z_rest) / self.GRAVITY))
        vy = -self.cup_fwd_clearance / t_fall
        self.release_vel = np.array([float(self.roll_vel_3d[0]), vy, 0.0])
        # extend ONLY the lateral (x) term by 2x t_fall to capture the ball's continued sideways
        # drift as it falls; the forward (y) term stays at the base t_fall.
        self.landing = np.array([float(self.ball_edge[0] + self.release_vel[0] * t_fall * 2.0),
                                 float(self.ball_edge[1] + vy * t_fall)])

        self.ball = create_sphere(
            self.scene, sapien.Pose(self.ball_top.tolist()), radius=self.ball_radius,
            color=[0.85, 0.18, 0.18], is_static=False, name="ball",
        )
        self._ball_comp = None
        for cc in self.ball.get_components():
            if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                self._ball_comp = cc
        # inelastic + grippy material: the tall ramp gives a harder impact, so kill the
        # bounce-out (restitution 0) and add friction so the ball settles in the cup
        try:
            inelastic = sapien.physx.PhysxMaterial(static_friction=0.9, dynamic_friction=0.9, restitution=0.0)
            for sh in self._ball_comp.get_collision_shapes():
                sh.set_physical_material(inelastic)
        except Exception:
            pass
        if self._ball_comp is not None:
            self._ball_comp.set_disable_gravity(True)
            self._ball_comp.set_kinematic(True)
            self._ball_comp.set_kinematic_target(sapien.Pose(self.ball_top.tolist()))

        # cup spawns central, reachable by either arm
        self.cup_id = 0
        cup_pose = rand_pose(
            xlim=[-0.06, 0.06], ylim=[-0.10, -0.03], zlim=[self.table_top],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
        )
        self.cup = create_actor(
            self, pose=cup_pose, modelname="021_cup", model_id=self.cup_id, convex=True, is_static=False,
        )
        try:
            self.cup.set_mass(0.05)
        except Exception:
            pass
        # match the cup interior to the ball: inelastic so the ball doesn't rebound off the wall
        try:
            cup_mat = sapien.physx.PhysxMaterial(static_friction=0.9, dynamic_friction=0.9, restitution=0.0)
            cc = self._cup_comp()
            if cc is not None:
                for sh in cc.get_collision_shapes():
                    sh.set_physical_material(cup_mat)
        except Exception:
            pass

        self.add_prohibit_area(self.ramp, padding=0.02)
        self.add_prohibit_area(self.cup, padding=0.05)

        # per-step / phase state (set before _init_task_env_ may re-enter; guard reads use getattr)
        self._cup_welded = False
        self._cup_ee_offset = None
        self._ball_phase = "frozen"
        self._roll_i = 0
        self._loaded = True

        # optional debug rendering (CRB_RENDER=1): side camera capturing the roll/catch
        self._dbg = bool(os.environ.get("CRB_RENDER"))
        self._dbg_n = 0
        if self._dbg:
            import numpy as _np
            cam = self.scene.add_camera("crb_dbg", 640, 480, 0.9, 0.01, 10.0)
            eye = _np.array([0.70, 0.02, self.table_top + 0.42]); tgt = _np.array([0.0, 0.04, self.table_top + 0.14])
            fwd = tgt - eye; fwd /= _np.linalg.norm(fwd); lf = _np.cross([0, 0, 1], fwd); lf /= _np.linalg.norm(lf); up = _np.cross(fwd, lf)
            m = _np.eye(4); m[:3, 0] = fwd; m[:3, 1] = lf; m[:3, 2] = up; m[:3, 3] = eye
            cam.set_pose(sapien.Pose(m)); self._dbg_cam = cam

    # ---------------------------------------------------------- landing math
    def _predict_landing(self):
        """The cup goes at the analytic REAL-physics projectile landing (precomputed in load_actors
        from the known release state), which sits CUP_FWD_CLEARANCE in front of the ramp edge."""
        return np.array(self.landing), 0.0

    def _cup_comp(self):
        for cc in self.cup.actor.get_components():
            if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                return cc
        return None

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if getattr(self, "_ball_phase", None) == "rolling":
            # diagonal roll: linear top->edge at constant velocity (x-speed = (x2-x1)/roll_t)
            self._roll_i += 1
            frac = min(1.0, self._roll_i / float(self.roll_steps))
            p = self.ball_top + (self.ball_edge - self.ball_top) * frac
            self._ball_comp.set_kinematic_target(sapien.Pose(p.tolist()))
            if frac >= 1.0:
                # EVAL (not an expert demo): release as soon as the roll completes (self-contained).
                # DEMO (expert play_once running): hold at the edge until the cup is placed.
                if getattr(self, "_cup_ready", False) or not getattr(self, "_expert_demo", False):
                    self._release_ball()

    def _release_ball(self):
        # leave the front edge as a REAL projectile: switch the ball to dynamic + gravity with a
        # forward (-y) exit velocity (release_vel) so it clears the ramp and lands at the analytic
        # landing the cup was placed at. Gravity does the falling -- no scripted descent.
        self._ball_phase = "released"
        self.ball.set_pose(sapien.Pose(self.ball_edge.tolist()))
        self._ball_comp.set_kinematic(False)
        self._ball_comp.set_disable_gravity(False)
        self._ball_comp.set_linear_velocity(self.release_vel.tolist())
        self._ball_comp.set_angular_velocity([0.0, 0.0, 0.0])
        try:
            # light damping only; inelastic material (set in load_actors) prevents bounce-out
            self._ball_comp.set_linear_damping(0.1)
            self._ball_comp.set_angular_damping(1.0)
        except Exception:
            pass

    # ------------------------------------------------------------ dwell
    def _dwell(self, steps):
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if getattr(self, "_dbg", False) and i % 12 == 0:
                from PIL import Image
                self.scene.update_render(); self._dbg_cam.take_picture()
                rgb = (self._dbg_cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(rgb).save(f"/shared_work/markhsp/DOMINO/preview_out/crb_{self._dbg_n:03d}.png")
                self._dbg_n += 1

    # ------------------------------------------------------------- policy
    def play_once(self):
        landing, _ = self._predict_landing()
        x_land, y_land = float(landing[0]), float(landing[1])
        arm_tag = ArmTag("right" if x_land > 0 else "left")

        # The ball rolls from the VERY START. The robot's "wait" is really it OBSERVING the ball roll;
        # all the arm motions below happen WHILE the ball is rolling (the action-execution loops call
        # _update_kinematic_tasks every control step, so _roll_i advances throughout). The ball will
        # NOT fall until _cup_ready -> if it reaches the edge before the cup is placed it holds there.
        self._ball_phase = "rolling"
        self._roll_i = 0
        self._cup_ready = False
        self._expert_demo = True   # demo: hold the ball at the edge until the cup is placed

        # 1. observe: robot stays still for a random 2-4 s while the ball rolls down
        self._dwell(self.idle_steps)

        # 2. pick up the central cup (ball still rolling)
        self.move(self.grasp_actor(self.cup, arm_tag=arm_tag, pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        # 3. carry it to the predicted landing, set it down (base on table) and release
        cup_now = np.array(self.cup.get_pose().p)
        target = np.array([x_land, y_land, self.table_top + self.CUP_CENTER_Z])
        d = target - cup_now
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=float(d[0]), y=float(d[1]), z=float(d[2]), move_axis="world"))
        self.move(self.open_gripper(arm_tag))                       # release -> cup stands free at the landing
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))  # lift clear of the cup

        # 4. return the arm to its home pose
        self.move(self.back_to_origin(arm_tag))

        # 5. cup is now in place -> allow the ball to leave the edge and fall in. Dwell only the
        # steps still needed: finish whatever roll remains (0 if it's holding at the edge) + fall + settle.
        self._cup_ready = True
        remaining_roll = max(0, self.roll_steps - self._roll_i)
        self._dwell(remaining_roll + self.fall_steps + self.settle_steps)

        if getattr(self, "_dbg", False):
            off, inv, bp, cp = self._catch_state()
            print(f"[crb] predicted_landing={np.round(self.landing,3).tolist()} edge={np.round(self.ball_edge,3).tolist()} "
                  f"ball_final={np.round(bp,3).tolist()} cup_final={np.round(cp,3).tolist()} offset={off:.3f} "
                  f"rim={self.rim_radius} in_vessel={inv} arm={arm_tag}", flush=True)
        self.info["info"] = {"{A}": "ball", "{B}": "021_cup/base0", "{a}": str(arm_tag)}
        return self.info

    # ------------------------------------------------------------- success
    def _catch_state(self):
        bp = np.array(self.ball.get_pose().p)
        cp = np.array(self.cup.get_pose().p)
        offset = float(np.linalg.norm(bp[:2] - cp[:2]))
        # ball within the cup mouth horizontally and resting in the cup (near table, not flung away)
        in_vessel = bool(offset < self.rim_radius and (self.table_top - 0.01) < bp[2] < (self.table_top + 0.12))
        return offset, in_vessel, bp, cp

    def check_success(self):
        if getattr(self, "_ball_phase", None) != "released":
            return False
        offset, in_vessel, _, _ = self._catch_state()
        return bool(in_vessel)

    def get_obs(self):
        obs = super().get_obs()
        try:
            offset, in_vessel, bp, cp = self._catch_state()
            score = float(np.clip(1.0 - offset / self.rim_radius, 0.0, 1.0)) if in_vessel else 0.0
            obs["catch"] = {
                "ball_xy": [float(bp[0]), float(bp[1])],
                "cup_xy": [float(cp[0]), float(cp[1])],
                "offset": float(offset),
                "in_vessel": float(in_vessel),
                "catch_score": score,
                "predicted_landing": [float(self.landing[0]), float(self.landing[1])] if hasattr(self, "landing") else [0.0, 0.0],
                "roll_time": float(getattr(self, "roll_time", 0.0)),
                "fall_time": float(getattr(self, "fall_time", 0.0)),
                "idle_time": float(getattr(self, "idle_time", 0.0)),
            }
        except Exception:
            pass
        return obs
