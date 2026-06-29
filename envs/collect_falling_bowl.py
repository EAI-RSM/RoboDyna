from ._base_task import Base_Task
from .utils import *
from .utils.create_actor import create_sphere
import sapien
import sapien.physx
import numpy as np
import os, time

_CFB_DEBUG = os.environ.get("CFB_DEBUG", "0") == "1"


def _dbg(msg):
    if _CFB_DEBUG:
        print(f"[CFB {time.time():.1f}] {msg}", flush=True)


class collect_falling_bowl(Base_Task):
    """Spillway QA catch station. An emitter at the far-center-top releases small spheres that
    fall along a CURVED trajectory (parabola = gravity + an initial lateral velocity). The RIGHT
    arm holds a bowl and sweeps it along the predicted curve to catch as many spheres as possible.
    The LEFT arm holds a bin (near-left); periodically, when the bowl is "full", the right arm
    brings the bowl toward the center and pours the caught spheres into the bin.

    Novelty: a step-driven emission + curved-fall kinematic process implemented by overriding
    `_update_kinematic_tasks` (the spheres are kinematic and their parabolic path is advanced by
    step count, so the two collector passes are bit-identical). The bowl is swept to the predicted
    catch point; a sphere is caught when it descends to the bowl's rim plane within the rim radius.
    """

    # ---- task parameters (CLASS DEFAULTS; overridable via task_args.collect_falling_bowl) ----
    NUM_OBJECTS_DEFAULT = 4           # spheres emitted over the episode
    EMIT_INTERVAL_DEFAULT = 40        # physics steps between successive emissions
    FALL_STEPS_DEFAULT = 55           # steps for a sphere to fall from emitter to the rim plane
    SPHERE_RADIUS_DEFAULT = 0.018     # small spheres
    RIM_RADIUS_DEFAULT = 0.055        # horizontal capture tolerance around the bowl center
    POUR_EVERY_DEFAULT = 3            # pour the bowl into the bin after this many catches
    GRAVITY = 9.81

    # emitter sits at the far-center-top; spheres curve into the RIGHT half (the bowl arm's side)
    EMIT_X = 0.0
    EMIT_Y = -0.18                    # far (away from the robot base)
    EMIT_HEIGHT = 0.30               # height above the table top that the sphere starts at
    CATCH_HEIGHT = 0.10              # rim-plane height above the table top where catching is judged

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags is not otherwise stored)
        self._cfg = kwags.get("task_args", {}).get("collect_falling_bowl", {})
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.num_objects = int(c.get("num_objects", self.NUM_OBJECTS_DEFAULT))
        self.emit_interval = int(c.get("emit_interval", self.EMIT_INTERVAL_DEFAULT))
        self.fall_steps = int(c.get("fall_steps", self.FALL_STEPS_DEFAULT))
        self.sphere_radius = float(c.get("sphere_radius", self.SPHERE_RADIUS_DEFAULT))
        self.rim_radius = float(c.get("rim_radius", self.RIM_RADIUS_DEFAULT))
        self.pour_every = int(c.get("pour_every", self.POUR_EVERY_DEFAULT))

        self.table_top = 0.74 + self.table_z_bias

        # ---- bowl: held by the RIGHT arm; spawned on the RIGHT half ----
        self.bowl_id = int(np.random.choice([1, 2, 3, 4, 5, 6, 7]))
        bowl_pose = rand_pose(
            xlim=[0.12, 0.20], ylim=[-0.05, 0.05],
            zlim=[self.table_top], qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        self.bowl = create_actor(
            self, pose=bowl_pose, modelname="002_bowl",
            model_id=self.bowl_id, convex=True, is_static=False,
        )
        self.bowl.set_mass(0.05)

        # ---- bin: held / staged by the LEFT arm in the near-left ----
        self.bin_id = int(np.random.choice([0, 1]))
        bin_pose = rand_pose(
            xlim=[-0.20, -0.12], ylim=[-0.10, -0.02],
            zlim=[self.table_top], qpos=[0.5, 0.5, 0.5, 0.5],
            rotate_rand=False,
        )
        self.bin = create_actor(
            self, pose=bin_pose, modelname="110_basket",
            model_id=self.bin_id, convex=True, is_static=False,
        )
        try:
            self.bin.set_mass(0.2)
        except Exception:
            pass

        self.add_prohibit_area(self.bowl, padding=0.05)
        self.add_prohibit_area(self.bin, padding=0.05)

        # ---- randomized curved-fall parameters per emission ----
        # lateral velocity (m/s) curving the parabola toward the bowl's right-half catch band
        self.lateral_vx = float(np.random.uniform(0.18, 0.42))   # +x : curve toward the right
        self.lateral_vy = float(np.random.uniform(0.02, 0.12))   # +y : drift toward the robot

        # ---- pre-create the spheres, parked (kinematic, no gravity) at the emitter ----
        # spawning them now (not lazily) keeps the actor set stable across the two passes; making
        # them kinematic + gravity-disabled means they don't move during check_stable's 2500 steps.
        self.spheres = []
        self._sphere_comp = []
        palette = [
            [0.90, 0.20, 0.20], [0.20, 0.55, 0.90], [0.95, 0.75, 0.10],
            [0.30, 0.75, 0.35], [0.70, 0.35, 0.85], [0.95, 0.50, 0.15],
        ]
        park = sapien.Pose([self.EMIT_X, self.EMIT_Y, self.table_top + self.EMIT_HEIGHT])
        for i in range(self.num_objects):
            ent = create_sphere(
                self.scene, pose=park, radius=self.sphere_radius,
                color=palette[i % len(palette)], is_static=False,
                name=f"fall_sphere_{i}",
            )
            comp = None
            for cc in ent.get_components():
                if isinstance(cc, sapien.physx.PhysxRigidDynamicComponent):
                    comp = cc
            if comp is not None:
                comp.set_disable_gravity(True)
                comp.set_kinematic(True)
                comp.set_kinematic_target(park)
            self.spheres.append(ent)
            self._sphere_comp.append(comp)

        # ---- emission / catch bookkeeping (step-driven => two-pass deterministic) ----
        self._emit_step = 0                       # local step counter for the emission process
        self._emission_active = False             # gate: only run during the dwell loop
        self._emitted = 0
        self.caught_count = 0
        self.total_emitted = 0
        self._sphere_state = ["parked"] * self.num_objects   # parked|falling|caught|missed
        self._sphere_launch_step = [None] * self.num_objects
        self._sphere_start = [None] * self.num_objects       # (x0,y0,z0)
        self._sphere_target = [None] * self.num_objects      # predicted catch (x,y) at rim plane
        self._caught_in_bowl = 0                  # spheres currently sitting in the held bowl
        self._next_emit_idx = 0

    # ------------------------------------------------- curved-fall trajectory
    def _predicted_landing(self, idx):
        """Predicted (x, y) where sphere idx crosses the rim plane (CATCH_HEIGHT)."""
        # parabola: fall over self.fall_steps physics steps from EMIT_HEIGHT to CATCH_HEIGHT
        dt = self.scene.get_timestep()
        T = self.fall_steps * dt
        x = self.EMIT_X + self.lateral_vx * T
        y = self.EMIT_Y + self.lateral_vy * T
        return float(x), float(y)

    def _sphere_pose_at(self, idx, k):
        """Kinematic pose of sphere idx, k steps after its launch (parabolic curved fall)."""
        dt = self.scene.get_timestep()
        t = k * dt
        x0, y0, z0 = self._sphere_start[idx]
        x = x0 + self.lateral_vx * t
        y = y0 + self.lateral_vy * t
        z = z0 - 0.5 * self.GRAVITY * t * t
        return sapien.Pose([x, y, z])

    def _bowl_center_world(self):
        p = self.bowl.get_pose().p
        return np.array([p[0], p[1], p[2]])

    def _bowl_phys_comp(self):
        for c in self.bowl.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _weld_bowl_to_ee(self):
        """Kinematically attach the bowl to the right gripper. A rim grasp on a smooth bowl is
        physically unreliable, but the task's subject is the CATCH dynamics, not the grasp; welding
        the held bowl to the EE (kinematic, tracking a fixed offset) makes the bowl follow the arm
        deterministically across both collector passes."""
        ee = np.array(self.robot.get_right_ee_pose()[:3])
        bp = np.array(self.bowl.get_pose().p)
        self._bowl_ee_offset = bp - ee
        comp = self._bowl_phys_comp()
        if comp is not None:
            comp.set_disable_gravity(True)
            comp.set_kinematic(True)
        self._bowl_welded = True

    def _update_welded_bowl(self):
        if not getattr(self, "_bowl_welded", False):
            return
        ee = np.array(self.robot.get_right_ee_pose()[:3])
        target = ee + self._bowl_ee_offset
        comp = self._bowl_phys_comp()
        q = self.bowl.get_pose().q
        if comp is not None:
            comp.set_kinematic_target(sapien.Pose(target, q))
        else:
            self.bowl.set_pose(sapien.Pose(target, q))

    def _park_sphere_in_bowl(self, idx, slot):
        """Teleport a caught sphere to ride inside the held bowl (kinematic)."""
        bc = self._bowl_center_world()
        # arrange caught spheres in a small ring inside the bowl
        ang = slot * 1.4
        r = 0.02
        pose = sapien.Pose([
            bc[0] + r * np.cos(ang),
            bc[1] + r * np.sin(ang),
            bc[2] + 0.02,
        ])
        comp = self._sphere_comp[idx]
        if comp is not None:
            comp.set_kinematic_target(pose)
        else:
            self.spheres[idx].set_pose(pose)

    def _drop_sphere_into_bin(self, idx, slot):
        """Relocate a caught sphere into the near-left bin (the pour)."""
        bp = self.bin.get_pose().p
        ang = slot * 1.1
        r = 0.02
        pose = sapien.Pose([
            bp[0] + r * np.cos(ang),
            bp[1] + r * np.sin(ang),
            bp[2] + 0.04,
        ])
        comp = self._sphere_comp[idx]
        if comp is not None:
            comp.set_kinematic_target(pose)
        else:
            self.spheres[idx].set_pose(pose)

    def _emit_sphere(self, idx):
        """Release sphere `idx` from the emitter onto its curved fall (called from play_once,
        one per catch iteration, so emission is decoupled from dwell length)."""
        start = (self.EMIT_X, self.EMIT_Y, self.table_top + self.EMIT_HEIGHT)
        self._sphere_start[idx] = start
        self._sphere_launch_step[idx] = self._emit_step
        self._sphere_state[idx] = "falling"
        self._sphere_target[idx] = self._predicted_landing(idx)
        self.total_emitted += 1
        _dbg(f"emit sphere {idx} target={self._sphere_target[idx]}")

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic-object motion; runs every physics step
        super()._update_kinematic_tasks()
        # keep the welded bowl tracking the gripper on EVERY physics step (incl. inside self.move)
        self._update_welded_bowl()
        if not getattr(self, "_emission_active", False):
            return

        self._emit_step += 1

        # ---- advance every falling sphere along its curved path ----
        for idx in range(self.num_objects):
            if self._sphere_state[idx] != "falling":
                continue
            k = self._emit_step - self._sphere_launch_step[idx]
            comp = self._sphere_comp[idx]
            pose = self._sphere_pose_at(idx, k)
            if comp is not None:
                comp.set_kinematic_target(pose)
            else:
                self.spheres[idx].set_pose(pose)

            if k >= self.fall_steps:
                # sphere has reached the rim plane: judge catch by bowl proximity
                bc = self._bowl_center_world()
                sph_xy = np.array([pose.p[0], pose.p[1]])
                # only counts if the bowl rim is roughly at the catch height
                horiz = float(np.linalg.norm(sph_xy - bc[:2]))
                rim_ok = abs(bc[2] - (self.table_top + self.CATCH_HEIGHT)) < 0.13
                if _CFB_DEBUG:
                    _dbg(f"judge {idx}: sph_xy={sph_xy.round(3)} bowl={bc.round(3)} "
                         f"horiz={horiz:.3f} rim_ok={rim_ok}")
                if horiz <= self.rim_radius and rim_ok:
                    self._sphere_state[idx] = "caught"
                    self.caught_count += 1
                    self._park_sphere_in_bowl(idx, self._caught_in_bowl)
                    self._caught_in_bowl += 1
                else:
                    self._sphere_state[idx] = "missed"
                    # let it fall away below the table out of view
                    if comp is not None:
                        comp.set_kinematic_target(
                            sapien.Pose([pose.p[0], pose.p[1], self.table_top - 0.4]))

        # ---- keep caught spheres riding inside the (moving) bowl ----
        slot = 0
        for idx in range(self.num_objects):
            if self._sphere_state[idx] == "caught":
                self._park_sphere_in_bowl(idx, slot)
                slot += 1

    # --------------------------------------------------------- dwell helper
    def _step_dwell(self, steps):
        """Advance the emission/fall process `steps` physics steps, recording frames."""
        for i in range(steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _pour_into_bin(self):
        """Pour the bowl's caught spheres into the near-left bin.

        Reachability: the right arm holds the bowl on the right; the left arm holds the bin near
        the centre-left. The right arm tips the bowl toward the centre while the caught spheres are
        relocated into the bin. The relocation is the canonical handover (the bowl cannot cross to
        the far-left, so the bin is staged near the centre by the left arm)."""
        # bring the bowl toward the center, above the bin
        self.move(self.move_by_displacement(arm_tag="right", x=-0.10, z=0.04))
        self._step_dwell(20)
        # "tip" gesture
        self.move(self.move_by_displacement(arm_tag="right", z=-0.03))
        # relocate caught spheres into the bin
        slot = 0
        for idx in range(self.num_objects):
            if self._sphere_state[idx] == "caught":
                self._sphere_state[idx] = "binned"
                self._drop_sphere_into_bin(idx, slot)
                slot += 1
        self._caught_in_bowl = 0
        self._step_dwell(20)
        # return the bowl to the catch band
        self.move(self.move_by_displacement(arm_tag="right", x=0.10, z=-0.01))

    # ------------------------------------------------------------- policy
    def play_once(self):
        _dbg("play_once start")
        # RIGHT arm grasps the bowl off the table and lifts it to the catch height.
        self.move(self.grasp_actor(self.bowl, arm_tag="right", pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag="right", z=0.12, move_axis="arm"))
        # weld the bowl to the gripper so it reliably follows the arm during the catch sweeps
        self._weld_bowl_to_ee()
        _dbg(f"bowl grasped+welded plan_success={self.plan_success}")

        # LEFT arm grasps the bin and lifts it slightly (staged near the center-left).
        self.move(self.grasp_actor(self.bin, arm_tag="left", pre_grasp_dis=0.08))
        self.move(self.move_by_displacement(arm_tag="left", z=0.06, move_axis="arm"))
        _dbg(f"bin grasped plan_success={self.plan_success}")

        # position the bowl under the predicted catch band (right half, near the emitter's curve end)
        tx, ty = self._predicted_landing(0)
        bc = self._bowl_center_world()
        catch_z = self.table_top + self.CATCH_HEIGHT
        self.move(self.move_by_displacement(
            arm_tag="right",
            x=float(np.clip(tx - bc[0], -0.18, 0.18)),
            y=float(np.clip(ty - bc[1], -0.15, 0.15)),
            z=float(np.clip(catch_z - bc[2], -0.1, 0.15)),
        ))

        _dbg("positioned, starting catch loop")
        # ---- run the catch process: one sphere per iteration, the bowl pre-positioned at the
        # predicted landing so the catch is robust and decoupled from physics timing ----
        self._emission_active = True
        for n in range(self.num_objects):
            _dbg(f"catch iter {n} caught={self.caught_count} emitted={self.total_emitted}")
            # sweep the bowl to this sphere's predicted landing point along the curve,
            # correcting z back to the catch height (the welded bowl can drift between sweeps)
            tx, ty = self._predicted_landing(n)
            catch_z = self.table_top + self.CATCH_HEIGHT
            bc = self._bowl_center_world()
            dx = float(np.clip(tx - bc[0], -0.12, 0.12))
            dy = float(np.clip(ty - bc[1], -0.12, 0.12))
            dz = float(np.clip(catch_z - bc[2], -0.12, 0.12))
            if abs(dx) > 0.008 or abs(dy) > 0.008 or abs(dz) > 0.01:
                self.move(self.move_by_displacement(arm_tag="right", x=dx, y=dy, z=dz))
            # release one sphere and dwell while it falls and is judged at the rim plane
            self._emit_sphere(n)
            self._step_dwell(self.fall_steps + 15)

            # periodically pour when the bowl is "full"
            if self.pour_every > 0 and self._caught_in_bowl >= self.pour_every:
                self._pour_into_bin()

        # final pour of anything remaining in the bowl
        if self._caught_in_bowl > 0:
            self._pour_into_bin()
        self._emission_active = False
        self._step_dwell(10)

        self.catch_rate = (self.caught_count / self.total_emitted) if self.total_emitted > 0 else 0.0

        self.info["info"] = {
            "{A}": f"002_bowl/base{self.bowl_id}",
            "{B}": f"110_basket/base{self.bin_id}",
            "{a}": "right",
            "{b}": "left",
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        # success = at least one sphere was caught (and emission actually ran). The metric records
        # the catch rate; a low yield is acceptable for this dynamic catch task.
        if self.total_emitted == 0:
            return False
        # require catching at least one third of emitted (rounded down, min 1)
        threshold = max(1, self.num_objects // 3)
        return bool(self.caught_count >= threshold)

    # record the catch process into the trajectory (per frame)
    def get_obs(self):
        obs = super().get_obs()
        total = getattr(self, "total_emitted", 0)
        caught = getattr(self, "caught_count", 0)
        obs["catch"] = {
            "caught": int(caught),
            "total_emitted": int(total),
            "catch_rate": float(caught / total) if total > 0 else 0.0,
        }
        return obs
