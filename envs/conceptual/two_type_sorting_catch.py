from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import sapien.physx
import numpy as np


class two_type_sorting_catch(Base_Task):
    """Dual-arm sorting catch (DOMINO Task 13, extends the Task-12 falling-object emitter).

    A far-center emitter drops TWO interleaved object types along left/right-biased curves:
    type-A (blue) leans LEFT, type-B (red) leans RIGHT. The LEFT arm holds a BLUE bowl and
    catches type-A; the RIGHT arm holds a RED bowl and catches type-B. A matching color bin
    sits in each near corner. Each arm owns its own half of the table; the emitter's bias
    keeps most objects on the correct side, with occasional near-center cases that force a
    quick which-bowl decision. The expert classifies each falling object, slides the matching
    bowl under its step-driven landing point, and catches it into the correct (type-matched)
    bowl. Metric: per-type sorting accuracy and macro-F1 over {A, B}; a wrong-bowl catch or a
    miss is an error.

    Novel mechanics layered on the standard primitives:
      * step-driven two-type emitter with biased curved falls, implemented by overriding
        ``_update_kinematic_tasks`` (the per-physics-step hook) so the two collection passes
        stay deterministic (everything is driven by an integer step counter, never by pixels);
      * a dwell loop that advances ``_update_kinematic_tasks`` + ``scene.step`` while recording
        frames every ``save_freq`` steps;
      * per-type catch bookkeeping recorded into the trajectory via ``get_obs``.
    """

    # ---- class-default params (overridable from task_config task_args) -----------------
    NUM_OBJECTS_DEFAULT = 4           # total objects emitted across the episode
    FALL_STEPS_DEFAULT = 90           # physics steps for one object to descend the curve
    EMIT_RADIUS_DEFAULT = 0.035       # projectile sphere radius
    LATERAL_BIAS_DEFAULT = 0.16       # |x| the curve drifts toward the owning side at landing
    CENTER_RATE_DEFAULT = 0.25        # fraction of objects spawned as near-center (ambiguous)
    CATCH_TOL_DEFAULT = 0.07          # xy tolerance (m) for a bowl to "catch" the object

    # type -> (display color rgb, side sign); A leans left (-x), B leans right (+x)
    TYPE_COLORS = {
        "A": ([0.10, 0.30, 0.95], -1.0),   # blue
        "B": ([0.95, 0.12, 0.12], +1.0),   # red
    }
    BOWL_COLORS = {
        "left": [0.10, 0.30, 0.95, 1.0],   # blue bowl  (left arm, catches A)
        "right": [0.95, 0.12, 0.12, 1.0],  # red  bowl  (right arm, catches B)
    }

    EMIT_Y = -0.18                    # far (toward the robot's far side) center emit line
    EMIT_Z_BIAS = 0.32               # height above table at which objects appear

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags is not otherwise stored on self)
        self._cfg = kwags.get("task_args", {}).get("two_type_sorting_catch", {})
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.num_objects = int(c.get("num_objects", self.NUM_OBJECTS_DEFAULT))
        self.fall_steps = int(c.get("fall_steps", self.FALL_STEPS_DEFAULT))
        self.emit_radius = float(c.get("emit_radius", self.EMIT_RADIUS_DEFAULT))
        self.lateral_bias = float(c.get("lateral_bias", self.LATERAL_BIAS_DEFAULT))
        self.center_rate = float(c.get("center_rate", self.CENTER_RATE_DEFAULT))
        self.catch_tol = float(c.get("catch_tol", self.CATCH_TOL_DEFAULT))
        self.table_top = 0.74 + self.table_z_bias

        # ---- random typed sequence + per-object curve parameters (seed-driven) ----------
        self.type_seq = [str(np.random.choice(["A", "B"])) for _ in range(self.num_objects)]
        self.is_center = [bool(np.random.rand() < self.center_rate) for _ in range(self.num_objects)]
        # phase / amplitude of the lateral curve, randomized per object
        self.curve_phase = [float(np.random.uniform(0.0, np.pi)) for _ in range(self.num_objects)]
        self.curve_amp = [float(np.random.uniform(0.03, 0.06)) for _ in range(self.num_objects)]

        # ---- bowls: 002_bowl recolored, one per arm, near each arm's side ----------------
        self.bowl_id = int(np.random.choice([1, 2, 3, 5]))
        left_bowl_pose = rand_pose(
            xlim=[-0.22, -0.12], ylim=[0.04, 0.12],
            zlim=[self.table_top], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
        )
        right_bowl_pose = rand_pose(
            xlim=[0.12, 0.22], ylim=[0.04, 0.12],
            zlim=[self.table_top], qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
        )
        self.left_bowl = create_actor(
            self, pose=left_bowl_pose, modelname="002_bowl",
            model_id=self.bowl_id, convex=True, is_static=False,
        )
        self.right_bowl = create_actor(
            self, pose=right_bowl_pose, modelname="002_bowl",
            model_id=self.bowl_id, convex=True, is_static=False,
        )
        self.left_bowl.set_mass(0.30)
        self.right_bowl.set_mass(0.30)
        self._recolor(self.left_bowl, self.BOWL_COLORS["left"])
        self._recolor(self.right_bowl, self.BOWL_COLORS["right"])

        # ---- color-coded bins in the near corners (110_basket): blue near-left, red near-right
        self.bin_id = int(np.random.choice([0, 1, 2, 3]))
        self.blue_bin = create_actor(
            self, pose=sapien.Pose([-0.27, 0.18, self.table_top], [0.707, 0.707, 0, 0]),
            modelname="110_basket", model_id=self.bin_id, convex=True, is_static=True,
        )
        self.red_bin = create_actor(
            self, pose=sapien.Pose([0.27, 0.18, self.table_top], [0.707, 0.707, 0, 0]),
            modelname="110_basket", model_id=self.bin_id, convex=True, is_static=True,
        )
        self._recolor(self.blue_bin, self.BOWL_COLORS["left"])
        self._recolor(self.red_bin, self.BOWL_COLORS["right"])

        # ---- emitter marker (static visual cube at far-center) ---------------------------
        self.emitter = create_box(
            scene=self, pose=sapien.Pose([0.0, self.EMIT_Y, self.table_top + 0.02], [1, 0, 0, 0]),
            half_size=(0.025, 0.025, 0.02), color=(0.3, 0.3, 0.3), name="emitter", is_static=True,
        )

        self.add_prohibit_area(self.left_bowl, padding=0.03)
        self.add_prohibit_area(self.right_bowl, padding=0.03)
        self.add_prohibit_area(self.blue_bin, padding=0.03)
        self.add_prohibit_area(self.red_bin, padding=0.03)

        # ---- emitter / catch bookkeeping -------------------------------------------------
        self.proj = None              # current in-flight projectile entity
        self.proj_physx = None
        self._proj_idx = -1           # index of object currently/last in flight
        self._fall_step = 0           # step counter within the current fall
        self._fall_start = None       # (x0,y0,z0) emit point
        self._fall_end = None         # (x1,y1) landing point
        self._catch_target = None     # bowl actor to slide under this object
        self._catch_locked = False    # whether the catch has been registered for current proj
        self._caught_objs = []        # spawned-and-resolved projectile entities (kept resting)

        # per-type confusion bookkeeping: caught_correct / caught_wrong / missed by type
        self.results = {t: {"correct": 0, "wrong": 0, "missed": 0} for t in ("A", "B")}
        self.events = []              # per-object outcome log

    # ----------------------------------------------------------- recolor helper
    def _recolor(self, actor, rgba):
        try:
            comps = actor.actor.get_components()
        except AttributeError:
            comps = actor.get_components()
        for comp in comps:
            if isinstance(comp, sapien.render.RenderBodyComponent):
                for s in comp.render_shapes:
                    try:
                        s.material.set_base_color(list(rgba[:3]) + [1.0])
                    except Exception:
                        pass

    # ---------------------------------------------------- step-driven emitter
    def _emit(self, idx):
        """Spawn projectile `idx` at far-center with a side-biased curved descent."""
        t = self.type_seq[idx]
        rgb, side = self.TYPE_COLORS[t]
        # near-center cases get a reduced bias toward the (still type-defined) side
        bias = self.lateral_bias * (0.25 if self.is_center[idx] else 1.0)
        x1 = float(np.clip(side * bias, -0.24, 0.24))   # landing x on the owning side
        y1 = 0.06                                        # landing y near the bowls
        x0 = float(side * 0.02 * (0.0 if self.is_center[idx] else 1.0))  # emit near center
        z0 = self.table_top + self.EMIT_Z_BIAS

        ent = create_sphere(
            scene=self, pose=sapien.Pose([x0, self.EMIT_Y, z0], [1, 0, 0, 0]),
            radius=self.emit_radius, color=rgb, name=f"proj_{idx}_{t}",
        )
        physx = None
        for comp in ent.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                physx = comp
                comp.set_kinematic(True)   # we drive the pose deterministically by step count
        self.proj = ent
        self.proj_physx = physx
        self._proj_idx = idx
        self._fall_step = 0
        self._fall_start = np.array([x0, self.EMIT_Y, z0])
        self._fall_end = np.array([x1, y1])
        self._catch_locked = False

    def _advance_fall(self):
        """Move the in-flight projectile one step along its biased curve (kinematic)."""
        if self.proj is None or self._catch_locked:
            return
        idx = self._proj_idx
        n = max(1, self.fall_steps)
        s = min(1.0, self._fall_step / n)              # 0..1 progress, step-driven
        x0, y0, z0 = self._fall_start
        x1, y1 = self._fall_end
        # lateral curve: ease toward landing x with a sinusoidal lateral wiggle
        lat = self.curve_amp[idx] * np.sin(self.curve_phase[idx] + np.pi * s)
        x = x0 + (x1 - x0) * s + lat * (1.0 - s)
        y = y0 + (y1 - y0) * s
        z = z0 - (z0 - (self.table_top + 0.10)) * s    # descend to just above the bowl rim
        self.proj.set_pose(sapien.Pose([float(x), float(y), float(z)], [1, 0, 0, 0]))
        self._fall_step += 1

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's registered kinematic motions; run it first
        super()._update_kinematic_tasks()
        # _init_task_env_ calls this during setup, before load_actors initializes state
        if not hasattr(self, "proj"):
            return
        # then advance our own step-driven projectile descent
        self._advance_fall()

    # --------------------------------------------------------------- dwell loop
    def _dwell(self, steps):
        """Step the sim for `steps`, advancing the emitter and recording frames."""
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    # ----------------------------------------------------- catch resolution
    def _resolve_catch(self):
        """Register the outcome of the current projectile and rest it (kinematic) in place."""
        if self.proj is None or self._catch_locked:
            return
        idx = self._proj_idx
        t = self.type_seq[idx]
        # the matching bowl is the one this arm should have moved under the object
        match_arm = "left" if t == "A" else "right"
        bowl = self.left_bowl if match_arm == "left" else self.right_bowl
        proj_xy = np.array(self.proj.get_pose().p[:2])
        bowl_xy = np.array(bowl.get_pose().p[:2])
        dist = float(np.linalg.norm(proj_xy - bowl_xy))

        # also check the *wrong* bowl (a wrong-bowl catch is an error, not a success)
        other = self.right_bowl if match_arm == "left" else self.left_bowl
        other_xy = np.array(other.get_pose().p[:2])
        other_dist = float(np.linalg.norm(proj_xy - other_xy))

        if dist <= self.catch_tol and dist <= other_dist:
            outcome = "correct"
            self.results[t]["correct"] += 1
        elif other_dist <= self.catch_tol:
            outcome = "wrong"
            self.results[t]["wrong"] += 1
        else:
            outcome = "missed"
            self.results[t]["missed"] += 1
        self.events.append({"idx": idx, "type": t, "outcome": outcome, "dist": dist})

        # rest the projectile where it ended (drop it into / near the bowl)
        self._caught_objs.append(self.proj)
        self.proj = None
        self.proj_physx = None
        self._catch_locked = True

    # ------------------------------------------------------------- policy
    def play_once(self):
        # 1) each arm grasps and lifts its color-coded bowl
        self.move(
            self.grasp_actor(self.left_bowl, arm_tag=ArmTag("left"),
                             pre_grasp_dis=0.08, grasp_dis=0.0, contact_point_id=2),
            self.grasp_actor(self.right_bowl, arm_tag=ArmTag("right"),
                             pre_grasp_dis=0.08, grasp_dis=0.0, contact_point_id=0),
        )
        self.move(
            self.move_by_displacement(ArmTag("left"), z=0.10, move_axis="arm"),
            self.move_by_displacement(ArmTag("right"), z=0.10, move_axis="arm"),
        )

        # 2) sort each emitted object into the type-matched bowl
        for idx in range(self.num_objects):
            t = self.type_seq[idx]
            match_arm = ArmTag("left" if t == "A" else "right")
            other_arm = match_arm.opposite

            # emit the object and let it begin its biased descent
            self._emit(idx)
            self._dwell(max(8, self.fall_steps // 3))

            # 3) classify (known type t) and slide the matching bowl under the landing x.
            #    Move along the gripper x toward the predicted landing point; the other arm
            #    holds station. Relative displacement plans more reliably than absolute IK.
            land_x = float(self._fall_end[0])
            bowl_actor = self.left_bowl if t == "A" else self.right_bowl
            cur_x = float(bowl_actor.get_pose().p[0])
            dx = float(np.clip(land_x - cur_x, -0.18, 0.18))
            self.move(self.move_by_displacement(match_arm, x=dx))

            # 4) let the object finish falling into the (now positioned) bowl, then catch
            self._dwell(self.fall_steps)
            self._resolve_catch()

            # small settle dwell so the catch renders
            self._dwell(max(6, self.save_freq or 6))

        # macro-F1 over {A,B} computed at the end
        self._compute_metrics()

        self.info["info"] = {
            "{A}": f"002_bowl/base{self.bowl_id} (blue, left)",
            "{B}": f"002_bowl/base{self.bowl_id} (red, right)",
            "{a}": "left+right",
        }
        return self.info

    # ------------------------------------------------------------- metrics
    def _compute_metrics(self):
        # per-type accuracy = correct / total-of-that-type
        per_type_acc = {}
        f1s = []
        for t in ("A", "B"):
            r = self.results[t]
            total = r["correct"] + r["wrong"] + r["missed"]
            per_type_acc[t] = (r["correct"] / total) if total else 0.0
            # treat "catch object of type t into the t-bowl" as the positive class:
            # TP = correct; FP = wrong catches landing in the t-bowl from the OTHER type;
            # FN = this type's wrong + missed. Approximate precision/recall from the log.
            tp = r["correct"]
            fn = r["wrong"] + r["missed"]
            other = "B" if t == "A" else "A"
            fp = self.results[other]["wrong"]   # other-type objects mis-caught into t's bowl
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
            f1s.append(f1)
        self.metrics = {
            "per_type_accuracy": per_type_acc,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "results": self.results,
            "events": self.events,
        }

    # ------------------------------------------------------------- success
    def check_success(self):
        # success: every emitted object was caught into the correct (type-matched) bowl,
        # with no wrong-bowl catches and no misses.
        total_correct = sum(self.results[t]["correct"] for t in ("A", "B"))
        total_wrong = sum(self.results[t]["wrong"] for t in ("A", "B"))
        total_missed = sum(self.results[t]["missed"] for t in ("A", "B"))
        if (total_correct + total_wrong + total_missed) < self.num_objects:
            return False
        return bool(total_correct == self.num_objects and total_wrong == 0 and total_missed == 0)

    # --------------------------------------- record per-type catches per frame
    def get_obs(self):
        obs = super().get_obs()
        obs["sorting"] = {
            "results": {t: dict(self.results[t]) for t in ("A", "B")},
            "macro_f1": float(getattr(self, "metrics", {}).get("macro_f1", 0.0)),
            "num_objects": int(self.num_objects),
        }
        return obs
