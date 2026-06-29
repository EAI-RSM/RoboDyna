from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np


class stamp_moving_files(Base_Task):
    """A conveyor belt carries a row of file-boxes through a mid zone. A stamp head is
    mounted on a FIXED gantry above one fixed point of the belt and descends when its
    button (in the near-right zone) is pressed. The RIGHT arm hovers over the button and
    presses it at the right instant so the descending stamp meets each passing file's
    target spot; the left arm idles.

    The stamp is the actuator (NOT held by the arm). Belt motion and the stamp-head
    descent are both step-driven via an override of `_update_kinematic_tasks`, so the two
    collection passes (plan / render) stay byte-identical. The mark on each file is
    recorded as the file's position at the instant the descending stamp contacts the belt
    plane, scored against that file's randomized target-spot offset."""

    # ----------------------------------------------------------- class defaults
    # Belt advances once per physics step; the stamp cycle is ~2*travel+hold steps, so the
    # file spacing must exceed belt_speed * cycle so each file clears one full stamp cycle.
    N_FILES_DEFAULT = 4               # number of file-boxes on the belt
    BELT_SPEED_DEFAULT = 0.0030       # m advanced per physics step
    FILE_SPACING_DEFAULT = 0.16       # nominal gap (m) between consecutive files along the belt
    STAMP_TOL_DEFAULT = 0.035         # tolerance (m) for the offset->score clamp

    # geometry of the fixed installation (table-local, x,y in metres; z added to table top)
    BELT_X = 0.0                      # belt runs along the table centreline (lateral midline)
    BELT_HALF_LEN = 0.30              # belt extent along y (far <-> near)
    BELT_HALF_WID = 0.07              # belt extent along x
    BELT_THICK = 0.012                # belt slab half-thickness
    STAMP_Y = -0.02                   # fixed belt point the stamp sits above (near table centre)
    STAMP_UP_DZ = 0.16                # stamp resting height above the belt surface
    STAMP_DOWN_DZ = 0.012             # stamp height above belt surface at full descent (= contact)
    STAMP_TRAVEL_STEPS = 10           # physics steps for a full stamp down-stroke (the actuation latency)
    STAMP_HOLD_STEPS = 4              # steps held at the bottom (the mark dwell)

    FILE_HALF = [0.045, 0.05, 0.018]  # half-extents of a file-box
    BELT_Y_START = 0.24               # y where the lead file enters (far end)
    BELT_Y_END = -0.27               # y where files leave the belt (near end)

    # button in the near-right zone (RIGHT arm presses it)
    BUTTON_X = 0.22
    BUTTON_Y = -0.16
    BUTTON_HALF = [0.03, 0.03, 0.018]

    def setup_demo(self, **kwags):
        # capture task-scoped params BEFORE init (kwags is not stored on self otherwise)
        self._cfg = kwags.get("task_args", {}).get("stamp_moving_files", {})
        # guards: _update_kinematic_tasks runs from the very first gripper-open inside
        # _init_task_env_ (BEFORE load_actors) and throughout check_stable (~2500 steps).
        # Belt motion + stamp actuation must NOT run until the policy starts.
        self._stamp_ready = False
        self._belt_running = False
        super()._init_task_env_(**kwags)

    # --------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.n_files = int(cfg.get("n_files", self.N_FILES_DEFAULT))
        self.stamp_tol = float(cfg.get("stamp_tol", self.STAMP_TOL_DEFAULT))

        # ---- per-episode randomization (seed-driven) ----
        self.belt_speed = float(cfg.get("belt_speed",
                                        self.BELT_SPEED_DEFAULT * np.random.uniform(0.8, 1.25)))
        self.file_spacing = float(cfg.get("file_spacing",
                                          self.FILE_SPACING_DEFAULT * np.random.uniform(0.9, 1.15)))

        z0 = 0.74 + self.table_z_bias                  # table top
        self.belt_top_z = z0 + 2 * self.BELT_THICK     # surface files ride on

        # ---- the belt slab (static scenery) ----
        self.belt = create_box(
            scene=self,
            pose=sapien.Pose([self.BELT_X, 0.0, z0 + self.BELT_THICK], [1, 0, 0, 0]),
            half_size=[self.BELT_HALF_WID, self.BELT_HALF_LEN, self.BELT_THICK],
            color=[0.18, 0.18, 0.20],
            name="belt",
            is_static=True,
        )

        # ---- the fixed gantry + stamp head over a fixed belt point ----
        self.stamp_x = self.BELT_X
        self.stamp_y = self.STAMP_Y
        self.stamp_up_z = self.belt_top_z + self.STAMP_UP_DZ
        self.stamp_down_z = self.belt_top_z + self.STAMP_DOWN_DZ

        # two static gantry posts + a crossbar (pure scenery, immovable)
        post_h = self.STAMP_UP_DZ + 0.10
        for sx in (-1, 1):
            create_box(
                scene=self,
                pose=sapien.Pose([self.stamp_x + sx * (self.BELT_HALF_WID + 0.02),
                                  self.stamp_y, z0 + post_h / 2], [1, 0, 0, 0]),
                half_size=[0.012, 0.012, post_h / 2],
                color=[0.45, 0.45, 0.48],
                name=f"gantry_post_{sx}",
                is_static=True,
            )
        create_box(
            scene=self,
            pose=sapien.Pose([self.stamp_x, self.stamp_y, z0 + post_h], [1, 0, 0, 0]),
            half_size=[self.BELT_HALF_WID + 0.035, 0.014, 0.014],
            color=[0.45, 0.45, 0.48],
            name="gantry_bar",
            is_static=True,
        )

        # the stamp head: a kinematic dynamic body driven by set_pose every step
        self.stamp_half = [0.03, 0.03, 0.03]
        self.stamp = create_box(
            scene=self,
            pose=sapien.Pose([self.stamp_x, self.stamp_y, self.stamp_up_z], [1, 0, 0, 0]),
            half_size=self.stamp_half,
            color=[0.85, 0.15, 0.12],
            name="stamp_head",
            is_static=False,
        )
        self._make_kinematic(self.stamp)

        # ---- the button (near-right; pressed by the RIGHT arm) ----
        self.button = create_box(
            scene=self,
            pose=sapien.Pose([self.BUTTON_X, self.BUTTON_Y, z0 + self.BUTTON_HALF[2]], [1, 0, 0, 0]),
            half_size=self.BUTTON_HALF,
            color=[0.10, 0.65, 0.18],
            name="button",
            is_static=True,
        )

        # ---- the files riding the belt ----
        # spawn them spaced along +y (far end) so they advance toward the near end.
        self.files = []
        self.file_target_off = []     # per-file target-spot offset along belt (local +y from file centre)
        self.file_marked = []         # whether the stamp has marked this file yet
        self.file_mark_offset = []    # recorded |mark - target| for scoring
        self.file_mark_shapes = []    # render shapes for the visible mark recolor
        base_color = [0.80, 0.72, 0.52]
        for i in range(self.n_files):
            y_i = self.BELT_Y_START + i * self.file_spacing
            f = create_box(
                scene=self,
                pose=sapien.Pose([self.BELT_X, y_i, self.belt_top_z + self.FILE_HALF[2]], [1, 0, 0, 0]),
                half_size=self.FILE_HALF,
                color=base_color,
                name=f"file_{i}",
                is_static=False,
            )
            self._make_kinematic(f)
            self.files.append(f)
            # randomized target-spot offset on each file (where the stamp should land on it)
            self.file_target_off.append(float(np.random.uniform(-0.025, 0.025)))
            self.file_marked.append(False)
            self.file_mark_offset.append(None)
            shapes = []
            for c in f.actor.get_components():
                if isinstance(c, sapien.render.RenderBodyComponent):
                    shapes = list(c.render_shapes)
            self.file_mark_shapes.append(shapes)

        # ---- stamp / belt runtime state (step-driven) ----
        self.stamp_phase = "up"       # "up" | "down" | "hold" | "rising"
        self.stamp_phase_step = 0
        self.stamp_requested = False  # set True when the button is pressed; consumed when idle&up
        self._belt_accum = 0.0        # fractional-step belt advance accumulator
        self.stamp_score = 0.0

        # remember the authored spawn poses so we can restore them after check_stable
        # (which steps the sim ~2500 times before play_once) and the stamp resting height.
        self._file_init_poses = [f.get_pose() for f in self.files]
        self._stamp_init_z = self.stamp_up_z

        # keep the moving installation out of the clutter/spawn areas
        self.add_prohibit_area(self.belt, padding=0.02)
        self.add_prohibit_area(self.button, padding=0.05)

        # now safe to let _update_kinematic_tasks touch belt/stamp state
        self._stamp_ready = True

    def _reset_belt(self):
        """Restore files + stamp to their authored start state at the policy start."""
        for f, pose in zip(self.files, self._file_init_poses):
            f.actor.set_pose(pose)
        self._set_pose_z(self.stamp, self.stamp_up_z)
        self.stamp_phase = "up"
        self.stamp_phase_step = 0
        self.stamp_requested = False

    def _make_kinematic(self, actor):
        for c in actor.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                c.set_kinematic(True)

    def _set_pose_z(self, actor, z):
        p = actor.get_pose()
        actor.actor.set_pose(sapien.Pose([p.p[0], p.p[1], z], p.q))

    # ----------------------------------------------------- step-driven dynamics
    def _advance_belt(self):
        """Move every (unmarked-or-marked) file one belt tick toward the near end."""
        for f in self.files:
            p = f.actor.get_pose()
            ny = p.p[1] - self.belt_speed
            f.actor.set_pose(sapien.Pose([p.p[0], ny, p.p[2]], p.q))

    def _file_target_world_y(self, i):
        """World-y of file i's target spot (file centre + its local offset)."""
        return float(self.files[i].get_pose().p[1] + self.file_target_off[i])

    def _record_mark(self):
        """At the instant of stamp contact, mark whichever file is under the stamp."""
        sx, sy = self.stamp_x, self.stamp_y
        best_i, best_d = None, 1e9
        for i, f in enumerate(self.files):
            if self.file_marked[i]:
                continue
            fp = f.get_pose().p
            # the file must actually be under the stamp footprint
            if abs(fp[0] - sx) < (self.FILE_HALF[0] + self.stamp_half[0]) and \
               abs(fp[1] - sy) < (self.FILE_HALF[1] + self.stamp_half[1]):
                d = abs(self._file_target_world_y(i) - sy)
                if d < best_d:
                    best_d, best_i = d, i
        if best_i is not None:
            self.file_marked[best_i] = True
            self.file_mark_offset[best_i] = best_d
            # paint a visible mark (darken the file) so the stamp is observable
            for s in self.file_mark_shapes[best_i]:
                try:
                    s.material.set_base_color([0.15, 0.12, 0.45, 1.0])
                except Exception:
                    pass
            self._update_score()

    def _update_score(self):
        scores = []
        for off in self.file_mark_offset:
            if off is None:
                continue
            scores.append(float(np.clip(1.0 - off / self.stamp_tol, 0.0, 1.0)))
        self.stamp_score = float(np.mean(scores)) if scores else 0.0

    def _update_kinematic_tasks(self):
        # base hook drives any registered DOMINO motion; runs every physics step
        super()._update_kinematic_tasks()

        # This hook also fires before load_actors (first gripper-open) and all through
        # check_stable; only drive belt/stamp once the policy has started.
        if not getattr(self, "_stamp_ready", False) or not getattr(self, "_belt_running", False):
            return

        # --- belt advance (one tick per physics step toward the near end) ---
        self._advance_belt()

        # --- stamp-head state machine ---
        if self.stamp_phase == "up":
            self._set_pose_z(self.stamp, self.stamp_up_z)
            if self.stamp_requested:
                self.stamp_requested = False
                self.stamp_phase = "down"
                self.stamp_phase_step = 0
        elif self.stamp_phase == "down":
            self.stamp_phase_step += 1
            t = min(1.0, self.stamp_phase_step / self.STAMP_TRAVEL_STEPS)
            z = self.stamp_up_z + (self.stamp_down_z - self.stamp_up_z) * t
            self._set_pose_z(self.stamp, z)
            if self.stamp_phase_step >= self.STAMP_TRAVEL_STEPS:
                self.stamp_phase = "hold"
                self.stamp_phase_step = 0
                self._record_mark()       # the mark is recorded at contact
        elif self.stamp_phase == "hold":
            self.stamp_phase_step += 1
            if self.stamp_phase_step >= self.STAMP_HOLD_STEPS:
                self.stamp_phase = "rising"
                self.stamp_phase_step = 0
        elif self.stamp_phase == "rising":
            self.stamp_phase_step += 1
            t = min(1.0, self.stamp_phase_step / self.STAMP_TRAVEL_STEPS)
            z = self.stamp_down_z + (self.stamp_up_z - self.stamp_down_z) * t
            self._set_pose_z(self.stamp, z)
            if self.stamp_phase_step >= self.STAMP_TRAVEL_STEPS:
                self.stamp_phase = "up"
                self.stamp_phase_step = 0

    def _stamp_idle(self):
        return self.stamp_phase == "up" and not self.stamp_requested

    def _press_button(self):
        """Fire the stamp (the button toggles/triggers the descent)."""
        self.stamp_requested = True

    def _belt_dwell(self, steps):
        """Advance belt + stamp for `steps` physics steps, recording frames periodically.
        Belt motion is enabled only for the duration of this loop."""
        prev = self._belt_running
        self._belt_running = True
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        self._belt_running = prev

    # ------------------------------------------------------------- policy
    def play_once(self):
        import os
        dbg = bool(os.environ.get("STAMP_DEBUG"))
        right = ArmTag("right")

        # Belt motion advances ONLY inside _belt_dwell (where _belt_running is True), never
        # during the arm's self.move(...) planning/execution -- this decouples the (variable)
        # arm-motion step count from belt timing so the press lands deterministically.
        self._reset_belt()
        self._belt_running = False

        # 1) right arm hovers over the button; left arm idles.
        z0 = 0.74 + self.table_z_bias
        hover = [self.BUTTON_X, self.BUTTON_Y, z0 + 0.14, 0, 1, 0, 0]
        self.move(self.move_to_pose(right, hover))

        # latency lead: belt distance a file covers during the stamp's down-stroke. We fire
        # when the target spot is `lead` short of the stamp so the descending head meets it.
        lead = self.belt_speed * self.STAMP_TRAVEL_STEPS
        cycle_steps = (2 * self.STAMP_TRAVEL_STEPS + self.STAMP_HOLD_STEPS) + 4

        max_wait = 8000
        waited = 0
        idx = 0
        n = self.n_files
        while idx < n and waited < max_wait:
            i = idx
            if self.file_marked[i]:
                idx += 1
                continue
            tgt_y = self._file_target_world_y(i)        # world-y of file i's target spot
            d = tgt_y - self.stamp_y                     # +ve: still approaching from far side
            if d <= lead + 1e-6 and d > -0.04:
                # press the button (a quick dip + raise; belt paused so timing is exact),
                # then trigger the descent and dwell while the stamp completes its stroke.
                self.move(self.move_by_displacement(right, z=-0.045))
                self._press_button()
                self.move(self.move_by_displacement(right, z=0.045))
                self._belt_dwell(cycle_steps)            # records the mark at contact
                if dbg:
                    print(f"[stamp] fired file {i}: marked={self.file_marked[i]} "
                          f"off={self.file_mark_offset[i]} score={self.stamp_score:.3f} "
                          f"plan={self.plan_success}", flush=True)
                idx += 1
            else:
                # advance the belt a few ticks and re-check
                step = max(1, int(min(8, (d - lead) / max(self.belt_speed, 1e-6))))
                self._belt_dwell(step)
                waited += step

        if dbg:
            print(f"[stamp] done marked={self.file_marked} offs={self.file_mark_offset} "
                  f"score={self.stamp_score:.3f} plan={self.plan_success}", flush=True)

        self.info["info"] = {
            "{A}": "file boxes",
            "{a}": str(right),
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        # all files must be stamped
        if not all(self.file_marked):
            return False
        self._update_score()
        return True

    # record the stamp marks / belt state into the trajectory (per-frame)
    def get_obs(self):
        obs = super().get_obs()
        self._update_score()
        obs["stamping"] = {
            "stamp_score": float(self.stamp_score),
            "n_files": int(self.n_files),
            "n_marked": int(sum(1 for m in self.file_marked if m)),
            "belt_speed": float(self.belt_speed),
            "file_positions": [float(f.get_pose().p[1]) for f in self.files],
            "mark_offsets": [None if o is None else float(o) for o in self.file_mark_offset],
            "stamp_z": float(self.stamp.get_pose().p[2]),
        }
        return obs
