from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import sapien.render
import numpy as np


class catch_marbles_trapdoors(Base_Task):
    """Four short conveyor belts span the far/mid zone, each carrying 2 marbles that advance
    toward a trapdoor point with a per-belt velocity profile (constant / accelerating /
    decelerating). A button in the near zone arms each belt's trapdoor: when the belt's button
    is pressed while a marble is over the trapdoor, the marble drops under gravity into a
    collection box sitting beneath the trapdoor line.

    The two LEFT belts/buttons are served by the LEFT arm, the two RIGHT by the RIGHT arm
    (each arm only reaches its own half of the table). The hard constraint is MUTUAL EXCLUSION:
    at most one button may be held at any instant. The expert policy schedules single,
    non-overlapping presses; any moment with two buttons held simultaneously VOIDS the episode.

    Introduces: 4 per-belt kinematic velocity profiles advanced in _update_kinematic_tasks, a
    press-armed trapdoor drop mechanic, a mutual-exclusion press schedule, and a marble-in-box
    catch metric -- on top of the standard press primitives.
    """

    # ----- geometry (class defaults; all in metres, table-local x,y) -----
    N_BELTS_DEFAULT = 4
    MARBLES_PER_BELT_DEFAULT = 2
    BELT_X_DEFAULT = [-0.21, -0.07, 0.07, 0.21]   # four belts left -> right
    BELT_HALF_W_DEFAULT = 0.035                   # belt strip half-width (x)
    BELT_Y_FAR_DEFAULT = 0.10                     # marbles start at the far end
    TRAPDOOR_Y_DEFAULT = -0.02                    # trapdoor / drop point along belt (near zone)
    BUTTON_Y_DEFAULT = -0.18                      # buttons sit in the near zone (reachable)
    BUTTON_HALF_DEFAULT = [0.022, 0.022, 0.018]   # button box half-size
    MARBLE_RADIUS_DEFAULT = 0.012
    MARBLE_SPACING_DEFAULT = 0.05                 # gap between the 2 marbles on a belt (y)

    # collection box (built from thin walls) centred under the trapdoor line
    BOX_Y_DEFAULT = -0.02
    BOX_HALF_W_DEFAULT = 0.30                     # x half-width (spans all four belts)
    BOX_HALF_D_DEFAULT = 0.055                    # y half-depth
    BOX_WALL_H_DEFAULT = 0.04                     # wall height
    BOX_WALL_T_DEFAULT = 0.006                    # wall thickness

    # belt velocity profiles: distance/step from BELT_Y_FAR toward TRAPDOOR_Y.
    # base nominal speed (m / sim step); profile reshapes it over the belt run.
    BELT_BASE_SPEED_DEFAULT = 0.0016
    # press timing: how long to hold a button (steps) and dwell budget per marble
    PRESS_HOLD_STEPS_DEFAULT = 24
    MARBLE_DWELL_STEPS_DEFAULT = 900               # max steps to wait for a marble to reach trapdoor
    DROP_WINDOW_DEFAULT = 0.045                     # |y - trapdoor_y| within which a press drops it

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_marbles_trapdoors", {})
        # pre-initialise per-step bookkeeping: the base env calls _update_kinematic_tasks()
        # during scene/robot setup, BEFORE load_actors() runs.
        self.marbles = []
        self.marble_belt = []
        self.marble_state = []
        self.marble_progress = []
        self._buttons_held = set()
        self._belt_armed = []
        self._mutex_violation = False
        self._dropped_count = 0
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.n_belts = int(c.get("n_belts", self.N_BELTS_DEFAULT))
        self.marbles_per_belt = int(c.get("marbles_per_belt", self.MARBLES_PER_BELT_DEFAULT))
        self.belt_x = list(c.get("belt_x", self.BELT_X_DEFAULT))
        self.belt_half_w = float(c.get("belt_half_w", self.BELT_HALF_W_DEFAULT))
        self.belt_y_far = float(c.get("belt_y_far", self.BELT_Y_FAR_DEFAULT))
        self.trapdoor_y = float(c.get("trapdoor_y", self.TRAPDOOR_Y_DEFAULT))
        self.button_y = float(c.get("button_y", self.BUTTON_Y_DEFAULT))
        self.button_half = list(c.get("button_half", self.BUTTON_HALF_DEFAULT))
        self.marble_radius = float(c.get("marble_radius", self.MARBLE_RADIUS_DEFAULT))
        self.marble_spacing = float(c.get("marble_spacing", self.MARBLE_SPACING_DEFAULT))
        self.box_y = float(c.get("box_y", self.BOX_Y_DEFAULT))
        self.box_half_w = float(c.get("box_half_w", self.BOX_HALF_W_DEFAULT))
        self.box_half_d = float(c.get("box_half_d", self.BOX_HALF_D_DEFAULT))
        self.box_wall_h = float(c.get("box_wall_h", self.BOX_WALL_H_DEFAULT))
        self.box_wall_t = float(c.get("box_wall_t", self.BOX_WALL_T_DEFAULT))
        self.belt_base_speed = float(c.get("belt_base_speed", self.BELT_BASE_SPEED_DEFAULT))
        self.press_hold_steps = int(c.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.marble_dwell_steps = int(c.get("marble_dwell_steps", self.MARBLE_DWELL_STEPS_DEFAULT))
        self.drop_window = float(c.get("drop_window", self.DROP_WINDOW_DEFAULT))

        self.table_z = 0.74 + self.table_z_bias
        self.belt_top_z = self.table_z + 0.006          # marble rest height on the belt surface

        # ---- per-belt randomized velocity-profile assignment ----
        profiles = ["constant", "accel", "decel"]
        self.belt_profile = [profiles[np.random.randint(len(profiles))] for _ in range(self.n_belts)]
        # per-belt speed multiplier (randomized speeds)
        self.belt_speed_mult = [float(np.random.uniform(0.8, 1.25)) for _ in range(self.n_belts)]
        # per-belt marble spacing (randomized)
        self.belt_spacing = [float(self.marble_spacing * np.random.uniform(0.85, 1.2)) for _ in range(self.n_belts)]

        # ---------------- build belts (cosmetic strips) ----------------
        self.belts = []
        belt_len = abs(self.belt_y_far - self.trapdoor_y) + 0.10
        belt_cy = (self.belt_y_far + self.trapdoor_y) / 2.0
        for bx in self.belt_x:
            strip = create_visual_box(
                self,
                pose=sapien.Pose([bx, belt_cy, self.table_z + 0.002]),
                half_size=[self.belt_half_w, belt_len / 2.0, 0.002],
                color=[0.18, 0.18, 0.22],
                name="belt_strip",
            )
            self.belts.append(strip)

        # ---------------- build the collection box (4 walls + floor) ----------------
        bx0, by0 = 0.0, self.box_y
        floor_z = self.table_z + 0.001
        # floor
        create_box(self, pose=sapien.Pose([bx0, by0, floor_z]),
                   half_size=[self.box_half_w, self.box_half_d, self.box_wall_t / 2.0],
                   color=[0.55, 0.4, 0.25], is_static=True, name="catch_box_floor")
        wh = self.box_wall_h
        wt = self.box_wall_t
        wall_cz = floor_z + wh / 2.0
        # front (near) and back (far) walls span x
        create_box(self, pose=sapien.Pose([bx0, by0 - self.box_half_d, wall_cz]),
                   half_size=[self.box_half_w, wt / 2.0, wh / 2.0],
                   color=[0.55, 0.4, 0.25], is_static=True, name="catch_box_front")
        create_box(self, pose=sapien.Pose([bx0, by0 + self.box_half_d, wall_cz]),
                   half_size=[self.box_half_w, wt / 2.0, wh / 2.0],
                   color=[0.55, 0.4, 0.25], is_static=True, name="catch_box_back")
        # left and right end walls span y
        create_box(self, pose=sapien.Pose([bx0 - self.box_half_w, by0, wall_cz]),
                   half_size=[wt / 2.0, self.box_half_d, wh / 2.0],
                   color=[0.55, 0.4, 0.25], is_static=True, name="catch_box_left")
        create_box(self, pose=sapien.Pose([bx0 + self.box_half_w, by0, wall_cz]),
                   half_size=[wt / 2.0, self.box_half_d, wh / 2.0],
                   color=[0.55, 0.4, 0.25], is_static=True, name="catch_box_right")
        self.box_center = np.array([bx0, by0])
        self.box_floor_z = floor_z + self.box_wall_t / 2.0

        # ---------------- buttons (small graspable/pressable boxes) ----------------
        btn_colors = [[0.85, 0.2, 0.2], [0.9, 0.6, 0.15], [0.2, 0.55, 0.85], [0.3, 0.7, 0.3]]
        self.buttons = []
        self.button_base_z = []
        for i, bx in enumerate(self.belt_x):
            bz = self.table_z + self.button_half[2]
            btn = create_box(
                self,
                pose=sapien.Pose([bx, self.button_y, bz]),
                half_size=list(self.button_half),
                color=btn_colors[i % len(btn_colors)],
                is_static=True,
                name=f"trap_button_{i}",
            )
            self.buttons.append(btn)
            self.button_base_z.append(bz)

        # ---------------- marbles (sphere primitives) ----------------
        marble_colors = [[0.9, 0.9, 0.95], [0.95, 0.85, 0.3], [0.4, 0.8, 0.9], [0.9, 0.5, 0.9]]
        self.marbles = []            # flat list of sapien.Entity
        self.marble_belt = []        # belt index for each marble
        self.marble_state = []       # 'belt' | 'dropping' | 'landed'
        self.marble_progress = []    # 0..1 along belt (0 = far, 1 = trapdoor)
        for bi, bx in enumerate(self.belt_x):
            for mi in range(self.marbles_per_belt):
                y0 = self.belt_y_far - mi * self.belt_spacing[bi]
                ent = create_sphere(
                    self,
                    pose=sapien.Pose([bx, y0, self.belt_top_z + self.marble_radius]),
                    radius=self.marble_radius,
                    color=marble_colors[bi % len(marble_colors)],
                    is_static=False,
                    name=f"marble_{bi}_{mi}",
                )
                # keep marbles kinematic while riding the belt (deterministic motion)
                for comp in ent.get_components():
                    if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                        comp.set_kinematic(True)
                        comp.set_mass(0.01)
                self.marbles.append(ent)
                self.marble_belt.append(bi)
                self.marble_state.append("belt")
                self.marble_progress.append(0.0)

        # ---------------- mutual-exclusion + scheduling bookkeeping ----------------
        self._buttons_held = set()      # which button indices are currently pressed
        self._belt_armed = [False] * self.n_belts   # a press has armed this belt's trapdoor (this hold)
        self._mutex_violation = False
        self._dropped_count = 0

        # reserve space so clutter / randomization stays clear of the apparatus
        for btn in self.buttons:
            self.add_prohibit_area(btn, padding=0.03)
        self.add_prohibit_area([0.0, self.box_y, self.table_z, 1, 0, 0, 0], padding=0.02)
        # prohibit the belt lanes
        for bx in self.belt_x:
            self.add_prohibit_area([bx, (self.belt_y_far + self.trapdoor_y) / 2.0, self.table_z, 1, 0, 0, 0],
                                   padding=0.04)

    # --------------------------------------------------------- belt dynamics
    def _profile_step(self, belt_idx, progress):
        """Per-step progress increment for a belt given its profile and current progress (0..1)."""
        base = self.belt_base_speed * self.belt_speed_mult[belt_idx]
        # convert linear speed (m/step) into progress/step over the belt run length
        run = abs(self.belt_y_far - self.trapdoor_y)
        if run <= 1e-6:
            return 1.0
        dprog = base / run
        prof = self.belt_profile[belt_idx]
        if prof == "accel":
            # slow at the far end, faster near the trapdoor
            factor = 0.4 + 1.2 * progress
        elif prof == "decel":
            # fast at the far end, slows approaching the trapdoor
            factor = 1.4 - 1.0 * progress
        else:  # constant
            factor = 1.0
        return dprog * max(0.05, factor)

    def _marble_belt_pose(self, belt_idx, progress):
        bx = self.belt_x[belt_idx]
        y = self.belt_y_far + (self.trapdoor_y - self.belt_y_far) * float(np.clip(progress, 0.0, 1.0))
        z = self.belt_top_z + self.marble_radius
        return sapien.Pose([bx, y, z])

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step
        super()._update_kinematic_tasks()

        # mutual-exclusion guard (checked every physics step)
        if len(self._buttons_held) > 1:
            self._mutex_violation = True

        for idx, ent in enumerate(self.marbles):
            state = self.marble_state[idx]
            bi = self.marble_belt[idx]
            if state == "belt":
                # advance along the belt with its profile, but stop AT the trapdoor point
                if self.marble_progress[idx] < 1.0:
                    self.marble_progress[idx] = min(
                        1.0, self.marble_progress[idx] + self._profile_step(bi, self.marble_progress[idx]))
                ent.set_pose(self._marble_belt_pose(bi, self.marble_progress[idx]))
                # if this belt's trapdoor is armed AND the marble is within the drop window, drop it
                if self._belt_armed[bi]:
                    y = ent.get_pose().p[1]
                    if abs(y - self.trapdoor_y) <= self.drop_window:
                        self._open_trapdoor_drop(idx)
            elif state == "dropping":
                # let physics carry it; mark landed once it settles near the box floor
                p = ent.get_pose().p
                if p[2] <= self.box_floor_z + self.marble_radius + 0.012:
                    self.marble_state[idx] = "landed"

    def _open_trapdoor_drop(self, marble_idx):
        ent = self.marbles[marble_idx]
        for comp in ent.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                comp.set_kinematic(False)
                comp.set_linear_velocity(np.array([0.0, 0.0, -0.05]))
                comp.set_angular_velocity(np.zeros(3))
                comp.set_linear_damping(0.2)
                comp.set_angular_damping(0.5)
        self.marble_state[marble_idx] = "dropping"
        self._dropped_count += 1

    # --------------------------------------------------------- press helpers
    def _press_button(self, arm_tag, btn_idx):
        """Press one button: approach its top, push down (arm trapdoor), hold, release, lift.
        Records the button as held for mutual-exclusion accounting; the schedule guarantees
        only one button is held at a time."""
        if not self.plan_success:
            return
        btn = self.buttons[btn_idx]
        # approach the button top-down (contact point 0 is the top-down grasp frame of a box)
        self.move(self.grasp_actor(btn, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.09,
                                   contact_point_id=0, gripper_pos=0.5))
        if not self.plan_success:
            return
        # register the press (mutual-exclusion bookkeeping) and arm the belt's trapdoor
        self._buttons_held.add(btn_idx)
        self._belt_armed[self.button_belt(btn_idx)] = True
        # push down to actuate
        self.move(self.move_by_displacement(arm_tag, z=-0.03))
        # hold the press while the trapdoor stays armed (dwell records frames)
        self._dwell(self.press_hold_steps)
        # release: lift and clear the press
        self._belt_armed[self.button_belt(btn_idx)] = False
        self._buttons_held.discard(btn_idx)
        self.move(self.move_by_displacement(arm_tag, z=0.04))

    def button_belt(self, btn_idx):
        return btn_idx  # one button per belt, same index

    def _dwell(self, steps):
        """Advance physics for `steps` while recording frames periodically (no arm motion)."""
        for i in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _wait_marble_near_trapdoor(self, belt_idx, max_steps=None):
        """Advance the belts (no arm motion) until the NEXT undropped marble on `belt_idx`
        is within the drop window of the trapdoor. Returns when ready (or budget exhausted)."""
        if max_steps is None:
            max_steps = self.marble_dwell_steps
        for i in range(int(max_steps)):
            # is there a belt-state marble on this belt already in the window?
            ready = False
            for idx, bi in enumerate(self.marble_belt):
                if bi == belt_idx and self.marble_state[idx] == "belt":
                    y = self.marbles[idx].get_pose().p[1]
                    if abs(y - self.trapdoor_y) <= self.drop_window * 0.6:
                        ready = True
                        break
                    # also stop if a marble has already reached the very end (progress saturated)
                    if self.marble_progress[idx] >= 0.999:
                        ready = True
                        break
            if ready:
                return True
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        return False

    # ------------------------------------------------------------- policy
    def play_once(self):
        # left arm owns belts/buttons 0,1 (x<0); right arm owns 2,3 (x>0)
        left_buttons = [i for i, bx in enumerate(self.belt_x) if bx < 0]
        right_buttons = [i for i, bx in enumerate(self.belt_x) if bx >= 0]

        # Build a single GLOBAL, non-overlapping press schedule. We interleave one marble at a
        # time across belts so the arms never press two buttons simultaneously. Each schedule
        # entry is (arm_tag, button_idx); we wait for that belt's next marble to reach the
        # trapdoor, then perform a single isolated press (mutual exclusion is structurally
        # guaranteed by doing one press at a time).
        schedule = []
        for m in range(self.marbles_per_belt):
            for bi in left_buttons:
                schedule.append((ArmTag("left"), bi))
            for bi in right_buttons:
                schedule.append((ArmTag("right"), bi))

        import os, sys
        dbg = os.environ.get("CMT_DEBUG")
        for si, (arm_tag, btn_idx) in enumerate(schedule):
            if not self.plan_success:
                if dbg:
                    print(f"[cmt] plan_success False before step {si}", file=sys.stderr, flush=True)
                break
            belt_idx = self.button_belt(btn_idx)
            # advance belts until this belt's next marble is over the trapdoor
            self._wait_marble_near_trapdoor(belt_idx)
            # single isolated press (arms the trapdoor; drop fires inside _update_kinematic_tasks)
            self._press_button(arm_tag, btn_idx)
            if dbg:
                print(f"[cmt] step {si} arm={arm_tag} btn={btn_idx} dropped={self._dropped_count} "
                      f"plan={self.plan_success}", file=sys.stderr, flush=True)
            # short settle so the dropped marble registers as landed
            self._dwell(40)

        # final settle for any in-flight marbles
        self._dwell(120)

        self.info["info"] = {
            "{A}": "marbles",
            "{B}": "collection box",
            "{a}": "left arm",
            "{b}": "right arm",
        }
        return self.info

    # ----------------------------------------------------------- metric/obs
    def _marble_in_box(self, idx):
        p = np.array(self.marbles[idx].get_pose().p)
        in_x = abs(p[0] - self.box_center[0]) <= self.box_half_w
        in_y = abs(p[1] - self.box_center[1]) <= self.box_half_d
        in_z = p[2] <= self.box_floor_z + self.box_wall_h + 0.02
        return bool(in_x and in_y and in_z)

    def compute_catch_metric(self):
        """Per marble: landing offset from box centre -> catch_score = mean clamp(1 - off/halfw)."""
        scores = []
        count_in = 0
        for idx in range(len(self.marbles)):
            if self._marble_in_box(idx):
                count_in += 1
                p = np.array(self.marbles[idx].get_pose().p[:2])
                off = float(np.linalg.norm(p - self.box_center))
                scores.append(float(np.clip(1.0 - off / max(1e-6, self.box_half_w), 0.0, 1.0)))
            else:
                scores.append(0.0)
        catch_score = float(np.mean(scores)) if scores else 0.0
        return {
            "catch_score": catch_score,
            "count_in_box": int(count_in),
            "n_marbles": int(len(self.marbles)),
            "mutex_violation": bool(self._mutex_violation),
        }

    def check_success(self):
        # A mutual-exclusion violation VOIDS the episode regardless of catches.
        if self._mutex_violation:
            return False
        m = self.compute_catch_metric()
        # success: a majority of marbles caught in the box (permissive about exact landing spot)
        return bool(m["count_in_box"] >= max(1, (len(self.marbles) + 1) // 2))

    def get_obs(self):
        obs = super().get_obs()
        marble_positions = [list(map(float, e.get_pose().p)) for e in self.marbles]
        m = self.compute_catch_metric()
        obs["catch_marbles"] = {
            "marble_positions": marble_positions,
            "marble_state": list(self.marble_state),
            "marble_belt": list(self.marble_belt),
            "belt_profile": list(self.belt_profile),
            "buttons_held": sorted(self._buttons_held),
            "mutex_violation": bool(self._mutex_violation),
            "catch_score": m["catch_score"],
            "count_in_box": m["count_in_box"],
        }
        return obs
