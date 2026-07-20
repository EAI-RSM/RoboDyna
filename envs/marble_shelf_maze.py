from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np
import transforms3d as t3d


class marble_shelf_maze(Base_Task):
    """A marble threads down a zig-zag stack of shelves hung between two parallel glass sheets.

    Two thin vertical "glass" panes face the robot, separated by a narrow gap. Four short shelves
    are wedged crosswise in that gap at four heights, each shelf shifted left/right of the one above
    it (overlapping by at most half its length), like the plates of a bagatelle board. A marble
    starts at rest on the centre of the top shelf. Two buttons sit on the table -- one on the left,
    one on the right. Pressing a button tilts whichever shelf currently holds the marble (the
    "active" shelf) 45 degrees toward that side; the marble rolls to the tilted-down edge and drops
    onto the shelf below, coming to rest there (which becomes the new active shelf). A bowl sits on
    the table under the left or right end of the bottom shelf. The robot must press the buttons in
    the sequence that walks the marble down through all four shelves and off the side of the bottom
    shelf that matches the bowl.

    Only the two shelf-offset directions (left/right at each level) determine which button press
    keeps the marble descending instead of missing the shelf below; the geometry is randomized every
    episode (alternating zig-zag, magnitude drawn so the overlap is at most half a shelf length), and
    the bowl side is randomized independently. The marble transitions are real PhysX dynamics (roll
    off the tilted shelf under gravity, land on the next shelf or in the bowl); only the tilt/reset
    animation of the (currently non-active) shelves and the marble's slide-to-the-edge while its own
    shelf is tilting are kinematically scripted, driven by step counts for two-pass determinism.

    The stack depth itself is a source of task variation: `n_shelves` (2-6) sets how many shelves
    are built, and can either be fixed or drawn per-episode from a [n_shelves_min, n_shelves_max]
    range. Since the bowl always sits under the *bottom* shelf's edge, changing the depth changes
    both the length and the left/right pattern of the button-press combination the robot must find
    -- deeper stacks need longer combinations, shallower ones shorter -- giving the "bowl placed in
    different places" variation without needing per-level bowl placement.
    """

    N_SHELVES_DEFAULT = 4
    N_SHELVES_HARD_MAX = 6               # absolute cap -- keeps the stack within the cameras' FOV
    SHELF_LENGTH_DEFAULT = 0.15          # x-length of each shelf ("x" in the task description)
    GLASS_GAP_DEFAULT = 0.05             # gap between the two glass sheets
    SHELF_HALF_DEPTH_DEFAULT = 0.0125    # shelf depth (y, into the glass gap)
    SHELF_HALF_THICK_DEFAULT = 0.00625   # shelf thickness (z, vertical) -- half of the previous value
    LEVEL_GAP_DEFAULT = 0.055            # vertical spacing between shelf centers (tuned so the
                                          # whole 4-shelf stack, table to top, stays inside the
                                          # head camera's vertical FOV)
    OFFSET_MIN_DEFAULT = 0.085           # min |x-shift| between consecutive shelves (keeps overlap <= half length)
    OFFSET_MAX_DEFAULT = 0.11            # max |x-shift| (keeps a comfortable catching overlap > 0)
    BOTTOM_CLEARANCE_DEFAULT = 0.15      # vertical gap from bottom shelf down to the table (raised ~10cm
                                          # so the whole stack sits higher above the table)

    BALL_RADIUS_DEFAULT = 0.012
    ROLL_OFF_SPEED_DEFAULT = 0.06        # m/s -- small horizontal speed imparted when a marble leaves a shelf edge
    NATURAL_BALL_STOP_DEFAULT = False    # if True, a landed marble is left to settle under its own real
                                          # friction/damping instead of being snapped to the shelf's centre

    TILT_ANGLE_DEG_DEFAULT = 45.0
    TILT_DURATION_SEC_DEFAULT = 0.5      # seconds to sweep from flat to full tilt (and to reset back) --
                                          # used as-is unless smooth_tilt is enabled (see below)
    SMOOTH_TILT_DEFAULT = False          # if True, tilt speed (not a fixed duration) drives the sweep,
                                          # so holding the button noticeably longer produces the same 45 deg max
    TILT_SPEED_DEG_PER_SEC_DEFAULT = 30.0  # deg/s used for the sweep when smooth_tilt is enabled
    FALL_SETTLE_STEPS_DEFAULT = 160      # physics steps to let the marble fall+settle onto the next shelf
    FINAL_FALL_SETTLE_STEPS_DEFAULT = 220  # physics steps for the last drop into the bowl (falls further)

    BUTTON_X_DEFAULT = [-0.15, 0.15]
    BUTTON_Y_DEFAULT = -0.18
    BUTTON_HALF_DEFAULT = [0.022, 0.022, 0.018]
    BUTTON_PRESS_DEPTH_DEFAULT = 0.03
    PRESS_HOLD_STEPS_DEFAULT = 20
    POST_PRESS_DWELL_DEFAULT = 10

    BOWL_ID_DEFAULT = 1
    BOWL_SCALE_MULT_DEFAULT = 0.65
    BOWL_CATCH_RADIUS_DEFAULT = 0.032     # horizontal capture tolerance around the bowl center
    BOWL_HEIGHT_DEFAULT = 0.045

    GRAVITY = 9.81

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("marble_shelf_maze", {})
        # The collector reuses this env across episodes; _init_task_env_ runs load_camera (which
        # calls _update_kinematic_tasks) BEFORE the new load_actors rebuilds the scene. Clear all
        # per-episode state here so the guard at the top of _update_kinematic_tasks blocks until
        # load_actors has run again.
        self._loaded = False
        self.shelves = []
        self.shelf_centers_x = []
        self.shelf_z = []
        self.shelf_half_len = 0.0
        self.shelf_half_thick = 0.0
        self.shelf_half_depth = 0.0
        self._shelf_cur_angle = []
        self._shelf_target_angle = []
        self.ball = None
        self._ball_rigid = None
        self.active_shelf_idx = 0
        self._ball_mode = "resting"      # resting | sliding | falling | done
        self._sliding_shelf_idx = -1
        self._sliding_dir = 0
        self._sliding_start_local_x = 0.0
        self.bowl = None
        self.bowl_side = "left"
        self.correct_dir = []
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        # n_shelves can be fixed (n_shelves: k) or randomized per-episode from a range
        # (n_shelves_min/n_shelves_max) -- varying how many shelves the marble has to descend
        # before reaching the bowl changes both the length and the direction pattern of the
        # required button-press combination from one episode to the next. Capped at
        # N_SHELVES_HARD_MAX (6) so the stack always fits inside the cameras' FOV.
        n_min = c.get("n_shelves_min", None)
        n_max = c.get("n_shelves_max", None)
        if n_min is not None and n_max is not None:
            n_min = max(2, int(n_min))
            n_max = min(self.N_SHELVES_HARD_MAX, int(n_max))
            n_max = max(n_min, n_max)
            self.n_shelves = int(np.random.randint(n_min, n_max + 1))
        else:
            self.n_shelves = min(self.N_SHELVES_HARD_MAX, max(2, int(c.get("n_shelves", self.N_SHELVES_DEFAULT))))
        self.shelf_length = float(c.get("shelf_length", self.SHELF_LENGTH_DEFAULT))
        self.glass_gap = float(c.get("glass_gap", self.GLASS_GAP_DEFAULT))
        # level_gap itself stays fixed (it's tuned so the marble reliably clears the fall between
        # consecutive shelves -- shrinking it to cram more shelves into the same height breaks that).
        # Instead, the *top* of the stack is pinned to the height it would be at with
        # stack_height_ref_n_shelves shelves, by adjusting bottom_clearance: fewer shelves leaves
        # more air between the bottom shelf and the table, more shelves leaves less (down to
        # min_bottom_clearance) -- either way the top of the stack, and so the cameras' fixed FOV
        # (tuned for the reference stack), stays framed the same regardless of n_shelves.
        self.level_gap = float(c.get("level_gap", self.LEVEL_GAP_DEFAULT))
        bottom_clearance_cfg = float(c.get("bottom_clearance", self.BOTTOM_CLEARANCE_DEFAULT))
        stack_ref_n = int(c.get("stack_height_ref_n_shelves", self.N_SHELVES_DEFAULT))
        min_bottom_clearance = float(c.get("min_bottom_clearance", 0.04))
        total_span_ref = bottom_clearance_cfg + max(stack_ref_n - 1, 0) * self.level_gap
        self.bottom_clearance = max(min_bottom_clearance, total_span_ref - (self.n_shelves - 1) * self.level_gap)
        self.offset_min = float(c.get("offset_min", self.OFFSET_MIN_DEFAULT))
        self.offset_max = float(c.get("offset_max", self.OFFSET_MAX_DEFAULT))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.roll_off_speed = float(c.get("roll_off_speed", self.ROLL_OFF_SPEED_DEFAULT))
        self.natural_ball_stop = bool(c.get("natural_ball_stop", self.NATURAL_BALL_STOP_DEFAULT))

        self.tilt_angle_deg = float(c.get("tilt_angle_deg", self.TILT_ANGLE_DEG_DEFAULT))
        self.smooth_tilt = bool(c.get("smooth_tilt", self.SMOOTH_TILT_DEFAULT))
        self.tilt_speed_deg_per_sec = float(c.get("tilt_speed_deg_per_sec", self.TILT_SPEED_DEG_PER_SEC_DEFAULT))
        if self.smooth_tilt:
            # Speed-driven sweep: the button must stay held roughly angle/speed seconds for the
            # shelf to reach the full 45 deg (the caller already holds throughout the tilt+settle
            # dwell, so slowing the speed automatically makes the press "hold longer").
            self.tilt_duration_sec = self.tilt_angle_deg / max(self.tilt_speed_deg_per_sec, 1e-6)
        else:
            self.tilt_duration_sec = float(c.get("tilt_duration_sec", self.TILT_DURATION_SEC_DEFAULT))
        self.fall_settle_steps = int(c.get("fall_settle_steps", self.FALL_SETTLE_STEPS_DEFAULT))
        self.final_fall_settle_steps = int(c.get("final_fall_settle_steps", self.FINAL_FALL_SETTLE_STEPS_DEFAULT))

        self.button_x = list(c.get("button_x", self.BUTTON_X_DEFAULT))
        self.button_y = float(c.get("button_y", self.BUTTON_Y_DEFAULT))
        self.button_half = list(c.get("button_half", self.BUTTON_HALF_DEFAULT))
        self.button_press_depth = float(c.get("button_press_depth", self.BUTTON_PRESS_DEPTH_DEFAULT))
        self.press_hold_steps = int(c.get("press_hold_steps", self.PRESS_HOLD_STEPS_DEFAULT))
        self.post_press_dwell = int(c.get("post_press_dwell", self.POST_PRESS_DWELL_DEFAULT))

        self.bowl_id = int(c.get("bowl_id", self.BOWL_ID_DEFAULT))
        self.bowl_scale_mult = float(c.get("bowl_scale_mult", self.BOWL_SCALE_MULT_DEFAULT))
        self.bowl_catch_radius = float(c.get("bowl_catch_radius", self.BOWL_CATCH_RADIUS_DEFAULT))
        self.bowl_height = float(c.get("bowl_height", self.BOWL_HEIGHT_DEFAULT))

        self.table_z = 0.74 + self.table_z_bias
        self.maze_y = 0.0

        self.shelf_half_len = self.shelf_length / 2.0
        self.shelf_half_thick = float(c.get("shelf_half_thick", self.SHELF_HALF_THICK_DEFAULT))    # z (vertical)
        self.shelf_half_depth = float(c.get("shelf_half_depth", self.SHELF_HALF_DEPTH_DEFAULT))    # y (into the gap)

        # ---- randomize the zig-zag: alternating left/right offsets between consecutive shelves ----
        s0 = float(np.random.choice([-1.0, 1.0]))
        signs = [s0 * ((-1.0) ** i) for i in range(self.n_shelves - 1)]
        offsets = [
            float(sign * np.random.uniform(self.offset_min, self.offset_max)) for sign in signs
        ]
        centers = [0.0]
        for off in offsets:
            centers.append(centers[-1] + off)
        self.shelf_centers_x = centers
        # the correct press direction at level i (0..n-2) is simply which way shelf i+1 sits
        self.correct_dir = ["right" if off > 0 else "left" for off in offsets]

        top_z = self.table_z + self.bottom_clearance + (self.n_shelves - 1) * self.level_gap
        self.shelf_z = [top_z - i * self.level_gap for i in range(self.n_shelves)]

        self.shelves = []
        self._shelf_cur_angle = [0.0] * self.n_shelves
        self._shelf_target_angle = [0.0] * self.n_shelves
        shelf_color = [0.62, 0.46, 0.30]
        for i in range(self.n_shelves):
            shelf = create_box(
                self.scene,
                pose=sapien.Pose([self.shelf_centers_x[i], self.maze_y, self.shelf_z[i]]),
                half_size=[self.shelf_half_len, self.shelf_half_depth, self.shelf_half_thick],
                color=shelf_color,
                is_static=False,
                name=f"maze_shelf_{i}",
            )
            rigid = self._get_rigid(shelf)
            if rigid is not None:
                try:
                    rigid.set_disable_gravity(True)
                    rigid.set_kinematic(True)
                except Exception:
                    pass
            self.shelves.append(shelf)

        # ---- two thin "glass" sheets flanking the shelf stack (static scenery) ----
        min_x = min(self.shelf_centers_x) - self.shelf_half_len - 0.03
        max_x = max(self.shelf_centers_x) + self.shelf_half_len + 0.03
        glass_half_w = (max_x - min_x) / 2.0
        glass_cx = (max_x + min_x) / 2.0
        glass_bottom_z = self.table_z + 0.01
        glass_top_z = top_z + self.shelf_half_thick + 0.05
        glass_half_h = (glass_top_z - glass_bottom_z) / 2.0
        glass_cz = (glass_top_z + glass_bottom_z) / 2.0
        # Both sheets are drawn as thin rectangular frames (four edges only, no filled center) so
        # the camera can see straight through into the maze from either side -- true render
        # transparency (RenderMaterial transmission/alpha) isn't respected by this renderer, so
        # "glass" has to be conveyed geometrically (an open frame) rather than via a translucent
        # material.
        glass_thick = 0.004
        near_dy = -self.glass_gap / 2.0 - glass_thick / 2.0
        far_dy = self.glass_gap / 2.0 + glass_thick / 2.0
        for side, dy in (("near", near_dy), ("far", far_dy)):
            self._build_glass_frame(
                [glass_cx, self.maze_y + dy, glass_cz],
                half_w=glass_half_w, half_h=glass_half_h, half_thick=glass_thick / 2.0,
                bar_half=0.006, side=side,
            )

        # ---- marble: starts frozen (kinematic, no gravity) at the centre of the top shelf ----
        ball_z0 = self.shelf_z[0] + self.shelf_half_thick + self.ball_radius + 0.001
        self.ball = create_sphere(
            self.scene,
            pose=sapien.Pose([self.shelf_centers_x[0], self.maze_y, ball_z0]),
            radius=self.ball_radius,
            color=[0.85, 0.15, 0.15],
            is_static=False,
            name="maze_marble",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        if self._ball_rigid is not None:
            try:
                if self.natural_ball_stop:
                    # Real dynamics from the start -- it just sits at rest under gravity/friction
                    # on the (flat) top shelf rather than being kinematically frozen.
                    self._ball_rigid.set_disable_gravity(False)
                    self._ball_rigid.set_kinematic(False)
                else:
                    self._ball_rigid.set_disable_gravity(True)
                    self._ball_rigid.set_kinematic(True)
                inelastic = sapien.physx.PhysxMaterial(
                    static_friction=0.9, dynamic_friction=0.9, restitution=0.0,
                )
                for shape in self._ball_rigid.get_collision_shapes():
                    shape.set_physical_material(inelastic)
            except Exception:
                pass
        self.active_shelf_idx = 0
        self._ball_mode = "resting"
        self._sliding_shelf_idx = -1
        self._sliding_dir = 0

        # ---- two buttons in the near zone, one per side/arm ----
        self.buttons = []
        for bx in self.button_x:
            bz = self.table_z + self.button_half[2]
            is_right = bx > 0
            btn = create_box(
                self.scene,
                pose=sapien.Pose([bx, self.button_y, bz]),
                half_size=list(self.button_half),
                color=[0.75, 0.20, 0.20] if bx < 0 else [0.20, 0.55, 0.75],
                is_static=True,
                name=f"shelf_button_{'left' if bx < 0 else 'right'}",
            )
            self.buttons.append(btn)
            self.add_prohibit_area(btn, padding=0.03)
            # Decal on the button's top face: a curved arrow spelling out which way the active
            # shelf will tilt if this button is pressed (clockwise for right, counter-clockwise
            # for left -- matches _set_shelf_pose's +angle-tilts-right-edge-down convention).
            self._build_turn_arrow(
                [bx, self.button_y, bz + self.button_half[2]],
                radius=min(self.button_half[0], self.button_half[1]) * 0.65,
                clockwise=is_right,
                color=[0.96, 0.96, 0.96],
            )
        self.left_button, self.right_button = self.buttons

        # ---- bowl: placed under the left or right end of the bottom shelf ----
        self.bowl_side = str(np.random.choice(["left", "right"]))
        bottom_center_x = self.shelf_centers_x[-1]
        bowl_x = bottom_center_x + (self.shelf_half_len if self.bowl_side == "right" else -self.shelf_half_len)
        bowl_pose = sapien.Pose([bowl_x, self.maze_y, self.table_z], [0.5, 0.5, 0.5, 0.5])
        self.bowl = create_actor(
            self.scene, pose=bowl_pose, modelname="002_bowl", model_id=self.bowl_id,
            convex=True, is_static=True, scale_mult=self.bowl_scale_mult,
        )
        self.bowl_center_xy = np.array([bowl_x, self.maze_y], dtype=np.float64)

        self._loaded = True

    GLASS_COLOR = [0.80, 0.90, 0.93]

    def _build_glass_frame(self, center_xyz, half_w, half_h, half_thick, bar_half, side):
        """A visual-only (no collision) rectangular frame -- four thin edge bars, open in the
        middle -- standing in for a pane of glass. Used for both sheets (near and far) so the
        camera can see straight through the whole maze from any viewpoint, not just the one it
        happened to be tuned for."""
        cx, cy, cz = center_xyz
        bars = [
            ("top", [cx, cy, cz + half_h - bar_half], [half_w, half_thick, bar_half]),
            ("bottom", [cx, cy, cz - half_h + bar_half], [half_w, half_thick, bar_half]),
            ("left", [cx - half_w + bar_half, cy, cz], [bar_half, half_thick, half_h]),
            ("right", [cx + half_w - bar_half, cy, cz], [bar_half, half_thick, half_h]),
        ]
        for name, pos, half_size in bars:
            create_visual_box(self.scene, sapien.Pose(pos), half_size, color=self.GLASS_COLOR,
                               name=f"glass_sheet_{side}_{name}")

    def _build_turn_arrow(self, top_center_xyz, radius, clockwise, color):
        """A small curved-arrow decal lying flat on a button's top face (the x/y plane), made of
        thin visual-only box segments along an arc plus a chevron arrowhead at the leading end --
        clockwise=True draws a clockwise-reading arrow (right button), False a counter-clockwise
        one (left button), matching which way pressing that button tilts the active shelf."""
        cx, cy, cz = top_center_xyz
        cz += 0.0022  # sit just above the button's top face, clear of z-fighting

        n_pts = 9
        angles_deg = np.linspace(-140.0, 140.0, n_pts)
        if clockwise:
            angles_deg = angles_deg[::-1]
        pts = [np.array([radius * np.cos(np.deg2rad(a)), radius * np.sin(np.deg2rad(a))]) for a in angles_deg]

        seg_half_thick = radius * 0.11
        seg_half_h = 0.0009

        def _place_segment(p0, p1, half_thick):
            mid = (p0 + p1) / 2.0
            seg_len = float(np.linalg.norm(p1 - p0))
            if seg_len < 1e-6:
                return
            heading = float(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            quat = t3d.quaternions.axangle2quat([0.0, 0.0, 1.0], heading)
            create_visual_box(
                self.scene, sapien.Pose([cx + mid[0], cy + mid[1], cz], quat),
                half_size=[seg_len / 2.0 + seg_half_thick, half_thick, seg_half_h],
                color=color, name="turn_arrow_seg",
            )

        for p0, p1 in zip(pts[:-1], pts[1:]):
            _place_segment(p0, p1, seg_half_thick)

        tangent = pts[-1] - pts[-2]
        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        normal = np.array([-tangent[1], tangent[0]])
        tip = pts[-1] + tangent * radius * 0.30
        head_len = radius * 0.55
        back = tip - tangent * head_len
        for side in (1.0, -1.0):
            wing = back + normal * side * head_len * 0.62
            _place_segment(wing, tip, seg_half_thick * 1.3)

    def _get_rigid(self, entity):
        base_entity = entity.actor if hasattr(entity, "actor") else entity
        for component in base_entity.get_components():
            if isinstance(component, sapien.physx.PhysxRigidDynamicComponent):
                return component
        return None

    # --------------------------------------------------------- kinematic scene motion
    def _set_shelf_pose(self, idx: int, angle_deg: float):
        """Rotate shelf `idx` about its own centre (its y axis) so the +x (right) end dips when
        angle_deg > 0 and the -x (left) end dips when angle_deg < 0."""
        cx = self.shelf_centers_x[idx]
        cz = self.shelf_z[idx]
        phi = np.deg2rad(angle_deg)      # sign convention: +angle => right edge goes down
        quat = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], phi)
        self.shelves[idx].actor.set_pose(sapien.Pose([cx, self.maze_y, cz], quat))
        self._shelf_cur_angle[idx] = float(angle_deg)

    def _shelf_ball_local_point(self, idx: int, angle_deg: float, dir_sign: float, start_local_x: float = 0.0):
        """World position of the marble while it rides shelf `idx`'s top surface as the shelf tilts
        toward dir_sign (+1 right, -1 left): slides from wherever it started (start_local_x, the
        shelf's centre unless natural_ball_stop left it off-centre) toward the down-hill edge in
        lock-step with the tilt fraction, staying ball_radius above the (rotating) surface."""
        frac = float(np.clip(abs(angle_deg) / max(self.tilt_angle_deg, 1e-6), 0.0, 1.0))
        edge_x = dir_sign * self.shelf_half_len
        local_x = start_local_x + (edge_x - start_local_x) * frac
        local_z = self.shelf_half_thick + self.ball_radius
        phi = -np.deg2rad(angle_deg)
        cphi, sphi = np.cos(phi), np.sin(phi)
        wx = local_x * cphi - local_z * sphi
        wz = local_x * sphi + local_z * cphi
        cx = self.shelf_centers_x[idx]
        cz = self.shelf_z[idx]
        return np.array([cx + wx, self.maze_y, cz + wz], dtype=np.float64)

    def _advance_shelf_tilts(self):
        dt = float(self.scene.get_timestep())
        speed = abs(self.tilt_angle_deg) / max(self.tilt_duration_sec, 1e-6)
        step = speed * dt
        for idx in range(self.n_shelves):
            cur = self._shelf_cur_angle[idx]
            tgt = self._shelf_target_angle[idx]
            if abs(cur - tgt) <= 1e-3:
                continue
            if cur < tgt:
                cur = min(cur + step, tgt)
            else:
                cur = max(cur - step, tgt)
            self._set_shelf_pose(idx, cur)
            if idx == self._sliding_shelf_idx and self._ball_mode == "sliding":
                p = self._shelf_ball_local_point(idx, cur, self._sliding_dir, self._sliding_start_local_x)
                if self._ball_rigid is not None:
                    try:
                        self._ball_rigid.set_kinematic_target(sapien.Pose(p.tolist()))
                    except Exception:
                        self.ball.set_pose(sapien.Pose(p.tolist()))
                else:
                    self.ball.set_pose(sapien.Pose(p.tolist()))
                if abs(cur) >= self.tilt_angle_deg - 1e-3:
                    self._release_ball(idx, self._sliding_dir)

    def _release_ball(self, idx: int, dir_sign: float):
        if self._ball_rigid is None or self._ball_mode != "sliding":
            return
        p = self._shelf_ball_local_point(idx, self._sliding_dir * self.tilt_angle_deg, dir_sign,
                                         self._sliding_start_local_x)
        try:
            self.ball.set_pose(sapien.Pose(p.tolist()))
            self._ball_rigid.set_kinematic(False)
            self._ball_rigid.set_disable_gravity(False)
            self._ball_rigid.set_linear_velocity([dir_sign * abs(self.roll_off_speed), 0.0, 0.0])
            self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
            self._ball_rigid.set_linear_damping(0.05)
            self._ball_rigid.set_angular_damping(0.5)
        except Exception:
            pass
        self._ball_mode = "falling"
        self._shelf_target_angle[idx] = 0.0   # ease the fired shelf back to flat once it's done its job

    def _freeze_ball_on_shelf(self, idx: int):
        if self._ball_rigid is None:
            return
        if self.natural_ball_stop:
            # Leave the marble exactly where its own physics (gravity + the friction material set
            # on release) already settled it during the fall/settle dwell -- no re-centering snap,
            # it just naturally stopped rolling wherever friction took it.
            try:
                self._ball_rigid.set_disable_gravity(False)
                self._ball_rigid.set_kinematic(False)
            except Exception:
                pass
        else:
            p = np.array([self.shelf_centers_x[idx], self.maze_y,
                          self.shelf_z[idx] + self.shelf_half_thick + self.ball_radius + 0.001])
            try:
                self.ball.set_pose(sapien.Pose(p.tolist()))
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self.active_shelf_idx = idx
        self._ball_mode = "resting"

    def _ball_in_bowl(self) -> bool:
        if self.ball is None:
            return False
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        horiz = float(np.linalg.norm(p[:2] - self.bowl_center_xy))
        in_z = (self.table_z - 0.01) <= p[2] <= (self.table_z + self.bowl_height)
        return bool(horiz <= (self.bowl_catch_radius + self.ball_radius) and in_z)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._advance_shelf_tilts()

    # --------------------------------------------------------- press / dwell helpers
    def _dwell(self, steps: int):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _tilt_active_shelf_and_wait(self, direction: str):
        """Kick off the active shelf's tilt toward `direction` (rolling the marble off it), then
        dwell fixed step counts (deterministic across the plan/render passes) for the fall+settle."""
        idx = self.active_shelf_idx
        dir_sign = 1.0 if direction == "right" else -1.0

        # Snapshot wherever the marble actually is right now (its centre, unless natural_ball_stop
        # left it settled off-centre) and lock it kinematic -- from here the slide-to-the-edge is
        # always a deterministic, scripted sweep in lock-step with the tilt.
        cur_p = self.ball.get_pose().p
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_linear_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_angular_velocity([0.0, 0.0, 0.0])
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
            except Exception:
                pass
        self._sliding_start_local_x = float(cur_p[0]) - self.shelf_centers_x[idx]

        self._sliding_shelf_idx = idx
        self._sliding_dir = dir_sign
        self._ball_mode = "sliding"
        self._shelf_target_angle[idx] = dir_sign * self.tilt_angle_deg

        tilt_steps = int(np.ceil(self.tilt_duration_sec * self.scene.get_timestep() ** -1)) + 2
        self._dwell(tilt_steps)   # sweeps the shelf to full tilt and releases the marble mid-loop
        is_last = idx >= self.n_shelves - 1
        settle_steps = self.final_fall_settle_steps if is_last else self.fall_settle_steps
        self._dwell(settle_steps)

        if not is_last and self._ball_mode == "falling":
            landed_idx = self._locate_landed_shelf(idx)
            if landed_idx is not None:
                self._freeze_ball_on_shelf(landed_idx)
            else:
                self._ball_mode = "done"
                self.active_shelf_idx = -1
        elif self._ball_mode == "falling":
            self._ball_mode = "done"
            self.active_shelf_idx = -1

    def _locate_landed_shelf(self, from_idx: int):
        """After a fall, find which shelf below `from_idx` the marble is now resting on (x within
        its span, z near its top surface), by checking shelves top-down starting right below."""
        if self.ball is None:
            return None
        p = np.array(self.ball.get_pose().p, dtype=np.float64)
        for j in range(from_idx + 1, self.n_shelves):
            cx = self.shelf_centers_x[j]
            top_z = self.shelf_z[j] + self.shelf_half_thick
            in_x = abs(p[0] - cx) <= self.shelf_half_len
            in_z = abs(p[2] - (top_z + self.ball_radius)) <= max(0.02, 3.0 * self.ball_radius)
            if in_x and in_z:
                return j
        return None

    def _press_button(self, arm_tag: ArmTag, direction: str):
        if not self.plan_success or self.active_shelf_idx < 0:
            return
        btn = self.right_button if direction == "right" else self.left_button
        self.move(
            self.grasp_actor(
                btn, arm_tag=arm_tag, pre_grasp_dis=0.09, grasp_dis=0.09,
                contact_point_id=0, gripper_pos=0.5,
            )
        )
        if not self.plan_success:
            return
        self.move(self.move_by_displacement(arm_tag, z=-self.button_press_depth))
        self._tilt_active_shelf_and_wait(direction)
        self._dwell(self.press_hold_steps)
        self.move(self.move_by_displacement(arm_tag, z=self.button_press_depth + 0.01))
        self.move(self.back_to_origin(arm_tag))
        self._dwell(self.post_press_dwell)

    # ------------------------------------------------------------- policy
    def play_once(self):
        presses_made = []
        for level in range(self.n_shelves):
            if self.active_shelf_idx < 0 or not self.plan_success:
                break
            if level < self.n_shelves - 1:
                direction = self.correct_dir[level]
            else:
                direction = self.bowl_side
            arm_tag = ArmTag("right" if direction == "right" else "left")
            self._press_button(arm_tag, direction)
            presses_made.append(direction)

        self.info["info"] = {
            "{A}": "marble",
            "{B}": "shelves",
            "{C}": "bowl",
            "{D}": "buttons",
            "{a}": "left arm",
            "{b}": "right arm",
        }
        self.info["presses_made"] = presses_made
        return self.info

    # ----------------------------------------------------------- metric/obs
    def check_success(self):
        in_bowl = self._ball_in_bowl()
        self.info["n_shelves"] = int(self.n_shelves)
        self.info["bowl_side"] = self.bowl_side
        self.info["correct_dir"] = list(self.correct_dir)
        self.info["active_shelf_idx"] = int(self.active_shelf_idx)
        self.info["ball_mode"] = str(self._ball_mode)
        self.info["ball_position"] = list(map(float, self.ball.get_pose().p)) if self.ball is not None else []
        self.info["ball_in_bowl"] = bool(in_bowl)
        return bool(in_bowl)

    def get_obs(self):
        obs = super().get_obs()
        obs["shelf_maze"] = {
            "n_shelves": int(self.n_shelves),
            "shelf_centers_x": list(map(float, self.shelf_centers_x)),
            "shelf_z": list(map(float, self.shelf_z)),
            "shelf_angles_deg": list(map(float, self._shelf_cur_angle)),
            "active_shelf_idx": int(self.active_shelf_idx),
            "ball_mode": str(self._ball_mode),
            "ball_position": list(map(float, self.ball.get_pose().p)) if self.ball is not None else [0.0, 0.0, 0.0],
            "bowl_side": str(self.bowl_side),
            "correct_dir": list(self.correct_dir),
            "ball_in_bowl": bool(self._ball_in_bowl()),
        }
        return obs
