from ._base_task import Base_Task
from .utils import *
from ._GLOBAL_CONFIGS import *
import sapien
import sapien.physx
import numpy as np
import os

_CSM_DEBUG = os.environ.get("CSM_DEBUG", "0") == "1"


class catch_shelf_marble(Base_Task):
    """A belt-mounted bowl catches a marble that cascades down four tilted, interleaved shelves.

    Four short shelves hang above a belt, stacked top to bottom, each shifted left/right of the
    one above it (overlapping by at most half a shelf length) so the layout zig-zags like a
    bagatelle board; each shelf's tilt (15-45 degrees, direction tied to which way the marble must
    exit to reach the shelf below it) is randomized independently every episode. A marble starts
    at rest on the centre of the top shelf and slides/falls down through all four shelves onto the
    belt below.

    Unlike a button-hop belt, the bowl here moves CONTINUOUSLY: a left action key (pressed by the
    left arm) slides the bowl left at a constant speed for as long as it is held, and a right
    action key (pressed by the right arm) slides it right; releasing either key freezes the bowl on
    the spot. The robot must hold the correct key long enough to place the bowl under the marble's
    predicted landing point on the belt before it arrives.

    The full descent (kinematic slide along each shelf's fixed tilt, then a short parabolic
    free-fall onto the next shelf or, from the bottom shelf, onto the belt) is precomputed
    analytically in `load_actors` from the randomized geometry, so both collector passes replay the
    identical marble path and the same `target_catch_x` the expert policy aims for.

    Two task_args knobs beyond the basics:
    - `tilt_min_deg`/`tilt_max_deg`: the randomized range (degrees) each shelf's tilt magnitude is
      drawn from independently every episode (direction is separately tied to the zig-zag offset;
      see `load_actors`). Defaults 15-45.
    - `reactive_marble` (bool, default False): if True, the marble is released the instant
      `play_once` starts (before the arm even begins moving) instead of waiting for the bowl's key
      to be pressed -- the robot has to react to an already-falling marble rather than
      pre-positioning the bowl and then dropping it on cue. Because the fixed close_gripper +
      reach-the-key + press-down sequence alone takes far longer than a descent run at the default
      `roll_speed`, this mode also switches the *effective* roll speed to `reactive_roll_speed`
      (much slower, so the marble's slide legs take long enough for the arm to still catch up;
      the physically-timed free-fall legs between shelves are unaffected).
    """

    N_SHELVES_DEFAULT = 4
    SHELF_LENGTH_DEFAULT = 0.20
    SHELF_DEPTH_DEFAULT = 0.10
    SHELF_THICK_DEFAULT = 0.016
    LEVEL_GAP_DEFAULT = 0.13
    OFFSET_MIN_FRAC_DEFAULT = 0.55        # min |offset| as a fraction of shelf_length -> overlap <= 45%
    OFFSET_MAX_FRAC_DEFAULT = 0.92        # max |offset| as a fraction of shelf_length -> overlap >= 8%
    TILT_MIN_DEG_DEFAULT = 15.0
    TILT_MAX_DEG_DEFAULT = 45.0
    BOTTOM_CLEARANCE_DEFAULT = 0.22       # belt surface up to the bottom shelf's underside
    STACK_SHIFT_RANGE_DEFAULT = 0.05      # random overall shift of the whole cascade along the belt

    BALL_RADIUS_DEFAULT = 0.014
    ROLL_SPEED_DEFAULT = 0.35             # m/s, constant scripted speed for both slide and fall legs
    MAX_FALL_STEPS_DEFAULT = 500          # safety cap (per leg) for the offline descent-plan search
    GRAVITY = 9.81

    REACTIVE_MARBLE_DEFAULT = False       # if True: release at play_once start, not on key-press
    REACTIVE_ROLL_SPEED_DEFAULT = 0.09    # m/s; slower slide speed used only when reactive_marble
                                           # is on, so the descent lasts long enough for the arm's
                                           # fixed reach/press sequence to still catch up in time

    BOWL_ID_DEFAULT = 1
    BOWL_RADIUS_DEFAULT = 0.05
    BOWL_SCALE_MULT_DEFAULT = 0.8
    BOWL_CATCH_XY_TOL_DEFAULT = 0.035
    BOWL_SPEED_DEFAULT = 0.18             # m/s, continuous bowl speed while a key is held
    BELT_THICKNESS_DEFAULT = 0.015
    BELT_MARGIN_DEFAULT = 0.10

    KEY_HALF_DEFAULT = [0.028, 0.028, 0.016]
    KEY_HOVER_DIS_DEFAULT = 0.06
    KEY_PRESS_DEPTH_DEFAULT = 0.055
    KEY_PRESS_XY_DEFAULT = 0.045
    KEY_PRESS_DZ_DEFAULT = 0.17
    KEY_TRAVEL_DEFAULT = 0.008
    KEY_SPRING_STEP_DEFAULT = 0.0015
    KEY_X_LEFT_DEFAULT = -0.26
    KEY_X_RIGHT_DEFAULT = 0.26
    KEY_Y_DEFAULT = -0.13
    EE_TO_TCP = 0.12

    PRESS_LOOP_TOL_DEFAULT = 0.006
    PRESS_LOOP_MAX_STEPS_DEFAULT = 500
    POST_CATCH_DWELL_DEFAULT = 20

    SHELF_COLOR = [0.62, 0.46, 0.30]
    BELT_COLOR = [0.10, 0.10, 0.12]
    KEY_BASE_COLOR = [0.28, 0.28, 0.31]
    LEFT_KEY_COLOR = [0.20, 0.70, 0.35]
    RIGHT_KEY_COLOR = [0.18, 0.48, 0.82]
    MARBLE_COLOR = [0.85, 0.15, 0.15]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_shelf_marble", {})
        # The collector reuses this env across episodes; _init_task_env_ runs load_camera (which
        # calls _update_kinematic_tasks) BEFORE the new load_actors rebuilds the scene, so every
        # per-episode state variable is cleared here and the _loaded guard blocks stale updates.
        self._loaded = False
        self.shelves = []
        self.shelf_centers_x = []
        self.shelf_z = []
        self.shelf_dir = []
        self.shelf_angle_deg = []
        self.shelf_half_len = 0.0
        self.shelf_half_thick = 0.0
        self.ball = None
        self.bowl = None
        self.bowl_q = [0.5, 0.5, 0.5, 0.5]
        self.belt = None
        self.descent_legs = []
        self.total_marble_steps = 0
        self.target_catch_x = 0.0
        self._marble_state = "parked"     # parked -> descending -> resolved
        self._marble_result = None        # None | "caught" | "missed"
        self._leg_idx = 0
        self._leg_step = 0
        self.key_xy = {}
        self.key_rest_xyz = {}
        self.key_arrows = {}
        self.keys = {}
        self._key_pressed = {"left": False, "right": False}
        self._key_depression = {"left": 0.0, "right": 0.0}
        self._bowl_force_stop = False
        self._bowl_drive_clamp = None
        super()._init_task_env_(**kwags)
        self._configure_observer_camera()

    def _configure_observer_camera(self):
        """Frame the whole belt + shelf stack from the table's upper-right corner (third-person
        overview), mirroring `dispense_gummy`'s convention. The shelf stack here is taller than
        that fixture, so the camera sits further back and higher."""
        camera = getattr(getattr(self, "cameras", None), "observer_camera", None)
        if camera is None:
            return
        camera_pos = np.array([0.38, 0.52, 1.45], dtype=np.float64)
        look_at = np.array([0.0, -0.05, 0.95], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        camera_matrix = np.eye(4)
        camera_matrix[:3, :3] = np.stack([forward, left, up], axis=1)
        camera_matrix[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(camera_matrix))

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
                return
            except Exception:
                pass
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    # -------------------------------------------------------- shelf geometry
    def _shelf_phi(self, idx):
        """`phi` fed into both the shelf's rendered/collision quaternion (`load_actors`) and the
        `_shelf_local_to_world`/`_shelf_surface_z_at_local` family below. No sign flip here: with
        `quat = [cos(phi/2), 0, sin(phi/2), 0]` (a standard +Y-axis rotation) and those helpers'
        `R_y(phi)` convention, `phi = +deg2rad(angle_deg)` is exactly what makes a positive
        `shelf_angle_deg` tip the +local_x (right) edge down, matching the convention the
        offset/tilt-direction coupling in `load_actors` (`shelf_angle_deg[i] = shelf_dir[i] *
        magnitude`) relies on."""
        return np.deg2rad(self.shelf_angle_deg[idx])

    def _shelf_local_to_world(self, idx, local_x):
        """World point on shelf `idx`'s (fixed, static) tilted top surface, `local_x` measured
        along the shelf's own long axis from its centre (matches the +x=>right-edge-down
        convention used throughout: positive angle tips the +local_x edge down).

        Must exactly match the rotation actually applied to the shelf's box actor -- the quaternion
        built from `phi` in `load_actors` (`[cos(phi/2), 0, sin(phi/2), 0]`) is a standard +Y-axis
        rotation, i.e. `world = R_y(phi) @ local` with `R_y(phi) = [[cphi,0,sphi],[0,1,0],
        [-sphi,0,cphi]]`. (A previous version of this formula used the mirror-image rotation
        `R_y(-phi)`, which put the marble's scripted position off the real tilted surface by an
        amount that grew with `|local_x|` -- visually, the marble drifted through the shelf mesh
        instead of riding its top face.)"""
        phi = self._shelf_phi(idx)
        cphi, sphi = np.cos(phi), np.sin(phi)
        local_z = self.shelf_half_thick + self.ball_radius
        wx = local_x * cphi + local_z * sphi
        wz = -local_x * sphi + local_z * cphi
        cx = self.shelf_centers_x[idx]
        cz = self.shelf_z[idx]
        return np.array([cx + wx, self.belt_y, cz + wz], dtype=np.float64)

    def _shelf_world_x_to_local(self, idx, world_x):
        """Inverse of `_shelf_local_to_world`'s x-mapping (same `R_y(phi)` convention)."""
        phi = self._shelf_phi(idx)
        cphi, sphi = np.cos(phi), np.sin(phi)
        if abs(cphi) < 1e-6:
            return None
        local_z = self.shelf_half_thick + self.ball_radius
        cx = self.shelf_centers_x[idx]
        return (world_x - cx - local_z * sphi) / cphi

    def _shelf_surface_z_at_local(self, idx, local_x):
        """Same `R_y(phi)` convention as `_shelf_local_to_world`'s z-mapping."""
        phi = self._shelf_phi(idx)
        cphi, sphi = np.cos(phi), np.sin(phi)
        local_z = self.shelf_half_thick + self.ball_radius
        cz = self.shelf_z[idx]
        return cz - local_x * sphi + local_z * cphi

    # -------------------------------------------------- offline descent plan
    def _compute_descent_plan(self):
        """Analytically precompute the marble's entire path (a list of deterministic kinematic
        "legs": slide-along-a-shelf, then parabolic-free-fall-to-the-next-shelf-or-the-belt) from
        the randomized shelf geometry. This is what makes the two collector passes replay
        bit-identically and gives the expert policy a `target_catch_x` to aim the bowl at before
        the marble ever starts moving."""
        dt = float(self.scene.get_timestep())
        g = self.GRAVITY
        legs = []
        cur_shelf = 0
        cur_local = 0.0
        x = z = 0.0
        for _ in range(self.n_shelves + 2):
            sign = self.shelf_dir[cur_shelf]
            edge_local = sign * self.shelf_half_len
            dist = abs(edge_local - cur_local)
            slide_steps = max(1, int(round((dist / max(self.roll_speed, 1e-4)) / dt)))
            legs.append({
                "type": "slide",
                "shelf": cur_shelf,
                "start_local": float(cur_local),
                "end_local": float(edge_local),
                "steps": int(slide_steps),
            })

            edge_pos = self._shelf_local_to_world(cur_shelf, edge_local)
            vx = sign * self.roll_speed
            landed_shelf, landed_local = None, None
            k = 0
            for k in range(1, self.max_fall_steps + 1):
                t = k * dt
                x = float(edge_pos[0] + vx * t)
                z = float(edge_pos[2] - 0.5 * g * t * t)
                if z <= self.belt_surface_z + self.ball_radius:
                    z = self.belt_surface_z + self.ball_radius
                    break
                hit = False
                for j in range(cur_shelf + 1, self.n_shelves):
                    lx = self._shelf_world_x_to_local(j, x)
                    if lx is None or abs(lx) > self.shelf_half_len:
                        continue
                    surf_z = self._shelf_surface_z_at_local(j, lx)
                    if z <= surf_z:
                        landed_shelf, landed_local = j, float(lx)
                        hit = True
                        break
                if hit:
                    break
            legs.append({
                "type": "fall",
                "start_pos": (float(edge_pos[0]), float(edge_pos[1]), float(edge_pos[2])),
                "vx": float(vx),
                "steps": int(k),
            })

            if landed_shelf is None:
                self.target_catch_x = float(x)
                break
            cur_shelf, cur_local = landed_shelf, landed_local
        else:
            self.target_catch_x = float(x)

        self.descent_legs = legs
        self.total_marble_steps = int(sum(leg["steps"] for leg in legs))

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.n_shelves = int(c.get("n_shelves", self.N_SHELVES_DEFAULT))
        self.shelf_length = float(c.get("shelf_length", self.SHELF_LENGTH_DEFAULT))
        self.shelf_depth = float(c.get("shelf_depth", self.SHELF_DEPTH_DEFAULT))
        self.shelf_thick = float(c.get("shelf_thick", self.SHELF_THICK_DEFAULT))
        self.level_gap = float(c.get("level_gap", self.LEVEL_GAP_DEFAULT))
        self.offset_min_frac = float(c.get("offset_min_frac", self.OFFSET_MIN_FRAC_DEFAULT))
        self.offset_max_frac = float(c.get("offset_max_frac", self.OFFSET_MAX_FRAC_DEFAULT))
        self.tilt_min_deg = float(c.get("tilt_min_deg", self.TILT_MIN_DEG_DEFAULT))
        self.tilt_max_deg = float(c.get("tilt_max_deg", self.TILT_MAX_DEG_DEFAULT))
        self.bottom_clearance = float(c.get("bottom_clearance", self.BOTTOM_CLEARANCE_DEFAULT))
        self.stack_shift_range = float(c.get("stack_shift_range", self.STACK_SHIFT_RANGE_DEFAULT))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        self.roll_speed = float(c.get("roll_speed", self.ROLL_SPEED_DEFAULT))
        self.max_fall_steps = int(c.get("max_fall_steps", self.MAX_FALL_STEPS_DEFAULT))

        self.reactive_marble = bool(c.get("reactive_marble", self.REACTIVE_MARBLE_DEFAULT))
        self.reactive_roll_speed = float(c.get("reactive_roll_speed", self.REACTIVE_ROLL_SPEED_DEFAULT))
        if self.reactive_marble:
            # The marble starts falling immediately (see `play_once`), well before the arm's fixed
            # reach/press sequence finishes -- slow the slide legs down so the descent has a chance
            # of still being catchable. The gravity-timed free-fall legs are untouched.
            self.roll_speed = self.reactive_roll_speed

        self.bowl_id = int(c.get("bowl_id", self.BOWL_ID_DEFAULT))
        self.bowl_radius = float(c.get("bowl_radius", self.BOWL_RADIUS_DEFAULT))
        self.bowl_scale_mult = float(c.get("bowl_scale_mult", self.BOWL_SCALE_MULT_DEFAULT))
        self.bowl_catch_xy_tol = float(c.get("bowl_catch_xy_tol", self.BOWL_CATCH_XY_TOL_DEFAULT))
        self.bowl_speed = float(c.get("bowl_speed", self.BOWL_SPEED_DEFAULT))
        self.belt_thickness = float(c.get("belt_thickness", self.BELT_THICKNESS_DEFAULT))
        self.belt_margin = float(c.get("belt_margin", self.BELT_MARGIN_DEFAULT))

        self.key_half = list(c.get("key_half", self.KEY_HALF_DEFAULT))
        self.key_hover_dis = float(c.get("key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        self.key_press_depth = float(c.get("key_press_depth", self.KEY_PRESS_DEPTH_DEFAULT))
        self.key_press_xy = float(c.get("key_press_xy", self.KEY_PRESS_XY_DEFAULT))
        self.key_press_dz = float(c.get("key_press_dz", self.KEY_PRESS_DZ_DEFAULT))
        self.key_travel = float(c.get("key_travel", self.KEY_TRAVEL_DEFAULT))
        self.key_spring_step = float(c.get("key_spring_step", self.KEY_SPRING_STEP_DEFAULT))
        self.key_x_left = float(c.get("key_x_left", self.KEY_X_LEFT_DEFAULT))
        self.key_x_right = float(c.get("key_x_right", self.KEY_X_RIGHT_DEFAULT))
        self.key_y = float(c.get("key_y", self.KEY_Y_DEFAULT))

        self.press_loop_tol = float(c.get("press_loop_tol", self.PRESS_LOOP_TOL_DEFAULT))
        self.press_loop_max_steps = int(c.get("press_loop_max_steps", self.PRESS_LOOP_MAX_STEPS_DEFAULT))
        self.post_catch_dwell = int(c.get("post_catch_dwell", self.POST_CATCH_DWELL_DEFAULT))

        self.table_top = 0.74 + self.table_z_bias
        self.belt_y = 0.0
        self.belt_surface_z = self.table_top + self.belt_thickness

        self.shelf_half_len = self.shelf_length / 2.0
        self.shelf_half_depth = self.shelf_depth / 2.0
        self.shelf_half_thick = self.shelf_thick / 2.0

        # ---- randomize the zig-zag positions: alternating left/right offsets between consecutive
        # shelves, magnitude in [offset_min_frac, offset_max_frac] * shelf_length so consecutive
        # shelves always overlap by somewhere in (0%, 50%] ----
        s0 = float(np.random.choice([-1.0, 1.0]))
        offset_signs = [s0 * ((-1.0) ** i) for i in range(self.n_shelves - 1)]
        offsets = [
            float(sign * np.random.uniform(
                self.offset_min_frac * self.shelf_length, self.offset_max_frac * self.shelf_length
            ))
            for sign in offset_signs
        ]
        centers = [0.0]
        for off in offsets:
            centers.append(centers[-1] + off)
        shift = float(np.random.uniform(-self.stack_shift_range, self.stack_shift_range))
        self.shelf_centers_x = [c + shift for c in centers]

        # ---- tilt: direction for shelves 0..N-2 is tied to the offset toward the shelf below it
        # (so the marble's downhill edge lines up with the shelf it must land on); the bottom
        # shelf's direction (which side of the belt the marble finally exits toward) is
        # independent/random. Magnitude (steepness) is randomized per shelf, 15-45 degrees. ----
        self.shelf_dir = [1.0 if off > 0 else -1.0 for off in offsets]
        self.shelf_dir.append(float(np.random.choice([-1.0, 1.0])))
        self.shelf_angle_deg = [
            self.shelf_dir[i] * float(np.random.uniform(self.tilt_min_deg, self.tilt_max_deg))
            for i in range(self.n_shelves)
        ]

        bottom_z = self.belt_surface_z + self.bottom_clearance
        top_z = bottom_z + (self.n_shelves - 1) * self.level_gap
        self.shelf_z = [top_z - i * self.level_gap for i in range(self.n_shelves)]

        # ---- precompute the marble's full path (needs belt_surface_z + shelf geometry, not the
        # belt's x-extent) so target_catch_x is known before the belt/bowl bounds are sized ----
        self._compute_descent_plan()

        shelf_min_x = min(self.shelf_centers_x) - self.shelf_half_len
        shelf_max_x = max(self.shelf_centers_x) + self.shelf_half_len
        self.belt_x_min = min(shelf_min_x, self.target_catch_x) - self.belt_margin
        self.belt_x_max = max(shelf_max_x, self.target_catch_x) + self.belt_margin
        self.bowl_x_min = self.belt_x_min + self.bowl_radius + 0.01
        self.bowl_x_max = self.belt_x_max - self.bowl_radius - 0.01
        self.target_catch_x = float(np.clip(self.target_catch_x, self.bowl_x_min, self.bowl_x_max))

        # ---- belt (static) ----
        belt_center_x = 0.5 * (self.belt_x_min + self.belt_x_max)
        belt_half_x = 0.5 * (self.belt_x_max - self.belt_x_min)
        self.belt = create_box(
            self,
            pose=sapien.Pose([belt_center_x, self.belt_y, self.table_top + 0.5 * self.belt_thickness]),
            half_size=[belt_half_x, self.shelf_half_depth + 0.02, 0.5 * self.belt_thickness],
            color=self.BELT_COLOR,
            is_static=True,
            name="marble_belt",
        )
        self.add_prohibit_area(self.belt, padding=0.03)

        # ---- shelves (static, fixed-tilt) ----
        self.shelves = []
        for i in range(self.n_shelves):
            phi = self._shelf_phi(i)
            quat = [np.cos(phi / 2.0), 0.0, np.sin(phi / 2.0), 0.0]
            shelf = create_box(
                self,
                pose=sapien.Pose([self.shelf_centers_x[i], self.belt_y, self.shelf_z[i]], quat),
                half_size=[self.shelf_half_len, self.shelf_half_depth, self.shelf_half_thick],
                color=self.SHELF_COLOR,
                is_static=True,
                name=f"catch_shelf_{i}",
            )
            self.shelves.append(shelf)

        # ---- marble: parked (kinematic) at the centre of the top shelf until play_once releases it ----
        ball_pose = self._shelf_local_to_world(0, 0.0)
        self.ball = create_sphere(
            self,
            pose=sapien.Pose(ball_pose.tolist()),
            radius=self.ball_radius,
            color=self.MARBLE_COLOR,
            is_static=False,
            name="catch_marble",
        )
        self._make_kinematic(self.ball)
        self._marble_state = "parked"
        self._marble_result = None
        self._leg_idx = 0
        self._leg_step = 0

        # ---- bowl: kinematic, rides the belt, starts at the belt's horizontal centre ----
        self.bowl_x_start = 0.5 * (self.bowl_x_min + self.bowl_x_max)
        bowl_pose = sapien.Pose([self.bowl_x_start, self.belt_y, self.belt_surface_z], self.bowl_q)
        self.bowl = create_actor(
            self,
            pose=bowl_pose,
            modelname="002_bowl",
            model_id=self.bowl_id,
            convex=True,
            is_static=False,
            scale_mult=self.bowl_scale_mult,
        )
        self.bowl.set_mass(0.06)
        self._make_kinematic(self.bowl)
        self.add_prohibit_area(self.bowl, padding=0.05)
        if _CSM_DEBUG:
            print(
                f"[CSM] load_actors done: bowl_x_start={self.bowl_x_start:.4f} "
                f"bowl_x_min={self.bowl_x_min:.4f} bowl_x_max={self.bowl_x_max:.4f} "
                f"bowl_pose_p={self.bowl.get_pose().p.round(4)} "
                f"target_catch_x={self.target_catch_x:.4f}",
                flush=True,
            )

        # ---- two action keys: left (pressed by the left arm) slides the bowl left while held,
        # right (pressed by the right arm) slides it right while held ----
        self.key_xy = {
            "left": (self.key_x_left, self.key_y),
            "right": (self.key_x_right, self.key_y),
        }
        self.key_top_z = self.table_top + 2.0 * self.key_half[2]
        key_colors = {"left": self.LEFT_KEY_COLOR, "right": self.RIGHT_KEY_COLOR}
        for side, (kx, ky) in self.key_xy.items():
            create_box(
                self,
                pose=sapien.Pose([kx, ky, self.table_top + 0.010]),
                half_size=[0.040, 0.034, 0.010],
                color=self.KEY_BASE_COLOR,
                is_static=True,
                name=f"action_key_base_{side}",
            )
            self.key_rest_xyz[side] = [kx, ky, self.table_top + self.key_half[2]]
            self.keys[side] = create_box(
                self,
                pose=sapien.Pose(self.key_rest_xyz[side]),
                half_size=self.key_half,
                color=key_colors[side],
                is_static=True,
                name=f"action_key_{side}",
            )
            self.key_arrows[side] = self._draw_arrow(side, kx, ky, self.key_top_z + 0.0015)
            self.add_prohibit_area(self.keys[side], padding=0.04)

        self._loaded = True

    def _draw_arrow(self, side, key_x, key_y, z):
        # Author one left-pointing arrow, then rigidly rotate it 180 degrees for the right key, so
        # the two icons are exact mirror opposites.
        rotation = 0.0 if side == "left" else np.pi
        color = [0.95, 0.95, 0.95]
        parts = [
            ("shaft", [0.003, 0.0], 0.0, [0.013, 0.0025, 0.001]),
            ("head_upper", [-0.011, 0.005], 0.75, [0.009, 0.0025, 0.001]),
            ("head_lower", [-0.011, -0.005], -0.75, [0.009, 0.0025, 0.001]),
        ]
        c, s = np.cos(rotation), np.sin(rotation)
        arrows = []
        for name, (local_x, local_y), local_yaw, half_size in parts:
            x = key_x + c * local_x - s * local_y
            y = key_y + s * local_x + c * local_y
            yaw = local_yaw + rotation
            q = [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]
            arrow = create_visual_box(
                self,
                sapien.Pose([x, y, z], q),
                half_size=half_size,
                color=color,
                name=f"{side}_key_arrow_{name}",
            )
            arrows.append((arrow, [x, y, z]))
        return arrows

    # --------------------------------------------------------------- marble
    def _release_marble(self):
        self._marble_state = "descending"
        self._marble_result = None
        self._leg_idx = 0
        self._leg_step = 0

    def _advance_marble(self):
        if self.ball is None or self._marble_state != "descending":
            return
        legs = self.descent_legs
        if self._leg_idx >= len(legs):
            self._marble_state = "landed"
            return
        leg = legs[self._leg_idx]
        self._leg_step += 1
        step = self._leg_step
        steps_total = max(1, leg["steps"])

        if leg["type"] == "slide":
            frac = min(1.0, step / steps_total)
            local_x = leg["start_local"] + frac * (leg["end_local"] - leg["start_local"])
            pos = self._shelf_local_to_world(leg["shelf"], local_x)
        else:
            dt = float(self.scene.get_timestep())
            t = min(step, steps_total) * dt
            sx, sy, sz = leg["start_pos"]
            x = sx + leg["vx"] * t
            z = max(sz - 0.5 * self.GRAVITY * t * t, self.belt_surface_z + self.ball_radius)
            pos = np.array([x, sy, z], dtype=np.float64)

        self._set_entity_pose(self.ball, sapien.Pose(pos.tolist()))

        if step >= steps_total:
            self._leg_idx += 1
            self._leg_step = 0
            if self._leg_idx >= len(legs):
                self._marble_state = "landed"

    def _resolve_marble(self):
        """Judge catch vs. miss from the bowl's position. Deliberately *not* called the instant
        the marble finishes its (precomputed, fixed-duration) descent -- `_advance_marble` only
        transitions to `"landed"` at that point, and `play_once` calls this afterwards, once the
        arm's own action sequence has actually finished. In `reactive_marble` mode especially, the
        marble's short scripted fall can finish well before the arm's fixed
        close_gripper/reach/press/hold sequence does (the bowl is still correctly *en route* to
        `target_catch_x` at that instant, just not there yet) -- resolving on the spot would judge
        a still-converging approach as a miss."""
        self._marble_state = "resolved"
        bowl_x = float(self.bowl.get_pose().p[0])
        if abs(bowl_x - self.target_catch_x) <= self.bowl_catch_xy_tol:
            self._marble_result = "caught"
        else:
            self._marble_result = "missed"
            self._set_entity_pose(
                self.ball,
                sapien.Pose([self.target_catch_x, self.belt_y, self.belt_surface_z + self.ball_radius]),
            )

    def _ride_marble_in_bowl(self):
        if self.ball is None or self.bowl is None:
            return
        bp = self.bowl.get_pose().p
        self._set_entity_pose(self.ball, sapien.Pose([bp[0], bp[1], bp[2] + self.ball_radius + 0.01]))

    # ------------------------------------------------------------------ keys
    def _detect_action_keys(self):
        if not hasattr(self, "robot"):
            return
        try:
            left_ee = np.asarray(self.robot.get_left_ee_pose()[:3], dtype=np.float64)
            right_ee = np.asarray(self.robot.get_right_ee_pose()[:3], dtype=np.float64)
        except Exception:
            return
        for side, ee in (("left", left_ee), ("right", right_ee)):
            kx, ky = self.key_xy[side]
            pressed = bool(
                abs(ee[0] - kx) <= self.key_press_xy
                and abs(ee[1] - ky) <= self.key_press_xy
                and ee[2] <= self.key_top_z + self.key_press_dz
            )
            if _CSM_DEBUG and pressed != self._key_pressed.get(side, False):
                print(
                    f"[CSM] {side} key pressed={pressed} ee={ee.round(3)} "
                    f"key_xy=({kx:.3f},{ky:.3f}) key_top_z={self.key_top_z:.3f} "
                    f"dz_thresh={self.key_press_dz:.3f}",
                    flush=True,
                )
            self._key_pressed[side] = pressed

    def _spring_key(self, actor, depression, pressed, rest_xyz):
        target = self.key_travel if pressed else 0.0
        delta = float(np.clip(target - depression, -self.key_spring_step, self.key_spring_step))
        depression = float(np.clip(depression + delta, 0.0, self.key_travel))
        pose = actor.get_pose()
        self._set_entity_pose(
            actor, sapien.Pose([rest_xyz[0], rest_xyz[1], rest_xyz[2] - depression], pose.q)
        )
        return depression

    def _animate_keys(self):
        for side, key in self.keys.items():
            self._key_depression[side] = self._spring_key(
                key, self._key_depression[side], self._key_pressed[side], self.key_rest_xyz[side]
            )
            for arrow, rest_xyz in self.key_arrows.get(side, []):
                pose = arrow.get_pose()
                self._set_entity_pose(
                    arrow,
                    sapien.Pose(
                        [rest_xyz[0], rest_xyz[1], rest_xyz[2] - self._key_depression[side]], pose.q
                    ),
                )

    # ------------------------------------------------------------------ bowl
    def _advance_bowl(self):
        if self.bowl is None:
            return
        if self._bowl_force_stop:
            return
        dt = float(self.scene.get_timestep())
        left_p = self._key_pressed.get("left", False)
        right_p = self._key_pressed.get("right", False)
        dx = 0.0
        if left_p and not right_p:
            dx = -self.bowl_speed * dt
        elif right_p and not left_p:
            dx = self.bowl_speed * dt
        if dx == 0.0:
            return
        cur_x = float(self.bowl.get_pose().p[0])
        new_x = cur_x + dx
        # Clamp against an in-progress `_hold_key_to_target` goal so a single fixed-size step can't
        # carry the bowl past the point the scripted policy is aiming for -- otherwise the last step
        # before the loop's tolerance check trips can overshoot by up to one step's worth of travel.
        clamp = self._bowl_drive_clamp
        if clamp is not None:
            clamp_sign, clamp_x = clamp
            new_x = min(new_x, clamp_x) if clamp_sign > 0 else max(new_x, clamp_x)
        new_x = float(np.clip(new_x, self.bowl_x_min, self.bowl_x_max))
        self._set_entity_pose(self.bowl, sapien.Pose([new_x, self.belt_y, self.belt_surface_z], self.bowl_q))

    # ---------------------------------------------------------- scene motion
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._detect_action_keys()
        self._animate_keys()
        self._advance_bowl()
        if self._marble_state == "descending":
            self._advance_marble()
        elif self._marble_state == "resolved" and self._marble_result == "caught":
            self._ride_marble_in_bowl()

    def _dwell(self, steps):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    # ------------------------------------------------------------------ keys / policy helpers
    def _key_tip_pose(self, side, tip_z_above_top):
        kx, ky = self.key_xy[side]
        tcp_z = self.key_top_z + tip_z_above_top
        return [kx, ky, tcp_z + self.EE_TO_TCP, *GRASP_DIRECTION_DIC["top_down"]]

    def _hold_key_to_target(self, side, target_x, max_steps=None, on_pressed=None):
        """Press and hold action key `side` (continuously sliding the bowl) until the bowl's x
        reaches `target_x`, then release. Uses live position feedback (not a fixed step count) so
        the stop point is accurate regardless of IK/timing jitter during the approach.

        `on_pressed`, if given, fires right after the key is physically pressed down (before the
        monitoring loop starts) -- this is used to release the marble at that moment, so the
        marble's (fixed-duration) descent overlaps with the window where the bowl can actually
        move, instead of ticking away during the arm's earlier close-gripper/approach motion."""
        if max_steps is None:
            max_steps = self.press_loop_max_steps
        arm = ArmTag(side)
        # Set the drive clamp *before* any arm motion: `_key_pressed[side]` can flip True mid-way
        # through the press-down displacement below (as soon as the EE crosses the depth
        # threshold), which already drives `_advance_bowl` via `move()`'s own per-step
        # `_update_kinematic_tasks` calls -- so the clamp must be active for that motion too, not
        # just for the explicit monitoring loop that follows.
        self._bowl_force_stop = False
        sign = 1.0 if side == "right" else -1.0
        self._bowl_drive_clamp = (sign, target_x)
        self.move(self.move_to_pose(arm, self._key_tip_pose(side, self.key_hover_dis)))
        if not self.plan_success:
            self._bowl_drive_clamp = None
            return
        self.move(self.move_by_displacement(arm, z=-self.key_press_depth))
        if not self.plan_success:
            self._bowl_drive_clamp = None
            return
        if on_pressed is not None:
            on_pressed()

        steps = 0
        while steps < max_steps:
            cur_x = float(self.bowl.get_pose().p[0])
            if sign * (target_x - cur_x) <= self.press_loop_tol:
                break
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (steps % self.save_freq == 0):
                self._take_picture()
            steps += 1
        self._bowl_drive_clamp = None

        # Lock the bowl in place *before* retracting: the arm's retract motion (move_by_displacement)
        # is IK/timing-driven and its exact duration jitters, which previously let the EE linger over
        # the key's press-depth zone for a variable number of extra steps -- adding an uncontrolled,
        # non-deterministic amount of extra bowl travel on top of the precisely-monitored loop above.
        # Forcing the stop here makes the bowl's final position bounded purely by `press_loop_tol`.
        self._bowl_force_stop = True
        self.move(self.move_by_displacement(arm, z=self.key_press_depth))
        self._dwell(6)

    # ------------------------------------------------------------------ policy
    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")

        if _CSM_DEBUG:
            print(f"[CSM] play_once start: bowl_pose_p={self.bowl.get_pose().p.round(4)}", flush=True)

        if self.reactive_marble:
            # Drop the marble before the arm does anything at all -- every subsequent move() call
            # steps `_update_kinematic_tasks` (and therefore `_advance_marble`) internally, so the
            # descent keeps running concurrently with close_gripper/reach/press below instead of
            # waiting for them.
            self._release_marble()

        self.move(self.close_gripper(left))
        self.move(self.close_gripper(right))

        if _CSM_DEBUG:
            print(
                f"[CSM] after close_gripper: bowl_pose_p={self.bowl.get_pose().p.round(4)} "
                f"plan_success={self.plan_success} total_marble_steps={self.total_marble_steps}",
                flush=True,
            )

        bowl_x0 = float(self.bowl.get_pose().p[0])
        dx = self.target_catch_x - bowl_x0
        held_side = None
        if abs(dx) > self.press_loop_tol:
            held_side = "right" if dx > 0 else "left"
            if _CSM_DEBUG:
                print(
                    f"[CSM] holding {held_side}: bowl_x0={bowl_x0:.4f} dx={dx:.4f} "
                    f"target_catch_x={self.target_catch_x:.4f}",
                    flush=True,
                )
            # In the default (non-reactive) mode, release the marble only once the key is actually
            # pressed down, so its fixed-duration fall overlaps with the window where the bowl can
            # move (not the earlier close-gripper/approach steps, which don't move the bowl at
            # all). In reactive mode the marble is already falling (released at the top of
            # play_once above), so there's nothing to trigger here.
            on_pressed = None if self.reactive_marble else self._release_marble
            self._hold_key_to_target(held_side, self.target_catch_x, on_pressed=on_pressed)
            if _CSM_DEBUG:
                print(
                    f"[CSM] after hold: bowl_x={float(self.bowl.get_pose().p[0]):.4f}",
                    flush=True,
                )
        elif not self.reactive_marble:
            self._release_marble()

        # Let the marble finish its full (precomputed) descent regardless of how long the key
        # hold above took; this loop is a pure step-and-wait, no arm motion.
        max_wait = self.total_marble_steps + 60
        waited = 0
        while self._marble_state == "descending" and waited < max_wait:
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (waited % self.save_freq == 0):
                self._take_picture()
            waited += 1

        # Judge catch vs. miss only now, using the bowl's actual (by now fully settled) position --
        # not at whatever earlier instant the marble's fixed-duration descent happened to finish.
        self._resolve_marble()
        self._dwell(self.post_catch_dwell)

        self.info["info"] = {
            "{A}": "marble",
            "{B}": "tilted shelves",
            "{C}": f"002_bowl/base{self.bowl_id}",
            "{D}": f"{held_side or 'either'} action key",
            "{a}": f"{held_side or 'left'} arm",
        }
        return self.info

    # ------------------------------------------------------------------ success
    def check_success(self):
        self.info["target_catch_x"] = float(self.target_catch_x)
        self.info["marble_result"] = str(self._marble_result)
        self.info["marble_position"] = list(map(float, self.ball.get_pose().p)) if self.ball is not None else []
        return bool(self._marble_state == "resolved" and self._marble_result == "caught")

    def get_obs(self):
        obs = super().get_obs()
        obs["catch_shelf_marble"] = {
            "shelf_centers_x": list(map(float, self.shelf_centers_x)),
            "shelf_z": list(map(float, self.shelf_z)),
            "shelf_angles_deg": list(map(float, self.shelf_angle_deg)),
            "target_catch_x": float(self.target_catch_x),
            "bowl_x": float(self.bowl.get_pose().p[0]) if self.bowl is not None else 0.0,
            "marble_state": str(self._marble_state),
            "marble_result": str(self._marble_result),
            "marble_position": (
                list(map(float, self.ball.get_pose().p)) if self.ball is not None else [0.0, 0.0, 0.0]
            ),
        }
        return obs
