from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import os
import sapien
import sapien.render
import sapien.physx
import numpy as np
import transforms3d as t3d


class play_billiard(Base_Task):
    """Single-arm billiards: strike the red primary ball into a pocket with a cue stick.

    Table is centered at (x=0, y=0.1). The primary ball spawns on a random side; the
    matching arm (left/right) grasps a cue placed on that same side. Success = primary
    ball falls through an allowed pocket into the hollow interior. Robot-link contact
    with the primary ball fails (cue contact is allowed).

    Config knobs (task_args.play_billiard):
      pocket_mode: "any" | "top_three"
        any       — success in any of the 6 pockets
        top_three — episode picks one of top-left / top-middle / top-right
      blocker_mode: "none" | "path_to_top" | "unused_tops"
        path_to_top — up to 2 colored balls between primary and a top-three pocket
        unused_tops — up to 2 balls blocking the top pockets that are NOT the goal
      num_extra_balls: 0..2

    Built entirely from SAPIEN primitives (reserved ids [350,359] in the NOTICE below).
    """

    # ----- table (centered; doubled playing surface)
    TABLE_CX_DEFAULT = 0.0
    TABLE_CY_DEFAULT = 0.1
    TABLE_HALF_LEN_DEFAULT = 0.22   # along +X (long side); full length 0.44 m
    TABLE_HALF_WID_DEFAULT = 0.14   # along +Y (short side); full width 0.28 m
    BASE_HEIGHT = 0.10              # 10 cm hollow wood box under the felt
    WALL_T = 0.012                  # wall thickness of the hollow box
    FLOOR_HALF_Z = 0.005            # thin inner floor so pocketed balls rest inside
    LID_HALF_Z = 0.003              # thin green lid (with real pocket openings)
    RAIL_HALF_H = 0.002             # very low cushion so the cue clears it
    RAIL_HALF_T = 0.006
    # Pocket diameter = 1.2 × ball diameter ⇒ radius = 1.2 × ball radius.
    POCKET_DIAMETER_SCALE = 1.2
    BASE_COLOR = [0.42, 0.26, 0.12]
    FELT_COLOR = [0.10, 0.45, 0.22]
    RAIL_COLOR = [0.35, 0.18, 0.08]
    POCKET_COLOR = [0.04, 0.04, 0.04]
    # Cylinder default axis is +X; this quat aligns it with +Z (vertical wells).
    Z_CYL_Q = t3d.quaternions.axangle2quat([0.0, 1.0, 0.0], np.pi / 2).tolist()

    # Pocket index layout from _build_table:
    # 0 near_left, 1 top_left, 2 near_right, 3 top_right, 4 near_middle, 5 top_middle
    POCKET_NAMES = [
        "near_left", "top_left", "near_right", "top_right", "near_middle", "top_middle",
    ]
    TOP_POCKET_IDS = (1, 5, 3)  # left, middle, right along the far (+Y) edge

    # ----- balls
    BALL_RADIUS_DEFAULT = 0.012
    NUM_EXTRA_BALLS_DEFAULT = 0
    PRIMARY_COLOR = [0.90, 0.12, 0.12]
    EXTRA_COLORS = [
        [0.15, 0.35, 0.90],  # blue
        [0.95, 0.75, 0.10],  # yellow
        [0.55, 0.20, 0.75],  # purple
    ]
    BALL_MASS = 0.04

    # ----- cue rod (local +X = tip direction; cylinder + spherical ends)
    CUE_RADIUS = 0.006
    CUE_HALF_LEN = 0.060
    CUE_COLOR = [0.72, 0.52, 0.28]
    CUE_TIP_COLOR = [0.92, 0.92, 0.85]
    CUE_MASS = 0.05

    # ----- strike / settle
    APPROACH_GAP = 0.050
    STRIKE_PUSH = 0.055
    # Tip hover clearance above the felt so the shaft clears the 10 cm box + rails.
    HOVER_CLEARANCE = 0.10
    SETTLE_STEPS_DEFAULT = 500
    STRIKE_IMPULSE = 0.48
    # Require tip within this of the ball surface before applying impulse.
    STRIKE_CONTACT_GAP = 0.006
    POCKET_SINK_Z = 0.012

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("play_billiard", {})
        # Invalidate per-step state before _init_task_env_ (it may call
        # _update_kinematic_tasks during camera load, before load_actors).
        self._loaded = False
        self._cue_welded = False
        self._primary_pocketed = False
        self._primary_pocket_id = None
        self._robot_ball_contact = False
        self._strike_armed = False
        self._strike_done = False
        self.cue = None
        self.primary_ball = None
        self.extra_balls = []
        self._pocket_centers = []
        self._arm_side = "right"
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.table_cx = float(c.get("table_cx", self.TABLE_CX_DEFAULT))
        self.table_cy = float(c.get("table_cy", self.TABLE_CY_DEFAULT))
        self.table_half_len = float(c.get("table_half_len", self.TABLE_HALF_LEN_DEFAULT))
        self.table_half_wid = float(c.get("table_half_wid", self.TABLE_HALF_WID_DEFAULT))
        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        # Pocket diameter = 1.2 × ball diameter (override with pocket_radius if given).
        default_pr = self.POCKET_DIAMETER_SCALE * self.ball_radius
        self.pocket_radius = float(c.get("pocket_radius", default_pr))
        self.num_extra_balls = int(np.clip(int(c.get("num_extra_balls", self.NUM_EXTRA_BALLS_DEFAULT)), 0, 2))
        self.pocket_mode = str(c.get("pocket_mode", "any")).lower()
        if self.pocket_mode not in ("any", "top_three"):
            self.pocket_mode = "any"
        self.blocker_mode = str(c.get("blocker_mode", "none")).lower()
        if self.blocker_mode not in ("none", "path_to_top", "unused_tops"):
            self.blocker_mode = "none"
        self.settle_steps = int(c.get("settle_steps", self.SETTLE_STEPS_DEFAULT))
        self.strike_impulse = float(c.get("strike_impulse", self.STRIKE_IMPULSE))

        self.z0 = 0.74 + self.table_z_bias
        self.base_top = self.z0 + self.BASE_HEIGHT
        self.felt_top = self.base_top + 2.0 * self.LID_HALF_Z
        self.floor_top = self.z0 + 2.0 * self.FLOOR_HALF_Z
        self.ball_z = self.felt_top + self.ball_radius + 0.001

        self._primary_pocketed = False
        self._primary_pocket_id = None
        self._robot_ball_contact = False
        self._cue_welded = False
        self._strike_armed = False
        self._strike_done = False
        self._aim_dir = np.array([1.0, 0.0], dtype=np.float64)
        self._target_pocket = None
        self._target_pocket_id = None
        self._target_pocket_name = "any"
        self._allowed_pocket_ids = list(range(6))

        self._build_table()
        self._spawn_balls()
        self._spawn_cue()
        self._robot_link_names = self._collect_robot_link_names()
        self._loaded = True

    # ------------------------------------------------------------- table build
    def _add_lid_panel(self, x, y, hx, hy, name):
        """Thin green lid panel with collision (leave gaps at pockets)."""
        if hx < 0.004 or hy < 0.004:
            return
        create_box(
            self,
            pose=sapien.Pose([x, y, self._lid_z], [1, 0, 0, 0]),
            half_size=[hx, hy, self.LID_HALF_Z],
            color=tuple(self.FELT_COLOR),
            is_static=True,
            name=name,
        )

    def _add_circular_pocket_visual(self, px, py, index):
        """Non-colliding dark disc so the opening reads as a round hole from above."""
        well_mat = sapien.render.RenderMaterial(base_color=[*self.POCKET_COLOR, 1.0])
        builder = self.scene.create_actor_builder()
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.Z_CYL_Q),
            radius=self.pocket_radius,
            half_length=0.0010,
            material=well_mat,
        )
        # Slightly below the lid plane so it sits in the opening (no collision).
        builder.set_initial_pose(
            sapien.Pose([px, py, self.felt_top - 0.001], [1, 0, 0, 0])
        )
        builder.build_static(name=f"pocket_well_{index}")

    def _build_hollow_box(self, cx, cy, hl, hw):
        """10 cm hollow wood box: floor + four walls (empty interior)."""
        wt = self.WALL_T
        wall_hz = 0.5 * self.BASE_HEIGHT
        wall_z = self.z0 + wall_hz
        floor_z = self.z0 + self.FLOOR_HALF_Z
        inner_hl = hl - wt
        inner_hw = hw - wt

        # Inner floor — pocketed balls land here.
        create_box(
            self,
            pose=sapien.Pose([cx, cy, floor_z], [1, 0, 0, 0]),
            half_size=[inner_hl, inner_hw, self.FLOOR_HALF_Z],
            color=tuple(self.BASE_COLOR),
            is_static=True,
            name="billiard_floor",
        )
        # Long walls (±Y).
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            create_box(
                self,
                pose=sapien.Pose(
                    [cx, cy + sign * (hw - 0.5 * wt), wall_z], [1, 0, 0, 0]
                ),
                half_size=[hl, 0.5 * wt, wall_hz],
                color=tuple(self.BASE_COLOR),
                is_static=True,
                name=f"billiard_wall_long_{tag}",
            )
        # Short walls (±X), inset so corners do not double-stack.
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            create_box(
                self,
                pose=sapien.Pose(
                    [cx + sign * (hl - 0.5 * wt), cy, wall_z], [1, 0, 0, 0]
                ),
                half_size=[0.5 * wt, hw - wt, wall_hz],
                color=tuple(self.BASE_COLOR),
                is_static=True,
                name=f"billiard_wall_short_{tag}",
            )

    def _build_table(self):
        cx, cy = self.table_cx, self.table_cy
        hl, hw = self.table_half_len, self.table_half_wid
        pr = self.pocket_radius
        # Inset so each pocket is a full circle inside the green lid.
        rim = 0.008
        pin = pr + rim
        self._lid_z = self.base_top + self.LID_HALF_Z

        self._pocket_centers = [
            np.array([cx - hl + pin, cy - hw + pin, self.felt_top], dtype=np.float64),
            np.array([cx - hl + pin, cy + hw - pin, self.felt_top], dtype=np.float64),
            np.array([cx + hl - pin, cy - hw + pin, self.felt_top], dtype=np.float64),
            np.array([cx + hl - pin, cy + hw - pin, self.felt_top], dtype=np.float64),
            np.array([cx, cy - hw + pin, self.felt_top], dtype=np.float64),
            np.array([cx, cy + hw - pin, self.felt_top], dtype=np.float64),
        ]

        # 1) Hollow 10 cm wood box (empty cavity under the lid).
        self._build_hollow_box(cx, cy, hl, hw)

        # 2) Green lid with real openings at each pocket (ball can fall through).
        # Center panel.
        self._add_lid_panel(cx, cy, hl - pin - pr, hw - pin - pr, "felt_center")
        # Long-edge bands (±Y), split around corner + side pockets.
        for sign_y, tag in ((-1.0, "neg"), (1.0, "pos")):
            y_outer = cy + sign_y * (hw - 0.5 * rim)
            y_inner = cy + sign_y * (hw - pin - pr)
            y = 0.5 * (y_outer + y_inner)
            hy = abs(y_outer - y_inner) * 0.5
            x_bounds = [
                (cx - hl, cx - hl + pin - pr),
                (cx - hl + pin + pr, cx - pr),
                (cx + pr, cx + hl - pin - pr),
                (cx + hl - pin + pr, cx + hl),
            ]
            for j, (x0, x1) in enumerate(x_bounds):
                self._add_lid_panel(
                    0.5 * (x0 + x1), y, 0.5 * (x1 - x0), hy, f"felt_long_{tag}_{j}"
                )
        # Short-edge bands (±X) between the two corner pockets.
        for sign_x, tag in ((-1.0, "neg"), (1.0, "pos")):
            x_outer = cx + sign_x * (hl - 0.5 * rim)
            x_inner = cx + sign_x * (hl - pin - pr)
            x = 0.5 * (x_outer + x_inner)
            hx = abs(x_outer - x_inner) * 0.5
            y0, y1 = cy - hw + pin + pr, cy + hw - pin - pr
            self._add_lid_panel(
                x, 0.5 * (y0 + y1), hx, 0.5 * (y1 - y0), f"felt_short_{tag}"
            )

        # Circular dark discs (visual only) marking each pocket opening.
        for i, p in enumerate(self._pocket_centers):
            self._add_circular_pocket_visual(float(p[0]), float(p[1]), i)

        # 3) Very low rails (cue clears them); gaps near pockets.
        rail_z = self.felt_top + self.RAIL_HALF_H
        gap = pr * 1.15
        for sign_y, tag in ((-1.0, "neg"), (1.0, "pos")):
            y = cy + sign_y * (hw + self.RAIL_HALF_T)
            segs = [
                (cx - (hl + gap) * 0.5, (hl - gap) * 0.5),
                (cx + (hl + gap) * 0.5, (hl - gap) * 0.5),
            ]
            for j, (sx, shx) in enumerate(segs):
                if shx <= 0.008:
                    continue
                create_box(
                    self,
                    pose=sapien.Pose([sx, y, rail_z], [1, 0, 0, 0]),
                    half_size=[shx, self.RAIL_HALF_T, self.RAIL_HALF_H],
                    color=tuple(self.RAIL_COLOR),
                    is_static=True,
                    name=f"rail_long_{tag}_{j}",
                )
        for sign_x, tag in ((-1.0, "neg"), (1.0, "pos")):
            x = cx + sign_x * (hl + self.RAIL_HALF_T)
            sh = hw - gap
            if sh > 0.01:
                create_box(
                    self,
                    pose=sapien.Pose([x, cy, rail_z], [1, 0, 0, 0]),
                    half_size=[self.RAIL_HALF_T, sh, self.RAIL_HALF_H],
                    color=tuple(self.RAIL_COLOR),
                    is_static=True,
                    name=f"rail_short_{tag}",
                )

        self.prohibited_area.append([
            cx - hl - 0.04,
            cy - hw - 0.04,
            cx + hl + 0.04,
            cy + hw + 0.04,
        ])

    # ------------------------------------------------------------- balls
    def _table_xy_bounds(self):
        margin = self.pocket_radius + self.ball_radius + 0.015
        x_lo = self.table_cx - self.table_half_len + margin
        x_hi = self.table_cx + self.table_half_len - margin
        y_lo = self.table_cy - self.table_half_wid + margin
        y_hi = self.table_cy + self.table_half_wid - margin
        return x_lo, x_hi, y_lo, y_hi

    def _select_goal_pocket(self):
        """Choose the episode goal pocket (and allowed success set) from pocket_mode."""
        if self.pocket_mode == "top_three":
            forced = str(self._cfg.get("target_pocket", "")).lower()
            name_to_id = {
                "top_left": 1, "top_middle": 5, "top_mid": 5, "top_right": 3,
            }
            if forced in name_to_id:
                pid = name_to_id[forced]
            else:
                pid = int(np.random.choice(self.TOP_POCKET_IDS))
            self._target_pocket_id = pid
            self._target_pocket = self._pocket_centers[pid]
            self._target_pocket_name = self.POCKET_NAMES[pid]
            self._allowed_pocket_ids = [pid]
        else:
            self._target_pocket_id = None
            self._target_pocket = None
            self._target_pocket_name = "any"
            self._allowed_pocket_ids = list(range(6))

    def _spawn_balls(self):
        x_lo, x_hi, y_lo, y_hi = self._table_xy_bounds()

        # Random left/right side for primary ball (arm must match that side).
        self._arm_side = "left" if np.random.rand() < 0.5 else "right"
        # Keep primary clearly on one half for reachability.
        if self._arm_side == "right":
            px = float(np.random.uniform(max(0.04, x_lo), min(0.16, x_hi)))
        else:
            px = float(np.random.uniform(max(x_lo, -0.16), min(-0.04, x_hi)))
        py = float(np.clip(self.table_cy + np.random.uniform(-0.05, 0.05), y_lo, y_hi))
        self.primary_ball = self._make_ball(
            [px, py, self.ball_z], self.PRIMARY_COLOR, "primary_ball"
        )
        self._primary_rigid = self._get_rigid_entity(self.primary_ball)

        # Goal pocket before blockers so unused_tops / path placement know the target.
        self._select_goal_pocket()

        self.extra_balls = []
        self._extra_rigids = []
        occupied = [np.array([px, py], dtype=np.float64)]
        if self.blocker_mode == "none" or self.num_extra_balls <= 0:
            return

        n = self.num_extra_balls
        if self.blocker_mode == "unused_tops":
            # Block top pockets that are not the chosen goal.
            focus_id = self._target_pocket_id
            if focus_id is None:
                # pocket_mode=any: pick a focus top pocket for blocker placement only.
                focus_id = int(np.random.choice(self.TOP_POCKET_IDS))
            unused = [i for i in self.TOP_POCKET_IDS if i != focus_id]
            np.random.shuffle(unused)
            for i, pid in enumerate(unused[:n]):
                pos = self._blocker_near_pocket(pid, occupied)
                if pos is None:
                    continue
                color = self.EXTRA_COLORS[i % len(self.EXTRA_COLORS)]
                ball = self._make_ball([pos[0], pos[1], self.ball_z], color, f"extra_ball_{i}")
                self.extra_balls.append(ball)
                self._extra_rigids.append(self._get_rigid_entity(ball))
                occupied.append(pos)
        else:
            # path_to_top: place up to 2 balls between primary and one of the top-three holes.
            top_ids = list(self.TOP_POCKET_IDS)
            np.random.shuffle(top_ids)
            path_pocket = self._pocket_centers[top_ids[0]]
            for i in range(n):
                pos = self._blocker_on_path(
                    occupied[0], path_pocket[:2], occupied, frac=0.35 + 0.2 * i
                )
                if pos is None:
                    # Fallback: near that pocket mouth.
                    pos = self._blocker_near_pocket(top_ids[0], occupied)
                if pos is None:
                    continue
                color = self.EXTRA_COLORS[i % len(self.EXTRA_COLORS)]
                ball = self._make_ball([pos[0], pos[1], self.ball_z], color, f"extra_ball_{i}")
                self.extra_balls.append(ball)
                self._extra_rigids.append(self._get_rigid_entity(ball))
                occupied.append(pos)

    def _blocker_near_pocket(self, pocket_id, occupied, min_sep=None):
        """Place a blocker just inboard of a pocket mouth."""
        if min_sep is None:
            min_sep = 3.0 * self.ball_radius
        x_lo, x_hi, y_lo, y_hi = self._table_xy_bounds()
        pxy = self._pocket_centers[pocket_id][:2]
        center = np.array([self.table_cx, self.table_cy], dtype=np.float64)
        inward = center - pxy
        n = float(np.linalg.norm(inward))
        if n < 1e-6:
            return None
        inward /= n
        for dist in (0.045, 0.055, 0.035, 0.065):
            for jitter in range(12):
                lat = np.array([-inward[1], inward[0]]) * np.random.uniform(-0.02, 0.02)
                cand = pxy + inward * dist + lat
                cand[0] = float(np.clip(cand[0], x_lo, x_hi))
                cand[1] = float(np.clip(cand[1], y_lo, y_hi))
                if all(np.linalg.norm(cand - o) >= min_sep for o in occupied):
                    return cand
        return None

    def _blocker_on_path(self, start_xy, end_xy, occupied, frac=0.45, min_sep=None):
        if min_sep is None:
            min_sep = 3.0 * self.ball_radius
        x_lo, x_hi, y_lo, y_hi = self._table_xy_bounds()
        d = end_xy - start_xy
        if float(np.linalg.norm(d)) < 1e-6:
            return None
        for _ in range(20):
            f = float(np.clip(frac + np.random.uniform(-0.08, 0.08), 0.25, 0.75))
            perp = np.array([-d[1], d[0]], dtype=np.float64)
            pn = float(np.linalg.norm(perp))
            if pn > 1e-6:
                perp = perp / pn * np.random.uniform(-0.015, 0.015)
            else:
                perp = 0.0
            cand = start_xy + d * f + perp
            cand[0] = float(np.clip(cand[0], x_lo, x_hi))
            cand[1] = float(np.clip(cand[1], y_lo, y_hi))
            if all(np.linalg.norm(cand - o) >= min_sep for o in occupied):
                return cand
        return None

    def _make_ball(self, pos, color, name):
        ball = create_sphere(
            self.scene,
            pose=sapien.Pose(pos, [1, 0, 0, 0]),
            radius=self.ball_radius,
            color=tuple(color),
            is_static=False,
            name=name,
        )
        rigid = self._get_rigid_entity(ball)
        if rigid is not None:
            try:
                rigid.set_mass(self.BALL_MASS)
                rigid.set_linear_damping(0.15)
                rigid.set_angular_damping(0.4)
            except Exception:
                pass
            # Low friction so the ball rolls toward pockets after a strike.
            try:
                mat = sapien.physx.PhysxMaterial(
                    static_friction=0.15, dynamic_friction=0.10, restitution=0.55
                )
                for shape in rigid.get_collision_shapes():
                    shape.set_physical_material(mat)
            except Exception:
                pass
        return ball

    # ------------------------------------------------------------- cue
    def _spawn_cue(self):
        # Near zone on the same side as the primary ball / arm.
        table_near_y = self.table_cy - self.table_half_wid
        if self._arm_side == "right":
            cue_x = float(np.random.uniform(0.12, 0.20))
        else:
            cue_x = float(np.random.uniform(-0.20, -0.12))
        cue_y = float(np.random.uniform(-0.18, min(-0.12, table_near_y - 0.06)))
        cue_z = self.z0 + self.CUE_RADIUS
        pose = sapien.Pose([cue_x, cue_y, cue_z], [1, 0, 0, 0])
        self.cue = self._build_cue(pose)
        self.cue.set_mass(self.CUE_MASS)
        self._cue_rigid = self._get_rigid(self.cue)
        self.add_prohibit_area(self.cue, padding=0.04)

    def _build_cue(self, pose):
        """Round cue rod: cylinder shaft with spherical tip and butt."""
        builder = self.scene.create_actor_builder()
        r = self.CUE_RADIUS
        half = self.CUE_HALF_LEN
        mat = self.scene.default_physical_material
        # Cylinder default axis = local +X (tip direction).
        builder.add_cylinder_collision(radius=r, half_length=half, material=mat)
        builder.add_sphere_collision(pose=sapien.Pose([half, 0, 0]), radius=r, material=mat)
        builder.add_sphere_collision(pose=sapien.Pose([-half, 0, 0]), radius=r, material=mat)

        shaft_mat = sapien.render.RenderMaterial(base_color=[*self.CUE_COLOR, 1.0])
        tip_mat = sapien.render.RenderMaterial(base_color=[*self.CUE_TIP_COLOR, 1.0])
        butt_mat = sapien.render.RenderMaterial(base_color=[*self.CUE_COLOR, 1.0])
        builder.add_cylinder_visual(radius=r, half_length=half, material=shaft_mat)
        builder.add_sphere_visual(pose=sapien.Pose([half, 0, 0]), radius=r, material=tip_mat)
        builder.add_sphere_visual(pose=sapien.Pose([-half, 0, 0]), radius=r, material=butt_mat)
        builder.set_initial_pose(pose)
        entity = builder.build(name="cue_stick")

        tip_x = half + r  # apex of the tip sphere
        data = {
            "scale": [1.0, 1.0, 1.0],
            "center": [0, 0, 0],
            "extents": [2 * tip_x, 2 * r, 2 * r],
            "transform_matrix": np.eye(4).tolist(),
            "target_pose": [np.eye(4).tolist()],
            # Top-down grasp on the shaft.
            "contact_points_pose": [
                [
                    [1, 0, 0, -0.015],
                    [0, 0, -1, 0.0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
                [
                    [1, 0, 0, 0.01],
                    [0, 0, -1, 0.0],
                    [0, 1, 0, 0.0],
                    [0, 0, 0, 1],
                ],
            ],
            "functional_matrix": [
                [
                    [1, 0, 0, tip_x],
                    [0, 1, 0, 0.0],
                    [0, 0, 1, 0.0],
                    [0, 0, 0, 1],
                ],
            ],
        }
        return Actor(entity, data, mass=self.CUE_MASS)

    # ------------------------------------------------------------- helpers
    def _get_rigid(self, actor):
        ent = actor.actor if isinstance(actor, Actor) else actor
        for c in ent.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _get_rigid_entity(self, entity):
        for c in entity.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
        return None

    def _collect_robot_link_names(self):
        names = set()
        for articulation in (self.robot.left_entity, self.robot.right_entity):
            if articulation is None:
                continue
            for link in articulation.get_links():
                names.add(link.get_name())
        return names

    def _pose_to_mat(self, pose7):
        T = np.eye(4)
        T[:3, :3] = t3d.quaternions.quat2mat(np.asarray(pose7[3:], dtype=np.float64))
        T[:3, 3] = np.asarray(pose7[:3], dtype=np.float64)
        return T

    def _mat_to_pose(self, T):
        q = t3d.quaternions.mat2quat(T[:3, :3])
        return sapien.Pose(T[:3, 3].tolist(), q.tolist())

    def _ball_xy(self, ball):
        return np.asarray(ball.get_pose().p[:2], dtype=np.float64)

    def _line_clear(self, start_xy, end_xy, blockers, clearance):
        """True if no blocker center is within `clearance` of the segment start→end."""
        d = end_xy - start_xy
        length = float(np.linalg.norm(d))
        if length < 1e-6:
            return True
        u = d / length
        for b in blockers:
            v = b - start_xy
            t = float(np.clip(np.dot(v, u), 0.0, length))
            closest = start_xy + t * u
            if np.linalg.norm(b - closest) < clearance:
                return False
        return True

    def _behind_in_workspace(self, behind):
        """Approach point must stay on the active arm's half of the table."""
        if behind[1] < -0.14 or behind[1] > 0.26:
            return False
        if self._arm_side == "right":
            return 0.02 <= behind[0] <= 0.32
        return -0.32 <= behind[0] <= -0.02

    def _choose_pocket(self):
        """Pick an aim pocket for the expert (respects goal mode + clear path)."""
        ball_xy = self._ball_xy(self.primary_ball)
        blockers = [self._ball_xy(b) for b in self.extra_balls]
        clearance = 2.2 * self.ball_radius

        # Prefer the episode goal when pocket_mode=top_three; else any allowed.
        if self.pocket_mode == "top_three" and self._target_pocket_id is not None:
            aim_ids = [self._target_pocket_id]
        else:
            aim_ids = list(self._allowed_pocket_ids)

        def collect(ids, require_plus_y=True):
            out = []
            for pid in ids:
                pocket = self._pocket_centers[pid]
                pxy = pocket[:2]
                aim = pxy - ball_xy
                dist = float(np.linalg.norm(aim))
                if dist < self.pocket_radius + self.ball_radius:
                    continue
                behind = ball_xy - (aim / dist) * self.APPROACH_GAP
                if not self._behind_in_workspace(behind):
                    continue
                if require_plus_y and aim[1] < -0.005:
                    continue
                clear = self._line_clear(ball_xy, pxy, blockers, clearance)
                score = dist + (0.0 if clear else 0.55)
                score -= 0.20 * max(0.0, float(aim[1]))
                # Prefer the three far (+Y) pockets — more reliable expert sinks.
                if pid in self.TOP_POCKET_IDS:
                    score -= 0.08
                out.append((score, clear, pocket, pid))
            return out

        candidates = collect(aim_ids, require_plus_y=True)
        if not candidates:
            candidates = collect(aim_ids, require_plus_y=False)
        if not candidates and self.pocket_mode == "any":
            candidates = collect(list(range(6)), require_plus_y=False)
        if not candidates:
            # Absolute fallback: farthest +Y pocket on this side.
            pid = max(
                (1, 5) if self._arm_side == "left" else (3, 5),
                key=lambda i: float(self._pocket_centers[i][1]),
            )
            return self._pocket_centers[pid], pid
        clear_ones = [c for c in candidates if c[1]]
        pool = clear_ones if clear_ones else candidates
        pool.sort(key=lambda x: x[0])
        return pool[0][2], pool[0][3]

    # ---------------------------------------------------- weld / kinematics
    def _finger_midpoint(self, arm_tag):
        """World-frame midpoint between the two WSG finger links."""
        ent = (
            self.robot.left_entity
            if str(arm_tag) == "left"
            else self.robot.right_entity
        )
        pts = []
        for link in ent.get_links():
            if "finger" in link.get_name().lower():
                pts.append(np.asarray(link.pose.p, dtype=np.float64))
        if len(pts) >= 2:
            return np.mean(pts, axis=0)
        tcp = (
            self.robot.get_left_tcp_pose()
            if str(arm_tag) == "left"
            else self.robot.get_right_tcp_pose()
        )
        return np.asarray(tcp[:3], dtype=np.float64)

    def _cue_grasp_local_T(self, arm_tag):
        """Cue pose in planning-EE frame with shaft through the jaw aperture.

        Keeps the post-grasp orientation and places the grasp contact (cue local
        x=-0.015) at the finger-link midpoint so the stick sits between the pads.
        """
        ee_T = self._pose_to_mat(
            np.asarray(self.get_arm_pose(str(arm_tag)), dtype=np.float64)
        )
        cue_T = self.cue.get_pose().to_transformation_matrix()
        R = cue_T[:3, :3]
        mid = self._finger_midpoint(arm_tag)
        contact_x = -0.015  # must match _build_cue contact_points_pose
        cue_ideal = np.eye(4, dtype=np.float64)
        cue_ideal[:3, :3] = R
        cue_ideal[:3, 3] = mid - R @ np.array([contact_x, 0.0, 0.0], dtype=np.float64)
        return np.linalg.inv(ee_T) @ cue_ideal

    def _seat_and_weld_cue(self, arm_tag):
        """Force-seat the cue between the fingers, then weld that transform."""
        local_T = self._cue_grasp_local_T(arm_tag)
        ee = np.asarray(self.get_arm_pose(str(arm_tag)), dtype=np.float64)
        cue_T = self._pose_to_mat(ee) @ local_T
        pose = self._mat_to_pose(cue_T)
        self.cue.actor.set_pose(pose)
        rigid = self._get_rigid(self.cue)
        if rigid is not None:
            rigid.set_disable_gravity(True)
            rigid.set_kinematic(True)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
            rigid.set_kinematic_target(pose)
        self._cue_ee_T = local_T.copy()
        self._cue_arm = str(arm_tag)
        self._cue_welded = True

    def _update_welded_cue(self):
        if not self._cue_welded:
            return
        ee = np.asarray(self.get_arm_pose(self._cue_arm), dtype=np.float64)
        cue_T = self._pose_to_mat(ee) @ self._cue_ee_T
        pose = self._mat_to_pose(cue_T)
        self.cue.actor.set_pose(pose)
        rigid = self._get_rigid(self.cue)
        if rigid is not None:
            rigid.set_kinematic_target(pose)

    def _check_robot_ball_contact(self):
        if self._robot_ball_contact or not self._loaded:
            return
        if self.primary_ball is None:
            return
        ball_name = self.primary_ball.get_name()
        for contact in self.scene.get_contacts():
            name0 = contact.bodies[0].entity.name
            name1 = contact.bodies[1].entity.name
            if (
                (name0 == ball_name and name1 in self._robot_link_names)
                or (name1 == ball_name and name0 in self._robot_link_names)
            ):
                self._robot_ball_contact = True
                return

    def _stick_tip_dir_xy(self):
        """Unit XY direction the cue tip is pointing (stick orientation, not pocket aim)."""
        tip = self._tip_xyz()
        cue_p = np.asarray(self.cue.get_pose().p, dtype=np.float64)
        d = tip[:2] - cue_p[:2]
        n = float(np.linalg.norm(d))
        if n < 1e-4:
            return np.array(self._aim_dir, dtype=np.float64)
        return d / n

    def _try_apply_strike_impulse(self):
        """Impart velocity only after real tip–ball contact, along the stick direction."""
        if not self._strike_armed or self._strike_done or self._primary_pocketed:
            return
        tip = self._tip_xyz()
        ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
        contact_thresh = self.ball_radius + self.CUE_RADIUS + self.STRIKE_CONTACT_GAP
        dist = float(np.linalg.norm(tip - ball_p))
        touching = False
        try:
            touching = bool(
                self.check_actors_contact(self.cue.get_name(), self.primary_ball.get_name())
            )
        except Exception:
            touching = False
        # Require actual contact, or tip pressed into the contact gap (no early telekinesis).
        if not touching and dist > contact_thresh:
            return
        rigid = self._primary_rigid
        if rigid is None:
            return
        # Prefer stick pointing dir; fall back to aim if the tip frame is near-degenerate.
        stick = self._stick_tip_dir_xy()
        if float(np.dot(stick, self._aim_dir)) < 0.25:
            stick = np.array(self._aim_dir, dtype=np.float64)
        direction = np.array([stick[0], stick[1], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(direction))
        if n < 1e-6:
            return
        direction /= n
        try:
            rigid.set_linear_velocity(direction * self.strike_impulse)
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        self._strike_done = True
        self._strike_armed = False

    def _ball_inside_hollow(self, p):
        """True once the ball has fallen through a pocket into the hollow box."""
        cx, cy = self.table_cx, self.table_cy
        hl = self.table_half_len - self.WALL_T
        hw = self.table_half_wid - self.WALL_T
        in_xy = abs(float(p[0]) - cx) <= hl and abs(float(p[1]) - cy) <= hw
        # Below the lid underside ⇒ inside the cavity.
        return in_xy and float(p[2]) < self.base_top - 0.008

    def _nearest_pocket_id(self, xy):
        best_i, best_d = None, 1e9
        for i, pocket in enumerate(self._pocket_centers):
            d = float(np.linalg.norm(xy - pocket[:2]))
            if d < best_d:
                best_d, best_i = d, i
        return best_i, best_d

    def _check_and_sink_pockets(self):
        if self._primary_pocketed:
            return
        balls = [(self.primary_ball, True)] + [(b, False) for b in self.extra_balls]
        for ball, is_primary in balls:
            if ball is None:
                continue
            p = np.asarray(ball.get_pose().p, dtype=np.float64)
            # Natural fall through a real pocket opening into the hollow interior.
            if self._ball_inside_hollow(p):
                if is_primary:
                    pid, _ = self._nearest_pocket_id(p[:2])
                    self._primary_pocket_id = pid
                    self._primary_pocketed = True
                self._rest_ball_on_floor(ball, p)
                continue
            for i, pocket in enumerate(self._pocket_centers):
                if np.linalg.norm(p[:2] - pocket[:2]) <= self.pocket_radius:
                    # Nudge through the opening so physics can finish the drop.
                    if float(p[2]) > self.base_top - 0.002:
                        self._drop_ball_through_pocket(ball, pocket)
                    if is_primary and self._ball_inside_hollow(
                        np.asarray(ball.get_pose().p, dtype=np.float64)
                    ):
                        self._primary_pocket_id = i
                        self._primary_pocketed = True
                    break

    def _drop_ball_through_pocket(self, ball, pocket):
        """Release the ball into the hollow cavity under the pocket."""
        drop_pos = [
            float(pocket[0]),
            float(pocket[1]),
            float(self.base_top - self.ball_radius - 0.005),
        ]
        pose = sapien.Pose(drop_pos, [1, 0, 0, 0])
        ball.set_pose(pose)
        rigid = self._get_rigid_entity(ball)
        if rigid is not None:
            try:
                rigid.set_kinematic(False)
                rigid.set_disable_gravity(False)
                rigid.set_linear_velocity([0.0, 0.0, -0.4])
                rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _rest_ball_on_floor(self, ball, p):
        """Settle a pocketed ball on the inner floor (keep it visible inside)."""
        rest_z = self.floor_top + self.ball_radius + 0.001
        pose = sapien.Pose([float(p[0]), float(p[1]), rest_z], [1, 0, 0, 0])
        ball.set_pose(pose)
        rigid = self._get_rigid_entity(ball)
        if rigid is not None:
            try:
                rigid.set_kinematic(True)
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._update_welded_cue()
        self._check_robot_ball_contact()
        self._try_apply_strike_impulse()
        self._check_and_sink_pockets()

    def _dwell(self, steps):
        for i in range(max(0, int(steps))):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and i % self.save_freq == 0:
                self._take_picture()
            if self._primary_pocketed or self._robot_ball_contact:
                break

    def _dbg(self, tag):
        import os
        if os.environ.get("PLAY_BILLIARD_DEBUG"):
            tip = np.asarray(self.cue.get_functional_point(0, "list")[:3])
            ball = np.asarray(self.primary_ball.get_pose().p)
            print(
                f"[PLAY_BILLIARD] {tag}: plan={self.plan_success} "
                f"tip={np.round(tip, 3).tolist()} ball={np.round(ball, 3).tolist()} "
                f"pocketed={self._primary_pocketed} robot_touch={self._robot_ball_contact}",
                flush=True,
            )

    def _tip_xyz(self):
        return np.asarray(self.cue.get_functional_point(0, "list")[:3], dtype=np.float64)

    def _move_tip_z_to(self, arm, target_z, max_step=0.08):
        """Raise/lower the tip in world Z (chunked) so it clears the hollow table."""
        for _ in range(4):
            tip = self._tip_xyz()
            dz = float(target_z - tip[2])
            if abs(dz) < 0.008:
                return True
            step = float(np.clip(dz, -max_step, max_step))
            self.move(
                self.move_by_displacement(arm_tag=arm, z=step, move_axis="world")
            )
            if not self.plan_success:
                return False
        return abs(float(target_z - self._tip_xyz()[2])) < 0.015

    # ------------------------------------------------------------------ policy
    def play_once(self):
        arm = ArmTag(self._arm_side)
        hover_z = float(self.felt_top + self.HOVER_CLEARANCE)

        # 1) Approach the cue with open jaws, seat shaft between fingers, then clamp.
        self.move(self.open_gripper(arm, pos=1.0))
        self.move(
            self.grasp_actor(
                self.cue,
                arm_tag=arm,
                pre_grasp_dis=0.10,
                contact_point_id=0,
                gripper_pos=1.0,  # stay open; seating places the shaft in the aperture
            )
        )
        self._dbg("after_grasp")
        if not self.plan_success:
            self.info["info"] = {
                "{A}": "cue stick",
                "{B}": "red ball",
                "{a}": str(arm),
                "{C}": self._target_pocket_name.replace("_", " "),
            }
            return self.info
        self._seat_and_weld_cue(arm)
        self.move(self.close_gripper(arm, pos=0.0))
        # Re-seat after close so a small EE settle cannot leave the shaft off-center.
        self._seat_and_weld_cue(arm)
        self._dbg("after_seat")

        # Retreat slightly in -Y then lift high BEFORE any approach.
        table_near_y = self.table_cy - self.table_half_wid
        tip = self._tip_xyz()
        if tip[1] > table_near_y - 0.04:
            self.move(
                self.move_by_displacement(
                    arm_tag=arm,
                    y=float((table_near_y - 0.06) - tip[1]),
                    move_axis="world",
                )
            )
        self._move_tip_z_to(arm, hover_z, max_step=0.09)
        self._dbg("after_lift")

        # 2) Choose aim pocket and point the stick at it.
        pocket, aim_pid = self._choose_pocket()
        # Keep episode goal for top_three; for any-mode record what we aimed at.
        if self.pocket_mode != "top_three":
            self._target_pocket = pocket
            self._target_pocket_id = aim_pid
            self._target_pocket_name = self.POCKET_NAMES[aim_pid]
        ball_xy = self._ball_xy(self.primary_ball)
        aim = pocket[:2] - ball_xy
        aim_n = float(np.linalg.norm(aim))
        if aim_n < 1e-6:
            aim = np.array([0.0, 1.0])
            aim_n = 1.0
        self._aim_dir = aim / aim_n

        tip = self._tip_xyz()
        cue_p = np.asarray(self.cue.get_pose().p, dtype=np.float64)
        tip_dir_xy = tip[:2] - cue_p[:2]
        if float(np.linalg.norm(tip_dir_xy)) < 1e-4:
            tip_ang = 0.0
        else:
            tip_ang = float(np.arctan2(tip_dir_xy[1], tip_dir_xy[0]))
        aim_ang = float(np.arctan2(self._aim_dir[1], self._aim_dir[0]))
        yaw_delta = (aim_ang - tip_ang + np.pi) % (2.0 * np.pi) - np.pi
        yaw_delta = float(np.clip(yaw_delta, -np.deg2rad(120), np.deg2rad(120)))
        cur_q = np.array(self.get_arm_pose(str(arm))[3:], dtype=np.float64)
        yaw_q = t3d.quaternions.axangle2quat([0, 0, 1], yaw_delta)
        new_q = t3d.quaternions.qmult(yaw_q, cur_q)
        tip_z = float(self._tip_xyz()[2])
        self.move(
            self.move_by_displacement(
                arm_tag=arm,
                z=float(hover_z - tip_z),
                quat=list(new_q),
                move_axis="world",
            )
        )
        self._dbg("after_yaw")
        if self.plan_success:
            self._move_tip_z_to(arm, hover_z)

        # 3) Tip behind the ball along the aim line (hover → XY → lower).
        behind = ball_xy - self._aim_dir * self.APPROACH_GAP
        safe_y = float(table_near_y - 0.05)
        waypoints = [
            np.array([float(behind[0]), safe_y], dtype=np.float64),
            np.array(
                [float(behind[0]), float(min(behind[1], table_near_y + 0.01))],
                dtype=np.float64,
            ),
            behind,
        ]
        for wp in waypoints:
            for _ in range(8):
                tip = self._tip_xyz()
                if tip[2] < hover_z - 0.015:
                    if not self._move_tip_z_to(arm, hover_z):
                        break
                    tip = self._tip_xyz()
                dx = float(wp[0] - tip[0])
                dy = float(wp[1] - tip[1])
                if abs(dx) < 0.010 and abs(dy) < 0.010:
                    break
                step = 0.055
                scale = min(1.0, step / max(np.hypot(dx, dy), 1e-6))
                dz_keep = float(max(0.0, hover_z - tip[2]))
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm,
                        x=dx * scale,
                        y=dy * scale,
                        z=dz_keep,
                        move_axis="world",
                    )
                )
                if not self.plan_success:
                    break
            if not self.plan_success:
                break
        self._dbg("after_align_xy")

        # Refresh ball pose, then lower and home tip onto the contact point.
        ball_xy = self._ball_xy(self.primary_ball)
        contact_gap = self.ball_radius + self.CUE_RADIUS + 0.002
        contact_xy = ball_xy - self._aim_dir * contact_gap
        behind = ball_xy - self._aim_dir * self.APPROACH_GAP

        def _home_tip_xy(target_xy, tol=0.008, hops=5):
            for _ in range(hops):
                tip = self._tip_xyz()
                dx = float(target_xy[0] - tip[0])
                dy = float(target_xy[1] - tip[1])
                if abs(dx) < tol and abs(dy) < tol:
                    return True
                step = 0.04
                scale = min(1.0, step / max(np.hypot(dx, dy), 1e-6))
                self.move(
                    self.move_by_displacement(
                        arm_tag=arm, x=dx * scale, y=dy * scale, move_axis="world"
                    )
                )
                if not self.plan_success:
                    return False
            tip = self._tip_xyz()
            return abs(float(target_xy[0] - tip[0])) < 0.015 and abs(
                float(target_xy[1] - tip[1])
            ) < 0.015

        if self.plan_success:
            _home_tip_xy(behind, tol=0.010)
        if self.plan_success:
            self._move_tip_z_to(arm, float(self.ball_z), max_step=0.06)
        if self.plan_success:
            _home_tip_xy(contact_xy, tol=0.006, hops=6)
        self._dbg("after_align_z")

        # Strike: push through the ball along aim. Impulse only after tip–ball contact.
        self._strike_armed = True
        self._strike_done = False
        for _ in range(5):
            if self._strike_done or self._primary_pocketed or not self.plan_success:
                break
            tip = self._tip_xyz()
            ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
            # Correct lateral error relative to the aim line, then advance.
            to_ball = ball_p[:2] - tip[:2]
            along = float(np.dot(to_ball, self._aim_dir))
            lateral = to_ball - self._aim_dir * along
            corr = lateral * 0.6 + self._aim_dir * 0.010
            self.move(
                self.move_by_displacement(
                    arm_tag=arm, x=float(corr[0]), y=float(corr[1]), move_axis="world"
                )
            )
            self._dwell(12)
        self._dbg("after_strike")
        retract = -self._aim_dir * 0.04
        tip = self._tip_xyz()
        self.move(
            self.move_by_displacement(
                arm_tag=arm,
                x=float(retract[0]),
                y=float(retract[1]),
                z=float(max(0.04, hover_z - tip[2])),
                move_axis="world",
            )
        )
        self._dbg("after_retract")

        self._dwell(self.settle_steps)
        self._dbg("after_settle")

        self.info["info"] = {
            "{A}": "cue stick",
            "{B}": "red ball",
            "{a}": str(arm),
            "{C}": self._target_pocket_name.replace("_", " "),
        }
        return self.info

    # ------------------------------------------------------------------ success
    def check_success(self):
        if self._robot_ball_contact:
            return False
        if not self._primary_pocketed:
            if self.primary_ball is None:
                return False
            p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
            if not self._ball_inside_hollow(p):
                return False
            pid, _ = self._nearest_pocket_id(p[:2])
            self._primary_pocket_id = pid
            self._primary_pocketed = True
        # Must land in an allowed pocket for this episode's mode.
        if self._primary_pocket_id is None:
            return False
        return int(self._primary_pocket_id) in set(self._allowed_pocket_ids)

    def get_obs(self):
        obs = super().get_obs()
        if not getattr(self, "_loaded", False) or self.primary_ball is None:
            return obs
        ball_p = np.asarray(self.primary_ball.get_pose().p, dtype=np.float64)
        tip = (
            np.asarray(self.cue.get_functional_point(0, "list")[:3], dtype=np.float64)
            if self.cue is not None
            else np.zeros(3)
        )
        pocket = (
            self._target_pocket.tolist()
            if self._target_pocket is not None
            else [0.0, 0.0, 0.0]
        )
        obs["play_billiard"] = {
            "primary_ball": ball_p.tolist(),
            "cue_tip": tip.tolist(),
            "target_pocket": pocket,
            "target_pocket_name": self._target_pocket_name,
            "pocket_mode": self.pocket_mode,
            "blocker_mode": self.blocker_mode,
            "arm_side": self._arm_side,
            "aim_dir": self._aim_dir.tolist(),
            "primary_pocketed": float(self._primary_pocketed),
            "robot_ball_contact": float(self._robot_ball_contact),
            "num_extra_balls": float(len(self.extra_balls)),
            "strike_done": float(self._strike_done),
        }
        return obs


# NOTICE: Primitive-only task. Reserved object-id range [350, 359] documents the
# logical identities used in language templates (cue_stick/base350, primary_ball/base351,
# billiard_table/base352). Nothing is written under assets/objects/.
