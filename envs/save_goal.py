from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class save_goal(Base_Task):
    """Keep a moving ball out of the goal by placing a square blocker in time.

    A ball travels across the table toward a goal on one side. Layout can be mirrored
    across the table midline (x → −x): default puts the goal/keeper on +x (right gripper);
    ``mirrored: true`` puts them on −x (left gripper). With ``random_mirror: true`` (default)
    the side is chosen per episode when ``mirrored`` is unset.
    A red deadline line marks the last moment the robot is allowed to reposition the save_goal.
    A green placement area sits directly in front of the goal. The robot must grasp the square
    save_goal, place it fully inside that green area while the ball is still behind the red line,
    then release it before the ball reaches the goal mouth.     A save requires the ball to hit the
    keeper's front face; side grazes do not stop the ball. A soccer-style net bag
    (visual lattice) hangs behind the goal mouth when ``net_enabled`` is true.

    Task options (set in ``task_args.save_goal``; independent toggles):
      - Option 1 — field players (bounce): ``players_enabled``
        Spawn ``players_max`` static soccer-player meshes before the red line, outside
        the green zone, and no farther from the goal than
        ``player_max_goal_dist_mult`` × green-zone depth (default 2×). The ball aims at
        one player at random, "bounces", then continues toward the goal mouth.
        CLI: ``--task-arg players_enabled=true`` or legacy ``--option 1``.
      - Option 2 — field cover: ``cover_enabled``
        A partial tunnel/cover over the center of the approach field (ball drop → red
        line). Ball enters after a ≥10 cm opening on the drop side, travels under the
        cover, then exits with a ≥10% field-length opening before the red line.
        With Option 1, the cover ends before the field players.
        CLI: ``--task-arg cover_enabled=true`` or legacy ``--option 2``.
      - Mirror — ``mirrored`` / ``random_mirror``
        Flip the whole field to the left (−x) so the expert uses the left gripper.
        CLI: ``--task-arg mirrored=true``.
    """

    BALL_RADIUS_DEFAULT = 0.018
    # Nominal ball speed; each episode samples ±20% by default (scale 0.8–1.2).
    BALL_SPEED_DEFAULT = 0.05470  # -15% vs 0.06435; episode still samples ±20%
    BALL_SPEED_SCALE_MIN_DEFAULT = 0.8
    BALL_SPEED_SCALE_MAX_DEFAULT = 1.2
    BALL_START_X_DEFAULT = 0.24
    BALL_START_Y_JITTER_DEFAULT = 0.02
    BALL_GOAL_END_X_OFFSET_DEFAULT = 0.08
    BALL_TARGET_Y_MARGIN_DEFAULT = 0.02
    BALL_ANGLE_DEG_MIN_DEFAULT = -10.0
    BALL_ANGLE_DEG_MAX_DEFAULT = 10.0

    GOAL_X_DEFAULT = 0.20
    GOAL_CENTER_Y_DEFAULT = 0.10
    GOAL_CENTER_Y_JITTER_DEFAULT = 0.0
    GOAL_HALF_W_DEFAULT = 0.11
    GOAL_POST_T_DEFAULT = 0.01
    GOAL_POST_H_DEFAULT = 0.12
    GOAL_BAR_T_DEFAULT = 0.01
    GREEN_AREA_X_LEN_DEFAULT = 0.10
    GREEN_AREA_Y_EXTRA_DEFAULT = 0.05
    RED_LINE_X_DEFAULT = 0.2
    # Layout mirror across y-axis (x → −x): left gripper when mirrored.
    RANDOM_MIRROR_DEFAULT = True
    MIRRORED_DEFAULT = None  # None → sample when random_mirror, else use explicit bool

    KEEPER_X_DEFAULT = 0.16
    KEEPER_SPAWN_X_DEFAULT = 0.10
    KEEPER_SPAWN_Y_DEFAULT = -0.12
    KEEPER_GOAL_CLEARANCE_DEFAULT = 0.05
    KEEPER_POSE_TOL_DEFAULT = 0.03

    BALL_SETTLE_STEPS_DEFAULT = 120

    # Option 1 — field players the ball can bounce off
    PLAYERS_ENABLED_DEFAULT = False  # Option 1 toggle
    PLAYERS_MAX_DEFAULT = 2
    PLAYER_MODEL_DEFAULT = "223_soccer_player"
    # Footprint half-width matches the goalkeeper square (ball_radius); height from mesh.
    PLAYER_HALF_XY_DEFAULT = 0.018
    PLAYER_HALF_Z_DEFAULT = 0.0435
    PLAYER_CORRIDOR_MARGIN_DEFAULT = 0.04
    PLAYER_Y_SPREAD_DEFAULT = 0.20
    # Min clear gap (edge-to-edge) between players; center spacing adds footprint widths.
    PLAYER_SEPARATION_DEFAULT = 0.15
    # Max |goal_x − player_x| as a multiple of green_area_x_len (keep players near the goal).
    PLAYER_MAX_GOAL_DIST_MULT_DEFAULT = 2.0
    # Blend of "face partner" vs "face goal" when orienting players (0=partner only, 1=goal only).
    PLAYER_GOAL_FACE_BIAS_DEFAULT = 0.22
    PLAYER_COLORS = (
        (0.15, 0.40, 0.90),
        (0.90, 0.20, 0.18),
    )

    # Option 2 — partial field cover / tunnel over the approach corridor
    COVER_ENABLED_DEFAULT = False  # Option 2 toggle
    COVER_ENTRY_GAP_DEFAULT = 0.10       # m; min open gap on the ball-drop side
    COVER_EXIT_GAP_FRAC_DEFAULT = 0.10   # fraction of field length open before red line
    COVER_HALF_Y_DEFAULT = 0.12          # m; half-width of cover cavity about goal_center_y
    COVER_CLEARANCE_Z_DEFAULT = 0.12     # m; clear height under roof (fits ball + players)
    COVER_WALL_T_DEFAULT = 0.008         # m; side-wall thickness
    COVER_ROOF_T_DEFAULT = 0.006         # m; roof thickness
    COVER_LEN_MIN_DEFAULT = 0.10         # m; min cover length along travel (x)
    # Clear gap between cover exit and the nearest Opt-1 player (when both enabled).
    COVER_PLAYER_CLEARANCE_DEFAULT = 0.02
    COVER_COLOR = (0.42, 0.45, 0.50)

    # Soccer-style goal net (visual lattice behind the posts)
    NET_ENABLED_DEFAULT = True
    NET_DEPTH_DEFAULT = 0.09          # m; how far behind the goal mouth the net sits
    NET_CELL_DEFAULT = 0.028          # m; approx mesh spacing
    NET_STRAND_T_DEFAULT = 0.0022     # m; strand thickness
    NET_COLOR = (0.92, 0.93, 0.95)

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("save_goal", {})
        self._loaded = False
        self._ball_motion_active = False
        self._ball_step = 0
        self._ball_blocked = False
        self._goal_conceded = False
        self._late_failure = False
        self.goalkeeper = None
        self.ball = None
        self._ball_rigid = None
        self.goalkeeper_target_pose = None
        self.ball_start_pose = None
        self.ball_target_pose = None
        self.ball_bounce_pose = None
        self._ball_waypoints = None
        self._ball_seg_cum = None
        self._ball_path_len = 0.0
        self._players = []
        self._bounce_player_idx = -1
        self.players_enabled = False
        self.cover_enabled = False
        self._cover_parts = []
        self.cover_x_min = None
        self.cover_x_max = None
        self.cover_len = 0.0
        self._ball_crossed_goal = False
        self.green_area_x_min = 0.0
        self.green_area_x_max = 0.0
        self.green_area_y_min = 0.0
        self.green_area_y_max = 0.0
        self._keeper_deployed = False
        self._keeper_drop_pose = None
        super()._init_task_env_(**kwags)
        # Ball starts in play_once so grasp/place planning time does not burn the shot clock.
        self._ball_motion_active = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _as_bool(value, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        s = str(value).strip().lower()
        if s in ("1", "true", "yes", "on"):
            return True
        if s in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"save_goal expected a boolean, got {value!r}")

    def _parse_players_enabled(self, c) -> bool:
        """Option 1 toggle: ``players_enabled`` (preferred) or legacy ``option: 1``."""
        players = c.get("players_enabled", None)
        legacy = c.get("option", None)
        if legacy is not None and players is None:
            # option: 1 enables players; option: 2 is cover-only (handled separately).
            if legacy in (1, "1", "players_enabled", "players", "field_players"):
                players = True
            elif legacy in (2, "2", "cover_enabled", "cover", "field_cover"):
                players = False
            else:
                raise ValueError(
                    "save_goal option must be 1/players_enabled or 2/cover_enabled "
                    "(or set players_enabled / cover_enabled booleans)"
                )
        return self._as_bool(players, self.PLAYERS_ENABLED_DEFAULT)

    def _parse_cover_enabled(self, c) -> bool:
        """Option 2 toggle: ``cover_enabled`` (preferred) or legacy ``option: 2``."""
        cover = c.get("cover_enabled", None)
        legacy = c.get("option", None)
        if legacy is not None and cover is None:
            if legacy in (2, "2", "cover_enabled", "cover", "field_cover"):
                cover = True
            elif legacy in (1, "1", "players_enabled", "players", "field_players"):
                cover = False
            else:
                raise ValueError(
                    "save_goal option must be 1/players_enabled or 2/cover_enabled "
                    "(or set players_enabled / cover_enabled booleans)"
                )
        return self._as_bool(cover, self.COVER_ENABLED_DEFAULT)

    def _parse_mirrored(self, c) -> bool:
        """Layout mirror (x → −x): explicit ``mirrored``, else random when ``random_mirror``."""
        random_mirror = bool(c.get("random_mirror", self.RANDOM_MIRROR_DEFAULT))
        mirror_cfg = c.get("mirrored", self.MIRRORED_DEFAULT)
        if mirror_cfg is None:
            return bool(random_mirror and (np.random.rand() < 0.5))
        return self._as_bool(mirror_cfg, False)

    def _option_label(self) -> str:
        """Caption token for active options (Option 1 / Option 2)."""
        parts = []
        if getattr(self, "players_enabled", False):
            parts.append("option 1")
        if getattr(self, "cover_enabled", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "baseline"

    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for comp in obj.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                return comp
        return None

    def _keeper_xy_bounds(self):
        if getattr(self, "goalkeeper", None) is None:
            return None
        pose_m = self.goalkeeper.get_pose().to_transformation_matrix()
        local_corners = np.array([
            [-self.keeper_half_x, -self.keeper_half_y, 0.0, 1.0],
            [-self.keeper_half_x,  self.keeper_half_y, 0.0, 1.0],
            [ self.keeper_half_x, -self.keeper_half_y, 0.0, 1.0],
            [ self.keeper_half_x,  self.keeper_half_y, 0.0, 1.0],
        ], dtype=np.float64)
        world_corners = (pose_m @ local_corners.T).T
        return (
            float(np.min(world_corners[:, 0])),
            float(np.max(world_corners[:, 0])),
            float(np.min(world_corners[:, 1])),
            float(np.max(world_corners[:, 1])),
        )

    def _ball_path_y_at_x(self, x: float) -> float:
        """Y of the ball's final approach segment (post-bounce if Opt 1 is on) at table x."""
        if self.ball_bounce_pose is not None:
            start = np.asarray(self.ball_bounce_pose, dtype=np.float64)
        else:
            start = np.asarray(self.ball_start_pose, dtype=np.float64)
        end = np.asarray(self.ball_target_pose, dtype=np.float64)
        start_x = float(start[0])
        end_x = float(end[0])
        if abs(end_x - start_x) < 1e-8:
            return float(start[1])
        t = float((x - start_x) / (end_x - start_x))
        return float(start[1] + t * (end[1] - start[1]))

    def _set_ball_waypoints(self, waypoints):
        pts = [np.asarray(p, dtype=np.float64).copy() for p in waypoints]
        self._ball_waypoints = pts
        cum = [0.0]
        for i in range(len(pts) - 1):
            cum.append(cum[-1] + float(np.linalg.norm(pts[i + 1] - pts[i])))
        self._ball_seg_cum = cum
        self._ball_path_len = float(cum[-1])

    def _ball_pos_at_progress(self, progress: float):
        progress = float(np.clip(progress, 0.0, 1.0))
        if self._ball_waypoints is None or self._ball_path_len < 1e-8:
            return self.ball_start_pose + (self.ball_target_pose - self.ball_start_pose) * progress
        dist = progress * self._ball_path_len
        cum = self._ball_seg_cum
        pts = self._ball_waypoints
        for i in range(len(pts) - 1):
            if dist <= cum[i + 1] + 1e-9:
                seg_len = cum[i + 1] - cum[i]
                t = 0.0 if seg_len < 1e-8 else (dist - cum[i]) / seg_len
                return pts[i] + (pts[i + 1] - pts[i]) * t
        return pts[-1].copy()

    def _spawn_field_players(self, start_x: float, start_y: float, goal_end_x: float):
        """Opt 1: place players before the red line, near the goal (≤2× green depth)."""
        self._players = []
        self._bounce_player_idx = -1
        self.ball_bounce_pose = None
        if not self.players_enabled:
            return

        # Allowed band along x (distance from goal):
        #   - before the red line and outside the green zone (near-goal bound)
        #   - no farther than player_max_goal_dist_mult × green_area_x_len (far bound)
        green_len = float(self.green_area_x_len)
        red_offset = float(abs(self.goal_x - self.red_line_x))
        min_goal_dist = float(max(green_len, red_offset) + self.player_half_xy)
        max_goal_dist = float(
            max(self.player_max_goal_dist_mult, 1.0) * green_len - self.player_half_xy
        )
        if max_goal_dist <= min_goal_dist + 1e-4:
            # Degenerate config: keep a thin band just outside the near-goal bound.
            max_goal_dist = float(min_goal_dist + max(2.0 * self.player_half_xy, 0.02))

        near_goal_x = float(self.goal_x - self.travel_dir * min_goal_dist)
        far_goal_x = float(self.goal_x - self.travel_dir * max_goal_dist)
        # Also stay clear of the ball drop.
        margin = self.player_corridor_margin + self.player_half_xy
        start_side = float(start_x) + self.travel_dir * margin
        if self.travel_dir > 0.0:
            lo = max(far_goal_x, start_side)
            hi = near_goal_x
        else:
            lo = near_goal_x
            hi = min(far_goal_x, start_side)
        if hi <= lo + 1e-4:
            return

        # Separation is a clear gap; convert to min center-to-center distance.
        min_center_dist = float(self.player_separation + 2.0 * self.player_half_xy)
        n = int(self.players_max)  # always fill the configured count when Opt 1 is on
        cy = float(self.goal_center_y)
        y_lo = float(cy - self.player_y_spread)
        y_hi = float(cy + self.player_y_spread)
        placed = []

        def _rand_x():
            return float(np.random.uniform(lo, hi))

        # Two players: one strictly above the goal centerline, one strictly below; x/y randomized.
        if n == 2 and y_hi > cy and y_lo < cy:
            center_margin = 1e-3
            above_lo = cy + center_margin
            below_hi = cy - center_margin
            for _ in range(96):
                px_a, py_a = _rand_x(), float(np.random.uniform(above_lo, y_hi))
                px_b, py_b = _rand_x(), float(np.random.uniform(y_lo, below_hi))
                if np.hypot(px_a - px_b, py_a - py_b) >= min_center_dist:
                    placed = [(px_a, py_a), (px_b, py_b)]
                    break
            else:
                # Fallback: still split across the centerline with the min gap.
                half_gap = 0.5 * min_center_dist
                placed = [
                    (_rand_x(), float(np.clip(cy + half_gap, above_lo, y_hi))),
                    (_rand_x(), float(np.clip(cy - half_gap, y_lo, below_hi))),
                ]
            # Randomize spawn order so neither side is privileged.
            if np.random.rand() < 0.5:
                placed.reverse()
        else:
            for i in range(n):
                # Alternate above / below the goal centerline when possible.
                if y_hi > cy and y_lo < cy:
                    if i % 2 == 0:
                        y_a, y_b = cy + 1e-3, y_hi
                    else:
                        y_a, y_b = y_lo, cy - 1e-3
                else:
                    y_a, y_b = y_lo, y_hi
                for _ in range(64):
                    px = _rand_x()
                    py = float(np.random.uniform(y_a, y_b))
                    if all(np.hypot(px - qx, py - qy) >= min_center_dist for qx, qy in placed):
                        placed.append((px, py))
                        break
                else:
                    side = 1.0 if (i % 2 == 0) else -1.0
                    py = float(np.clip(
                        cy + side * (0.5 * min_center_dist + 0.02 * (i // 2)),
                        y_lo,
                        y_hi,
                    ))
                    placed.append((_rand_x(), py))

        # Mesh feet sit at the actor origin (tiny sink so soles meet the table).
        # Boxes are centered on pose.
        mesh_z = float(self.table_top_z - 0.002)
        box_z = float(self.table_top_z + self.player_half_z)
        mid_y = float(np.mean([p[1] for p in placed])) if placed else float(self.goal_center_y)

        for i, (px, py) in enumerate(placed):
            # Face each other (toward partner) and slightly toward the goal.
            # 223_soccer_player faces local -Y at identity (head protrudes in -Y).
            if len(placed) >= 2:
                others = [p for j, p in enumerate(placed) if j != i]
                partner = min(others, key=lambda p: (p[0] - px) ** 2 + (p[1] - py) ** 2)
                toward = np.array([partner[0] - px, partner[1] - py], dtype=np.float64)
            else:
                toward = np.array([0.0, mid_y - py], dtype=np.float64)
            if float(np.linalg.norm(toward)) < 1e-6:
                toward = np.array([0.0, 1.0], dtype=np.float64)
            toward /= float(np.linalg.norm(toward))
            bias = float(np.clip(self.player_goal_face_bias, 0.0, 1.0))
            face = (1.0 - bias) * toward + bias * np.array([self.travel_dir, 0.0], dtype=np.float64)
            if float(np.linalg.norm(face)) < 1e-6:
                face = toward
            face /= float(np.linalg.norm(face))
            # Map local -Y → face: R_z(yaw) @ (0,-1) = (sin yaw, -cos yaw) == face
            # => yaw = atan2(face_x, -face_y)
            yaw = float(np.arctan2(face[0], -face[1]))
            qz = np.sin(yaw * 0.5)
            qw = np.cos(yaw * 0.5)
            player = create_actor(
                self,
                pose=sapien.Pose(
                    [px, py, mesh_z],
                    [qw, 0.0, 0.0, qz],
                ),
                modelname=self.player_model,
                model_id=0,
                convex=True,
                is_static=True,
            )
            if player is None:
                # Fallback primitive if the mesh asset is missing.
                color = self.PLAYER_COLORS[i % len(self.PLAYER_COLORS)]
                player = create_box(
                    self,
                    pose=sapien.Pose(
                        [px, py, box_z],
                        [qw, 0.0, 0.0, qz],
                    ),
                    half_size=[self.player_half_xy, self.player_half_xy, self.player_half_z],
                    color=color,
                    is_static=True,
                    name=f"field_player_{i}",
                )
            else:
                try:
                    player.set_name(f"field_player_{i}")
                except Exception:
                    pass
            self._players.append(player)
            self.add_prohibit_area(player, padding=0.02)

        self._bounce_player_idx = int(np.random.randint(0, len(self._players)))
        bx, by = placed[self._bounce_player_idx]
        # Hit point on the ball-facing side of the chosen player.
        to_player = np.array([bx - start_x, by - start_y], dtype=np.float64)
        dist_xy = float(np.linalg.norm(to_player))
        if dist_xy < 1e-6:
            to_player = np.array([self.travel_dir, 0.0], dtype=np.float64)
            dist_xy = 1.0
        inward = to_player / dist_xy
        hit_xy = np.array([bx, by], dtype=np.float64) - inward * (self.player_half_xy + self.ball_radius + 0.002)
        self.ball_bounce_pose = np.array(
            [hit_xy[0], hit_xy[1], self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        # Prefer a post-bounce end_y that still enters the goal from the bounce point.
        for _ in range(64):
            end_y = float(np.random.uniform(
                self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
                self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
            ))
            # Reject paths that clip the distractor players.
            ok = True
            for j, (px, py) in enumerate(placed):
                if j == self._bounce_player_idx:
                    continue
                if self._segment_clear_of_disk(
                    hit_xy, np.array([goal_end_x, end_y]), np.array([px, py]),
                    self.player_half_xy + self.ball_radius + 0.01,
                ):
                    continue
                ok = False
                break
            if ok:
                self._bounce_end_y = end_y
                return
        self._bounce_end_y = float(np.clip(
            by,
            self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
            self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
        ))

    @staticmethod
    def _segment_clear_of_disk(a_xy, b_xy, c_xy, radius: float) -> bool:
        a = np.asarray(a_xy, dtype=np.float64)
        b = np.asarray(b_xy, dtype=np.float64)
        c = np.asarray(c_xy, dtype=np.float64)
        ab = b - a
        ab2 = float(np.dot(ab, ab))
        if ab2 < 1e-12:
            return float(np.linalg.norm(c - a)) >= radius
        t = float(np.clip(np.dot(c - a, ab) / ab2, 0.0, 1.0))
        closest = a + t * ab
        return float(np.linalg.norm(c - closest)) >= radius

    def _spawn_field_cover(self, start_x: float):
        """Option 2: partial tunnel over the mid approach field (ball drop → red line).

        Leaves ≥ ``cover_entry_gap`` open on the ball-drop side and ≥
        ``cover_exit_gap_frac`` of field length open before the red line.
        Cover length is sampled uniformly in ``[cover_len_min, max_spannable]``
        (default min 10 cm; max = full mid region after those gaps).
        With Option 1, the cover also ends before the nearest field player.
        """
        self._cover_parts = []
        self.cover_x_min = None
        self.cover_x_max = None
        self.cover_len = 0.0
        if not self.cover_enabled:
            return

        field_len = float(abs(self.red_line_x - start_x))
        if field_len < 1e-4:
            return
        entry_gap = float(max(self.cover_entry_gap, 0.0))
        exit_gap = float(max(self.cover_exit_gap_frac, 0.0) * field_len)
        # Region the cover is allowed to occupy (already excludes the required openings).
        region_start = float(start_x + self.travel_dir * entry_gap)
        region_end = float(self.red_line_x - self.travel_dir * exit_gap)

        # Option 1+2: end the cover before the nearest player along the ball path.
        player_xs = []
        for p in getattr(self, "_players", []) or []:
            try:
                player_xs.append(float(p.get_pose().p[0]))
            except Exception:
                continue
        if player_xs:
            clearance = float(
                self.cover_player_clearance
                + getattr(self, "player_half_xy", self.PLAYER_HALF_XY_DEFAULT)
            )
            if self.travel_dir > 0.0:
                first_player_x = float(min(player_xs))
                player_end = first_player_x - clearance
                region_end = float(min(region_end, player_end))
            else:
                first_player_x = float(max(player_xs))
                player_end = first_player_x + clearance
                region_end = float(max(region_end, player_end))

        span = float(self.travel_dir * (region_end - region_start))
        len_min = float(max(self.cover_len_min, 0.0))
        if span < len_min - 1e-9:
            # Not enough room for the min cover length after mandated gaps / players.
            return

        # Random length in [10 cm, max spannable]; then randomize placement in the mid region.
        cover_len = float(np.random.uniform(len_min, span))
        slack = float(span - cover_len)
        offset = float(np.random.uniform(0.0, slack)) if slack > 1e-9 else 0.0
        cover_a = float(region_start + self.travel_dir * offset)
        cover_b = float(cover_a + self.travel_dir * cover_len)
        x_min = float(min(cover_a, cover_b))
        x_max = float(max(cover_a, cover_b))
        self.cover_x_min = x_min
        self.cover_x_max = x_max
        self.cover_len = cover_len

        cx = 0.5 * (x_min + x_max)
        half_x = 0.5 * (x_max - x_min)
        half_y = float(self.cover_half_y)
        wall_t = float(self.cover_wall_t)
        roof_t = float(self.cover_roof_t)
        clear_z = float(self.cover_clearance_z)
        cy = float(self.goal_center_y)
        color = self.COVER_COLOR

        # Side walls (leave cavity open along travel for the ball).
        wall_z = self.table_top_z + 0.5 * clear_z
        for sign, tag in ((-1.0, "neg"), (1.0, "pos")):
            wy = cy + sign * (half_y + 0.5 * wall_t)
            part = create_box(
                self,
                pose=sapien.Pose([cx, wy, wall_z], [1, 0, 0, 0]),
                half_size=[half_x, 0.5 * wall_t, 0.5 * clear_z],
                color=color,
                is_static=True,
                name=f"field_cover_wall_{tag}",
            )
            self._cover_parts.append(part)

        # Roof slab.
        roof_z = self.table_top_z + clear_z + 0.5 * roof_t
        roof = create_box(
            self,
            pose=sapien.Pose([cx, cy, roof_z], [1, 0, 0, 0]),
            half_size=[half_x, half_y + wall_t, 0.5 * roof_t],
            color=color,
            is_static=True,
            name="field_cover_roof",
        )
        self._cover_parts.append(roof)
        # Cover is a field obstacle for the ball path only; do not mark prohibit
        # areas (that blocks expert arm plans across the corridor).

    def _spawn_goal_net(self):
        """Visual soccer-style net bag behind the goal mouth (back + sides + top)."""
        self._net_parts = []
        if not getattr(self, "net_enabled", True):
            return

        depth = float(max(self.net_depth, 0.02))
        cell = float(max(self.net_cell, 0.015))
        half_t = 0.5 * float(max(self.net_strand_t, 0.001))
        cy = float(self.goal_center_y)
        yw = float(self.goal_half_w)
        h = float(self.goal_post_h)
        z0 = float(self.table_top_z)
        gx = float(self.goal_x)
        d = float(self.travel_dir)
        color = self.NET_COLOR
        back_x = float(gx + d * depth)
        mid_x = float(gx + d * (0.5 * depth))
        half_depth = 0.5 * depth

        n_y = max(3, int(round((2.0 * yw) / cell)) + 1)
        n_z = max(3, int(round(h / cell)) + 1)
        n_x = max(3, int(round(depth / cell)) + 1)
        ys = np.linspace(cy - yw, cy + yw, n_y)
        zs = np.linspace(z0 + half_t, z0 + h - half_t, n_z)
        xs = np.linspace(gx, back_x, n_x)

        def _strand(pose, half_size, name):
            part = create_visual_box(
                self,
                pose=pose,
                half_size=half_size,
                color=color,
                name=name,
            )
            self._net_parts.append(part)

        # Back panel (plane behind the mouth).
        for i, y in enumerate(ys):
            _strand(
                sapien.Pose([back_x, float(y), z0 + 0.5 * h], [1, 0, 0, 0]),
                [half_t, half_t, 0.5 * h],
                f"goal_net_back_v_{i}",
            )
        for i, z in enumerate(zs):
            _strand(
                sapien.Pose([back_x, cy, float(z)], [1, 0, 0, 0]),
                [half_t, yw, half_t],
                f"goal_net_back_h_{i}",
            )

        # Side panels (connect posts to the back corners).
        for side, y_side in (("neg", cy - yw), ("pos", cy + yw)):
            for i, x in enumerate(xs):
                _strand(
                    sapien.Pose([float(x), float(y_side), z0 + 0.5 * h], [1, 0, 0, 0]),
                    [half_t, half_t, 0.5 * h],
                    f"goal_net_side_{side}_v_{i}",
                )
            for i, z in enumerate(zs):
                _strand(
                    sapien.Pose([mid_x, float(y_side), float(z)], [1, 0, 0, 0]),
                    [half_depth, half_t, half_t],
                    f"goal_net_side_{side}_h_{i}",
                )

        # Top panel (under / along the crossbar plane, stretching back).
        top_z = float(z0 + h - half_t)
        for i, x in enumerate(xs):
            _strand(
                sapien.Pose([float(x), cy, top_z], [1, 0, 0, 0]),
                [half_t, yw, half_t],
                f"goal_net_top_y_{i}",
            )
        for i, y in enumerate(ys):
            _strand(
                sapien.Pose([mid_x, float(y), top_z], [1, 0, 0, 0]),
                [half_depth, half_t, half_t],
                f"goal_net_top_x_{i}",
            )

    def _keeper_in_zone(self):
        if getattr(self, "goalkeeper", None) is None:
            return False
        bounds = self._keeper_xy_bounds()
        if bounds is None:
            return False
        x_min, x_max, y_min, y_max = bounds
        keeper_z = float(self.goalkeeper.get_pose().p[2])
        target_z = float(self.table_top_z + self.keeper_half_z)
        z_ok = abs(keeper_z - target_z) <= 0.02
        return bool(
            x_min >= (self.green_area_x_min - 1e-4)
            and x_max <= (self.green_area_x_max + 1e-4)
            and y_min >= (self.green_area_y_min - 1e-4)
            and y_max <= (self.green_area_y_max + 1e-4)
            and z_ok
        )

    def _wait_for_outcome(self):
        max_steps = int(self.ball_total_steps + self.ball_settle_steps)
        for i in range(max(0, max_steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    def _place_keeper_from_top(self, arm_tag: ArmTag):
        if getattr(self, "goalkeeper", None) is None or self.goalkeeper_target_pose is None:
            return

        target_p = np.asarray(self.goalkeeper_target_pose.p, dtype=np.float64)
        frame_clearance_z = self.table_top_z + max(
            self.goal_post_h + self.keeper_half_z + 0.08,
            self.keeper_half_z + 0.18,
        )

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        lift_dz = float(frame_clearance_z - keeper_p[2])
        if lift_dz > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=lift_dz))

        # Approach over the goal, then iteratively correct XY so the *cube*
        # (not just the EE) lands on the intercept before release.
        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        goal_over_x = float(self.goal_x - self.travel_dir * (self.keeper_half_x + 0.01))
        goal_over_y = float(self.goal_center_y)
        dx_goal = float(goal_over_x - keeper_p[0])
        dy_goal = float(goal_over_y - keeper_p[1])
        if abs(dx_goal) > 1e-4 or abs(dy_goal) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx_goal, y=dy_goal))

        for _ in range(4):
            keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
            dx = float(target_p[0] - keeper_p[0])
            dy = float(target_p[1] - keeper_p[1])
            if abs(dx) < 0.003 and abs(dy) < 0.003:
                break
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))

        keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
        place_dz = float(target_p[2] - keeper_p[2])
        if abs(place_dz) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=place_dz))

        # Final XY polish at table height (grasp offset often shows up here).
        for _ in range(2):
            keeper_p = np.asarray(self.goalkeeper.get_pose().p, dtype=np.float64)
            dx = float(target_p[0] - keeper_p[0])
            dy = float(target_p[1] - keeper_p[1])
            if abs(dx) < 0.002 and abs(dy) < 0.002:
                break
            self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))

        self.move(self.open_gripper(arm_tag))
        for _ in range(8):
            self._update_kinematic_tasks()
            self.scene.step()
        # Freeze exactly where released — never snap/teleport to another XY.
        self._freeze_keeper_in_place()
        self._keeper_deployed = True

    def _retreat_arm_home(self, arm_tag: ArmTag, lift_z: float = 0.08):
        """After releasing the keeper, clear the goal then return the arm home.

        Forces ``plan_success`` so a failed place step cannot skip the retreat.
        A failed home plan does not invalidate an already-deployed save.
        """
        deployed = bool(getattr(self, "_keeper_deployed", False))
        self.plan_success = True
        if lift_z and abs(float(lift_z)) > 1e-4:
            self.move(self.move_by_displacement(arm_tag=arm_tag, z=float(lift_z), move_axis="arm"))
            self.plan_success = True
        self.move(self.back_to_origin(arm_tag))
        if deployed:
            self.plan_success = True

    def _freeze_keeper_in_place(self):
        """Hold the square at its current pose after the drop (never snap elsewhere)."""
        if getattr(self, "goalkeeper", None) is None:
            return
        pose = self.goalkeeper.get_pose()
        p = np.asarray(pose.p, dtype=np.float64).copy()
        # Seat on the table only; keep the dropped XY / orientation exactly.
        p[2] = float(self.table_top_z + self.keeper_half_z)
        freeze_pose = sapien.Pose(p.tolist(), list(pose.q))
        # Only adjust Z if we are seating; skip set_pose when already on the table
        # within tolerance so we do not flash a visible jump.
        if abs(float(pose.p[2]) - p[2]) > 1e-4:
            try:
                self.goalkeeper.set_pose(freeze_pose)
            except Exception:
                try:
                    self.goalkeeper.actor.set_pose(freeze_pose)
                except Exception:
                    freeze_pose = pose
        else:
            freeze_pose = pose
        self._keeper_drop_pose = freeze_pose
        rigid = self._get_rigid(self.goalkeeper)
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        try:
            rigid.set_kinematic(True)
        except Exception:
            pass

    def _hold_keeper_kinematic(self):
        """Keep the dropped square fixed without changing its pose."""
        rigid = self._get_rigid(getattr(self, "goalkeeper", None)) if getattr(self, "goalkeeper", None) is not None else None
        if rigid is None:
            return
        try:
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        try:
            rigid.set_kinematic(True)
        except Exception:
            pass

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.table_top_z = 0.74 + self.table_z_bias

        self.goal_x_abs = float(c.get("goal_x", c.get("goal_y", self.GOAL_X_DEFAULT)))
        self.goal_center_y = float(c.get("goal_center_y", self.GOAL_CENTER_Y_DEFAULT))
        goal_center_y_jitter = float(c.get("goal_center_y_jitter", self.GOAL_CENTER_Y_JITTER_DEFAULT))
        self.goal_center_y = float(np.random.uniform(
            self.goal_center_y - goal_center_y_jitter,
            self.goal_center_y + goal_center_y_jitter,
        ))
        self.goal_half_w = float(c.get("goal_half_w", self.GOAL_HALF_W_DEFAULT))
        self.goal_post_t = float(c.get("goal_post_t", self.GOAL_POST_T_DEFAULT))
        self.goal_post_h = float(c.get("goal_post_h", self.GOAL_POST_H_DEFAULT))
        self.goal_bar_t = float(c.get("goal_bar_t", self.GOAL_BAR_T_DEFAULT))
        self.green_area_x_len = float(c.get("green_area_x_len", self.GREEN_AREA_X_LEN_DEFAULT))
        self.green_area_y_extra = float(c.get("green_area_y_extra", self.GREEN_AREA_Y_EXTRA_DEFAULT))
        self.red_line_goal_offset = float(c.get("red_line_x", c.get("red_line_y", self.RED_LINE_X_DEFAULT)))

        self.keeper_x_abs = float(c.get("keeper_x", c.get("keeper_y", self.KEEPER_X_DEFAULT)))
        self.keeper_pose_tol = float(c.get("keeper_pose_tol", self.KEEPER_POSE_TOL_DEFAULT))
        self.keeper_spawn_x = float(c.get("keeper_spawn_x", self.KEEPER_SPAWN_X_DEFAULT))
        self.keeper_spawn_y = float(c.get("keeper_spawn_y", self.KEEPER_SPAWN_Y_DEFAULT))
        self.keeper_goal_clearance = float(c.get("keeper_goal_clearance", self.KEEPER_GOAL_CLEARANCE_DEFAULT))

        self.ball_radius = float(c.get("ball_radius", self.BALL_RADIUS_DEFAULT))
        # Nominal speed with per-episode ±20% sampling (override via ball_speed_scale_*).
        # Legacy absolute ball_speed_min/max still accepted if provided.
        self.ball_speed_default = float(c.get("ball_speed", self.BALL_SPEED_DEFAULT))
        self.ball_speed_scale_min = float(
            c.get("ball_speed_scale_min", self.BALL_SPEED_SCALE_MIN_DEFAULT)
        )
        self.ball_speed_scale_max = float(
            c.get("ball_speed_scale_max", self.BALL_SPEED_SCALE_MAX_DEFAULT)
        )
        if "ball_speed_min" in c or "ball_speed_max" in c:
            lo = float(c.get(
                "ball_speed_min",
                self.ball_speed_default * self.ball_speed_scale_min,
            ))
            hi = float(c.get(
                "ball_speed_max",
                self.ball_speed_default * self.ball_speed_scale_max,
            ))
            self.ball_speed_min = float(min(lo, hi))
            self.ball_speed_max = float(max(lo, hi))
        else:
            self.ball_speed_min = float(
                self.ball_speed_default * min(self.ball_speed_scale_min, self.ball_speed_scale_max)
            )
            self.ball_speed_max = float(
                self.ball_speed_default * max(self.ball_speed_scale_min, self.ball_speed_scale_max)
            )
        self.ball_start_x_abs = float(c.get("ball_start_x", self.BALL_START_X_DEFAULT))
        self.ball_start_y_jitter = float(c.get("ball_start_y_jitter", c.get("ball_start_x_jitter", self.BALL_START_Y_JITTER_DEFAULT)))
        self.ball_goal_end_x_offset = float(c.get("ball_goal_end_x_offset", c.get("ball_goal_end_y_offset", self.BALL_GOAL_END_X_OFFSET_DEFAULT)))
        self.ball_target_y_margin = float(c.get("ball_target_y_margin", c.get("ball_target_x_margin", self.BALL_TARGET_Y_MARGIN_DEFAULT)))
        self.ball_angle_deg_min = float(c.get("ball_angle_deg_min", self.BALL_ANGLE_DEG_MIN_DEFAULT))
        self.ball_angle_deg_max = float(c.get("ball_angle_deg_max", self.BALL_ANGLE_DEG_MAX_DEFAULT))
        self.ball_settle_steps = int(c.get("ball_settle_steps", self.BALL_SETTLE_STEPS_DEFAULT))

        # Option 1 — field players (also accepts legacy ``option: 1`` / --option 1).
        self.players_enabled = self._parse_players_enabled(c)
        self.players_max = max(1, int(c.get("players_max", self.PLAYERS_MAX_DEFAULT)))
        self.player_model = str(c.get("player_model", self.PLAYER_MODEL_DEFAULT))
        # Width matches the goalkeeper square (2 * ball_radius); half-z from mesh height.
        self.player_half_xy = float(c.get("player_half_xy", self.ball_radius))
        self.player_half_z = float(c.get("player_half_z", self.PLAYER_HALF_Z_DEFAULT))
        self.player_corridor_margin = float(
            c.get("player_corridor_margin", self.PLAYER_CORRIDOR_MARGIN_DEFAULT)
        )
        self.player_y_spread = float(c.get("player_y_spread", self.PLAYER_Y_SPREAD_DEFAULT))
        self.player_separation = float(c.get("player_separation", self.PLAYER_SEPARATION_DEFAULT))
        self.player_max_goal_dist_mult = float(
            c.get("player_max_goal_dist_mult", self.PLAYER_MAX_GOAL_DIST_MULT_DEFAULT)
        )
        self.player_goal_face_bias = float(
            c.get("player_goal_face_bias", self.PLAYER_GOAL_FACE_BIAS_DEFAULT)
        )
        self._players = []
        self._bounce_player_idx = -1
        self.ball_bounce_pose = None
        self._bounce_end_y = None

        # Option 2 — field cover (also accepts legacy ``option: 2`` / --option 2).
        self.cover_enabled = self._parse_cover_enabled(c)
        self.cover_entry_gap = float(c.get("cover_entry_gap", self.COVER_ENTRY_GAP_DEFAULT))
        self.cover_exit_gap_frac = float(
            c.get("cover_exit_gap_frac", self.COVER_EXIT_GAP_FRAC_DEFAULT)
        )
        self.cover_half_y = float(c.get("cover_half_y", self.COVER_HALF_Y_DEFAULT))
        self.cover_clearance_z = float(
            c.get("cover_clearance_z", self.COVER_CLEARANCE_Z_DEFAULT)
        )
        self.cover_wall_t = float(c.get("cover_wall_t", self.COVER_WALL_T_DEFAULT))
        self.cover_roof_t = float(c.get("cover_roof_t", self.COVER_ROOF_T_DEFAULT))
        self.cover_len_min = float(c.get("cover_len_min", self.COVER_LEN_MIN_DEFAULT))
        self.cover_player_clearance = float(
            c.get("cover_player_clearance", self.COVER_PLAYER_CLEARANCE_DEFAULT)
        )
        self._cover_parts = []
        self.cover_x_min = None
        self.cover_x_max = None
        self.cover_len = 0.0

        # Soccer-style visual net behind the goal mouth.
        self.net_enabled = self._as_bool(c.get("net_enabled", None), self.NET_ENABLED_DEFAULT)
        self.net_depth = float(c.get("net_depth", self.NET_DEPTH_DEFAULT))
        self.net_cell = float(c.get("net_cell", self.NET_CELL_DEFAULT))
        self.net_strand_t = float(c.get("net_strand_t", self.NET_STRAND_T_DEFAULT))
        self._net_parts = []

        # Mirror flips the whole field across the table midline (goal/ball/players/cover).
        # mirrored → travel_dir −1 → goal on −x → left gripper; else +x → right gripper.
        self.mirrored = self._parse_mirrored(c)
        self.travel_dir = -1.0 if self.mirrored else 1.0
        self.goal_x = float(self.travel_dir * abs(self.goal_x_abs))
        self.red_line_x = float(self.goal_x - self.travel_dir * abs(self.red_line_goal_offset))
        self.ball_speed = float(np.random.uniform(self.ball_speed_min, self.ball_speed_max))
        self.keeper_half_x = self.ball_radius
        self.keeper_half_y = self.ball_radius
        self.keeper_half_z = self.ball_radius
        green_half_x = 0.5 * self.green_area_x_len
        green_half_y = self.goal_half_w + 0.5 * self.green_area_y_extra
        self.green_area_center_x = float(self.goal_x - self.travel_dir * green_half_x)
        self.green_area_center_y = float(self.goal_center_y)
        self.green_area_x_min = float(min(self.goal_x, self.goal_x - self.travel_dir * self.green_area_x_len))
        self.green_area_x_max = float(max(self.goal_x, self.goal_x - self.travel_dir * self.green_area_x_len))
        self.green_area_y_min = float(self.goal_center_y - green_half_y)
        self.green_area_y_max = float(self.goal_center_y + green_half_y)

        goal_end_x = float(self.goal_x + self.travel_dir * abs(self.ball_goal_end_x_offset))
        dt = float(self.scene.get_timestep())
        start_x = float(-self.travel_dir * abs(self.ball_start_x_abs))
        start_y = float(np.random.uniform(
            self.goal_center_y - self.ball_start_y_jitter,
            self.goal_center_y + self.ball_start_y_jitter,
        ))

        # Spawn Opt-1 players before choosing the post-bounce goal aim.
        self._spawn_field_players(start_x, start_y, goal_end_x)
        # Option 2 cover over the mid field (after start/red-line are known).
        self._spawn_field_cover(start_x)

        if self.ball_bounce_pose is not None and self._bounce_end_y is not None:
            end_y = float(self._bounce_end_y)
        else:
            for _ in range(64):
                launch_angle_deg = float(np.random.uniform(self.ball_angle_deg_min, self.ball_angle_deg_max))
                angle_rad = np.deg2rad(launch_angle_deg)
                end_y = float(start_y + np.tan(angle_rad) * (goal_end_x - start_x))
                if abs(end_y - self.goal_center_y) <= (self.goal_half_w - self.ball_target_y_margin):
                    break
            else:
                end_y = float(np.clip(
                    end_y,
                    self.goal_center_y - self.goal_half_w + self.ball_target_y_margin,
                    self.goal_center_y + self.goal_half_w - self.ball_target_y_margin,
                ))

        self.launch_angle_deg = float(np.degrees(np.arctan2(end_y - start_y, goal_end_x - start_x)))
        self.ball_start_pose = np.array(
            [start_x, start_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        self.ball_target_pose = np.array(
            [goal_end_x, end_y, self.table_top_z + self.ball_radius],
            dtype=np.float64,
        )
        if self.ball_bounce_pose is not None:
            self._set_ball_waypoints(
                [self.ball_start_pose, self.ball_bounce_pose, self.ball_target_pose]
            )
            ball_vec = self.ball_target_pose - self.ball_bounce_pose
        else:
            self._set_ball_waypoints([self.ball_start_pose, self.ball_target_pose])
            ball_vec = self.ball_target_pose - self.ball_start_pose
        ball_dist = float(self._ball_path_len)
        self.ball_dir = ball_vec / max(float(np.linalg.norm(ball_vec)), 1e-8)
        self.ball_total_steps = max(1, int(np.ceil(ball_dist / max(self.ball_speed * dt, 1e-8))))

        keeper_x_min = float(self.green_area_x_min + self.keeper_half_x)
        keeper_x_max = float(self.green_area_x_max - self.keeper_half_x)
        self.goal_intersection_y = self._ball_path_y_at_x(self.goal_x)
        preferred_keeper_x = float(self.goal_x - self.travel_dir * (self.keeper_half_x + 0.002))
        self.keeper_x = float(np.clip(preferred_keeper_x, keeper_x_min, keeper_x_max))
        keeper_y = float(self._ball_path_y_at_x(self.keeper_x))
        keeper_y = float(np.clip(
            keeper_y,
            self.green_area_y_min + self.keeper_half_y,
            self.green_area_y_max - self.keeper_half_y,
        ))
        self.goalkeeper_target_pose = sapien.Pose(
            [self.keeper_x, keeper_y, self.table_top_z + self.keeper_half_z],
            [1, 0, 0, 0],
        )

        goal_color = (0.92, 0.92, 0.94)
        post_half = [self.goal_post_t * 0.5, self.goal_post_t * 0.5, self.goal_post_h * 0.5]
        self.goal_left_post = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y - self.goal_half_w, self.table_top_z + self.goal_post_h * 0.5], [1, 0, 0, 0]),
            half_size=post_half,
            color=goal_color,
            is_static=True,
            name="goal_left_post",
        )
        self.goal_right_post = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y + self.goal_half_w, self.table_top_z + self.goal_post_h * 0.5], [1, 0, 0, 0]),
            half_size=post_half,
            color=goal_color,
            is_static=True,
            name="goal_right_post",
        )
        self.goal_bar = create_box(
            self,
            pose=sapien.Pose([self.goal_x, self.goal_center_y, self.table_top_z + self.goal_post_h - self.goal_bar_t * 0.5], [1, 0, 0, 0]),
            half_size=[self.goal_post_t * 0.5, self.goal_half_w + self.goal_post_t, self.goal_bar_t * 0.5],
            color=goal_color,
            is_static=True,
            name="goal_bar",
        )
        self._spawn_goal_net()
        create_visual_box(
            self,
            pose=sapien.Pose([self.red_line_x, self.goal_center_y, self.table_top_z + 0.001], [1, 0, 0, 0]),
            half_size=[0.002, self.goal_half_w + 0.10, 0.001],
            color=(0.95, 0.12, 0.12),
            name="red_line",
        )
        create_visual_box(
            self,
            pose=sapien.Pose([self.green_area_center_x, self.green_area_center_y, self.table_top_z + 0.001], [1, 0, 0, 0]),
            half_size=[green_half_x, green_half_y, 0.001],
            color=(0.18, 0.72, 0.25),
            name="goal_green_area",
        )

        keeper_x0 = float(self.goal_x)
        goal_lower_y = float(self.goal_center_y - self.goal_half_w)
        keeper_y0 = float(goal_lower_y - self.keeper_goal_clearance - self.keeper_half_y)
        self.goalkeeper = create_box(
            self,
            pose=sapien.Pose([keeper_x0, keeper_y0, self.table_top_z + self.keeper_half_z], [1, 0, 0, 0]),
            half_size=[self.keeper_half_x, self.keeper_half_y, self.keeper_half_z],
            color=(0.90, 0.70, 0.12),
            name="goalkeeper_square",
            is_static=False,
        )
        self.goalkeeper.set_mass(0.08)
        for comp in self.goalkeeper.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    comp.set_linear_damping(10.0)
                    comp.set_angular_damping(10.0)
                except Exception:
                    pass

        self.ball = create_sphere(
            self.scene,
            pose=sapien.Pose(self.ball_start_pose.tolist(), [1, 0, 0, 0]),
            radius=self.ball_radius,
            color=(0.18, 0.56, 0.90),
            is_static=False,
            name="goal_ball",
        )
        self._ball_rigid = self._get_rigid(self.ball)
        if self._ball_rigid is not None:
            try:
                self._ball_rigid.set_disable_gravity(True)
                self._ball_rigid.set_kinematic(True)
                self._ball_rigid.set_linear_velocity(np.zeros(3))
                self._ball_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

        self.add_prohibit_area(self.goalkeeper, padding=0.02)
        self._loaded = True

    # ------------------------------------------------------------- motion / checks
    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if not getattr(self, "_ball_motion_active", False):
            return
        if self._ball_blocked:
            return
        if self._ball_rigid is None or getattr(self, "ball", None) is None:
            return

        self._ball_step += 1
        progress = min(1.0, self._ball_step / float(self.ball_total_steps))
        prev_progress = min(1.0, (self._ball_step - 1) / float(self.ball_total_steps))
        prev_p = self._ball_pos_at_progress(prev_progress)
        next_p = self._ball_pos_at_progress(progress)

        keeper_ok = bool(self._keeper_deployed and self._keeper_in_zone())
        if (not self._late_failure) and (self.travel_dir * next_p[0] >= self.travel_dir * self.red_line_x) and (not keeper_ok):
            self._late_failure = True

        if keeper_ok:
            x_min, x_max, y_min, y_max = self._keeper_xy_bounds()
            # Front face = ball-facing side of the square. Only a front-face hit
            # stops the ball; a side graze (ball center outside [y_min, y_max])
            # lets the ball keep moving through toward the goal.
            face_x = float(
                (x_min - self.ball_radius - 0.001)
                if self.travel_dir > 0
                else (x_max + self.ball_radius + 0.001)
            )
            # Segment test against the face plane so large kinematic steps cannot
            # tunnel through the keeper in a single update.
            crossed_face = (
                self.travel_dir * prev_p[0] < self.travel_dir * face_x
                and self.travel_dir * next_p[0] >= self.travel_dir * face_x
            ) or (
                abs(float(next_p[0] - face_x)) <= 1e-4
                and self.travel_dir * next_p[0] >= self.travel_dir * face_x
            )
            if crossed_face:
                dx = float(next_p[0] - prev_p[0])
                if abs(dx) < 1e-9:
                    hit_y = float(next_p[1])
                else:
                    t = float((face_x - prev_p[0]) / dx)
                    hit_y = float(prev_p[1] + t * (next_p[1] - prev_p[1]))
                # Small tolerance for placement / floating error on the narrow square.
                y_tol = 0.002
                hits_front = (y_min - y_tol) <= hit_y <= (y_max + y_tol)
                if hits_front:
                    next_p = next_p.copy()
                    next_p[0] = face_x
                    next_p[1] = hit_y
                    self._ball_blocked = True
                    self._ball_motion_active = False
                    pose = sapien.Pose(next_p.tolist(), [1, 0, 0, 0])
                    self.ball.set_pose(pose)
                    try:
                        self._ball_rigid.set_kinematic_target(pose)
                    except Exception:
                        pass
                    return

        if self.travel_dir * next_p[0] >= self.travel_dir * self.goal_x:
            self._ball_crossed_goal = True
        if (
            (not self._goal_conceded)
            and self.travel_dir * next_p[0] >= self.travel_dir * self.goal_x
            and abs(float(next_p[1] - self.goal_center_y)) <= self.goal_half_w
        ):
            self._goal_conceded = True

        pose = sapien.Pose(next_p.tolist(), [1, 0, 0, 0])
        self.ball.set_pose(pose)
        try:
            self._ball_rigid.set_kinematic_target(pose)
        except Exception:
            pass
        if progress >= 1.0:
            self._ball_motion_active = False

    # ----------------------------------------------------------------- policy
    def play_once(self):
        # Start the shot when the expert begins acting (deterministic across collector passes).
        self._ball_step = 0
        self._ball_blocked = False
        self._goal_conceded = False
        self._late_failure = False
        self._ball_crossed_goal = False
        self._keeper_deployed = False
        self._keeper_drop_pose = None
        self._ball_motion_active = True

        arm_tag = ArmTag("left" if self.mirrored else "right")
        grasp_contact_id = [0, 1, 2, 3]

        self.move(self.close_gripper(arm_tag, pos=0.6))
        self.move(
            self.grasp_actor(
                self.goalkeeper,
                arm_tag=arm_tag,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
                contact_point_id=grasp_contact_id,
            )
        )
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self._place_keeper_from_top(arm_tag)
        # After the yellow cube is dropped, always return the arm to its origin pose.
        self._retreat_arm_home(arm_tag, lift_z=0.08)
        # Do not re-set the cube pose after retreat (that caused a visible teleport).
        # Only re-assert kinematic so contacts cannot drag it.
        self._hold_keeper_kinematic()

        self._wait_for_outcome()

        self.info["info"] = {
            "{A}": "goalkeeper_square",
            "{B}": "goal_frame",
            "{a}": str(arm_tag),
        }
        # Option captions use {o}; attach when Option 1 and/or Option 2 is on.
        if self.players_enabled or self.cover_enabled:
            self.info["info"]["{o}"] = self._option_label()
        return self.info

    # ---------------------------------------------------------------- success / obs
    def check_success(self):
        keeper_ok = self._keeper_in_zone()
        success = bool(
            keeper_ok
            and self._ball_blocked
            and (not self._late_failure)
            and (not self._goal_conceded)
            and self.is_left_gripper_open()
            and self.is_right_gripper_open()
        )
        self.info["save_goal"] = {
            "keeper_in_zone": bool(keeper_ok),
            "ball_blocked": bool(self._ball_blocked),
            "ball_crossed_goal": bool(self._ball_crossed_goal),
            "late_failure": bool(self._late_failure),
            "goal_conceded": bool(self._goal_conceded),
            "players_enabled": bool(getattr(self, "players_enabled", False)),
            "n_players": int(len(getattr(self, "_players", []) or [])),
            "bounce_player_idx": int(getattr(self, "_bounce_player_idx", -1)),
            "cover_enabled": bool(getattr(self, "cover_enabled", False)),
            "mirrored": bool(getattr(self, "mirrored", False)),
            "net_enabled": bool(getattr(self, "net_enabled", False)),
            "n_net_parts": int(len(getattr(self, "_net_parts", []) or [])),
            "cover_x_min": (
                float(self.cover_x_min) if getattr(self, "cover_x_min", None) is not None else None
            ),
            "cover_x_max": (
                float(self.cover_x_max) if getattr(self, "cover_x_max", None) is not None else None
            ),
        }
        return success

    def get_obs(self):
        obs = super().get_obs()
        player_positions = []
        for p in getattr(self, "_players", []) or []:
            try:
                player_positions.append(p.get_pose().p.tolist())
            except Exception:
                player_positions.append([0.0, 0.0, 0.0])
        obs["save_goal"] = {
            "ball_pos": self.ball.get_pose().p.tolist() if getattr(self, "ball", None) is not None else [0.0, 0.0, 0.0],
            "keeper_pos": self.goalkeeper.get_pose().p.tolist() if getattr(self, "goalkeeper", None) is not None else [0.0, 0.0, 0.0],
            "keeper_target": self.goalkeeper_target_pose.p.tolist() if self.goalkeeper_target_pose is not None else [0.0, 0.0, 0.0],
            "keeper_in_zone": bool(self._keeper_in_zone()),
            "ball_blocked": bool(self._ball_blocked),
            "ball_crossed_goal": bool(self._ball_crossed_goal),
            "late_failure": bool(self._late_failure),
            "goal_conceded": bool(self._goal_conceded),
            "ball_speed": float(getattr(self, "ball_speed", 0.0)),
            "ball_speed_default": float(getattr(self, "ball_speed_default", 0.0)),
            "launch_angle_deg": float(getattr(self, "launch_angle_deg", 0.0)),
            "red_line_x": float(getattr(self, "red_line_x", 0.0)),
            "goal_x": float(getattr(self, "goal_x", 0.0)),
            "goal_center_y": float(getattr(self, "goal_center_y", 0.0)),
            "goal_intersection_y": float(getattr(self, "goal_intersection_y", 0.0)),
            "green_area_center": [float(getattr(self, "green_area_center_x", 0.0)), float(getattr(self, "green_area_center_y", 0.0))],
            "green_area_bounds": [
                float(getattr(self, "green_area_x_min", 0.0)),
                float(getattr(self, "green_area_x_max", 0.0)),
                float(getattr(self, "green_area_y_min", 0.0)),
                float(getattr(self, "green_area_y_max", 0.0)),
            ],
            "travel_dir": float(getattr(self, "travel_dir", 0.0)),
            "mirrored": bool(getattr(self, "mirrored", False)),
            "net_enabled": bool(getattr(self, "net_enabled", False)),
            "n_net_parts": int(len(getattr(self, "_net_parts", []) or [])),
            "players_enabled": bool(getattr(self, "players_enabled", False)),
            "n_players": int(len(getattr(self, "_players", []) or [])),
            "bounce_player_idx": int(getattr(self, "_bounce_player_idx", -1)),
            "player_positions": player_positions,
            "ball_bounce_pos": (
                self.ball_bounce_pose.tolist()
                if getattr(self, "ball_bounce_pose", None) is not None
                else [0.0, 0.0, 0.0]
            ),
            "cover_enabled": bool(getattr(self, "cover_enabled", False)),
            "cover_x_range": [
                float(self.cover_x_min) if getattr(self, "cover_x_min", None) is not None else 0.0,
                float(self.cover_x_max) if getattr(self, "cover_x_max", None) is not None else 0.0,
            ],
        }
        return obs
