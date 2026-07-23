"""Cook-meat task with contact-gated doneness and collision-aware placement.

Author: Rui Heng Yang
"""

from __future__ import annotations

import json
import os
from typing import Any, ClassVar, Sequence, TypeAlias

import numpy as np
import sapien
import sapien.render
import trimesh

from ._base_task import Base_Task
from ._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
from .utils.action import ArmTag
from .utils.create_actor import UnStableError, create_actor, create_box
from .utils.rand_create_actor import rand_pose

import transforms3d as t3d


AABB: TypeAlias = tuple[float, float, float, float]


class cook_meat(Base_Task):
    """Cook a steak to a sampled doneness and return it to the cutting board.

    Default: place steak on the pan; it cooks on contact while waiting, then
    return it to the cutting board. Cook speed samples around nominal
    ``cook_steps`` ± jitter.

    Options (``task_args.cook_meat``; independent toggles):
      Opt 1 — cook button  →  ``cook_button_enabled`` (**default: false**)
          Red keycap on a thin black base; hold to cook while the steak is on
          the pan. Key sits on the same lateral side as the station (left of
          pan when the station is left; right of pan when right).
          CLI: ``--task-arg cook_button_enabled=true`` / ``--option 1``.
      Opt 2 — dual setup  →  ``dual_setup_enabled`` (**default: false**)
          Mirror a second station (pan, board, steak) with ≥10 cm clearance
          between setups; both arms place and pick up meats together.
          Success requires **both** steaks cooked to target doneness.
          CLI: ``--task-arg dual_setup_enabled=true`` or ``--option 2``.
      Opt 1+2 — dual stations each with their own cook key; color advances
          only while that station's key is pressed and its steak is on the pan.
          Success still requires both steaks cooked properly.
    """

    COOK_STEPS_DEFAULT: ClassVar[int] = 1000
    COOK_SPEED_JITTER_DEFAULT: ClassVar[float] = 0.20  # per-ep cook_steps ~ U(nom×(1±j))
    TARGET_DONENESS_DEFAULT: ClassVar[float] = 0.5
    COOK_BUTTON_ENABLED_DEFAULT: ClassVar[bool] = False  # default = contact cook
    DUAL_SETUP_ENABLED_DEFAULT: ClassVar[bool] = False  # Opt 2
    # Allowed |grasp_doneness − target| for success (not under- or over-cooked).
    COOK_DONENESS_TOL_DEFAULT: ClassVar[float] = 0.08
    # Colored keycap + thin black base (marble / dual_hole_punch styling).
    KEY_HALF: ClassVar[tuple[float, float, float]] = (0.020, 0.020, 0.014)
    KEY_BASE_HALF: ClassVar[tuple[float, float, float]] = (0.032, 0.032, 0.005)
    KEY_BASE_COLOR: ClassVar[tuple[float, float, float]] = (0.08, 0.08, 0.08)
    KEY_COLOR: ClassVar[tuple[float, float, float]] = (0.85, 0.10, 0.10)  # red
    KEY_OFFSET_X_DEFAULT: ClassVar[float] = 0.14   # m; |lateral| offset from pan center
    KEY_Y_BIAS_DEFAULT: ClassVar[float] = -0.04    # m; toward robot (−y) from pan center
    KEY_PRESS_DEPTH_DEFAULT: ClassVar[float] = 0.055
    KEY_PRESS_XY_DEFAULT: ClassVar[float] = 0.06
    # EE frame sits ~EE_TO_TCP above the TCP; a true key contact leaves ee_z ≈
    # key_top_z + EE_TO_TCP. CuRobo often stops a few cm short of the mesh, so
    # allow ~key_top_z + 0.20 (matches dispense_gummy 0.17 with margin).
    KEY_PRESS_DZ_DEFAULT: ClassVar[float] = 0.20
    KEY_HOVER_DIS_DEFAULT: ClassVar[float] = 0.06  # m above key top before press
    EE_TO_TCP: ClassVar[float] = 0.12  # EE frame → TCP offset (dispense_gummy convention)
    # Skillet mesh AABB includes a long handle; board only needs the bowl clear.
    PAN_BOWL_HALF: ClassVar[float] = 0.095
    # Key must clear this radius from pan center (bowl + margin so it never sits on the rim).
    KEY_PAN_CLEAR_HALF: ClassVar[float] = 0.12
    # XY clearance between pan / board / key footprints within one station.
    PROP_CLEARANCE: ClassVar[float] = 0.03
    # Minimum gap between left and right pan bowls (Opt 2 / Opt 1+2).
    DUAL_STATION_CLEARANCE: ClassVar[float] = 0.10
    # Dual pan |x|; bowls need ≥ DUAL_STATION_CLEARANCE, so |x| ≥ bowl_half + 0.05.
    DUAL_PAN_NOM_X: ClassVar[float] = 0.15
    # Default (right-side) stable centers; Opt 2 / left mirror by negating X.
    PAN_NOM_X: ClassVar[float] = 0.10
    PAN_NOM_Y: ClassVar[float] = -0.10
    # Board sits mostly +y of the pan (clear of bowl, in head view).
    BOARD_OFF_X: ClassVar[float] = 0.02
    BOARD_OFF_Y: ClassVar[float] = 0.22
    # Small diversity only — keep the layout near the stable nominals.
    PLACE_JITTER_XY_DEFAULT: ClassVar[float] = 0.01   # ±1 cm
    PLACE_YAW_LIM_DEFAULT: ClassVar[float] = float(np.pi / 18)  # ±10° about +Z
    KEY_MIN_GAP: ClassVar[float] = 0.04  # min XY gap key ↔ pan/board
    SKILLET_BASE_QPOS: ClassVar[list[float]] = [0.0, 0.0, 0.707, 0.707]
    BOARD_BASE_QPOS: ClassVar[list[float]] = [0.707, 0.707, 0.0, 0.0]
    # cooking gradient: raw red -> MEDIUM warm red-brown (at the 0.5 target) -> well-done dark brown.
    # 0.5 is deliberately a clear MEDIUM (still reddish), not full brown, so picking at the 0.5
    # target reads visually as "medium", not "fully cooked".
    COLOR_STOPS: ClassVar[list[tuple[float, list[float]]]] = [
        (0.0, [1.00, 0.12, 0.09]),    # raw: vivid saturated red
        (0.5, [0.66, 0.30, 0.14]),    # medium: warm red-brown (clearly transitional)
        (1.0, [0.16, 0.08, 0.04]),    # well done: dark brown
    ]

    def setup_demo(self, **kwargs: Any) -> None:
        """Initialize one episode from the shared task configuration."""
        self._cook_cfg = dict(kwargs.get("task_args", {}).get("cook_meat", {}))
        self._apply_legacy_option()
        self._ep_seed = int(kwargs.get("seed", 0))
        super()._init_task_env_(**kwargs)

    def _apply_legacy_option(self) -> None:
        """Map record_demo ``--option`` / config ``option`` onto named toggles.

        1 / cook_button / button → Opt 1 cook_button_enabled=true
        2 / dual / dual_setup → Opt 2 dual_setup_enabled=true
        """
        legacy = self._cook_cfg.get("option", None)
        if legacy is None:
            return
        key = {
            1: "cook_button_enabled",
            "1": "cook_button_enabled",
            "cook_button": "cook_button_enabled",
            "cook_button_enabled": "cook_button_enabled",
            "button": "cook_button_enabled",
            2: "dual_setup_enabled",
            "2": "dual_setup_enabled",
            "dual": "dual_setup_enabled",
            "dual_setup": "dual_setup_enabled",
            "dual_setup_enabled": "dual_setup_enabled",
        }.get(legacy if not isinstance(legacy, str) else legacy.strip().lower())
        if key == "cook_button_enabled":
            self._cook_cfg["cook_button_enabled"] = True
        elif key == "dual_setup_enabled":
            self._cook_cfg["dual_setup_enabled"] = True
        else:
            raise ValueError(
                "cook_meat option must be 1/cook_button_enabled or "
                "2/dual_setup_enabled (or set those keys directly)"
            )

    def _option_label(self) -> str:
        parts: list[str] = []
        if getattr(self, "cook_button_enabled", False):
            parts.append("option 1")
        if getattr(self, "dual_setup_enabled", False):
            parts.append("option 2")
        return ", ".join(parts) if parts else "default"

    @property
    def use_cook_button(self) -> bool:
        """Whether cook keys are active (Opt 1 and Opt 1+2)."""
        return bool(getattr(self, "cook_button_enabled", False))

    @staticmethod
    def _union_aabb(aabbs: Sequence[AABB]) -> AABB:
        """Axis-aligned union of one or more XY AABBs."""
        return (
            min(a[0] for a in aabbs),
            min(a[1] for a in aabbs),
            max(a[2] for a in aabbs),
            max(a[3] for a in aabbs),
        )

    @staticmethod
    def _expand_aabb(aabb: AABB, margin: float) -> AABB:
        """Expand an XY AABB by ``margin`` on every side."""
        m = float(margin)
        return (aabb[0] - m, aabb[1] - m, aabb[2] + m, aabb[3] + m)

    # ------------------------------------------------------- footprint checks
    # rand_pose()/create_actor() do NOT check for overlap between objects, and add_prohibit_area()
    # only feeds the eval-time trajectory-extension check (_base_task.py) -- there is no built-in
    # guard against two hand-placed static props (e.g. the pan and the board) landing on top of
    # each other. Rather than approximate each object's footprint from model_data's "extents"
    # field, use the REAL collision mesh: script/create_object_data.py:275 computes "extents" via
    # trimesh's `bounding_box_oriented` -- a MINIMUM-VOLUME oriented box that can be tilted relative
    # to the mesh's own local axes (this is pronounced for the skillet, an irregular shape with a
    # handle: extents-based math overestimates its world footprint by ~35% in x, ~10% in y versus
    # the actual mesh, measured directly). Loading the real mesh and rotating/scaling its actual
    # vertices gives the true footprint regardless of how any given asset's OMBB happens to be
    # oriented, and needs no per-asset special-casing.
    _MESH_CACHE: ClassVar[dict[str, np.ndarray]] = {}

    @staticmethod
    def _applied_scale(modelname: str, model_id: int, scale_mult: float) -> float:
        """Return the scale applied to raw collision-mesh vertices."""
        # The scale create_actor() actually bakes into the mesh is model_data["scale"] * scale_mult
        # (create_actor.py:542,552,564). model_data's raw collision-glb vertices are in UN-scaled
        # mesh units, so the footprint helpers below MUST be fed this same product -- feeding only
        # scale_mult inflates the footprint by 1/authored_scale (e.g. for small authored scales).
        # Assets with no "scale" key (e.g. 104_board) get authored scale 1.0, matching create_actor's
        # (1,1,1) default for that case.
        try:
            model_data_path = f"assets/objects/{modelname}/model_data{model_id}.json"
            with open(model_data_path, encoding="utf-8") as model_data_file:
                model_data = json.load(model_data_file)
            authored = float(model_data["scale"][0])
        except (KeyError, TypeError, FileNotFoundError, ValueError):
            authored = 1.0
        return authored * float(scale_mult)

    @classmethod
    def _mesh_vertices(cls, collision_path: str) -> np.ndarray:
        """Load and cache vertices from a collision mesh."""
        if collision_path not in cls._MESH_CACHE:
            mesh = trimesh.load(collision_path, force="mesh")
            cls._MESH_CACHE[collision_path] = np.asarray(mesh.vertices, dtype=float)
        return cls._MESH_CACHE[collision_path]

    @classmethod
    def _mesh_aabb_xy(
        cls,
        collision_path: str,
        pose: sapien.Pose,
        scale: float,
        padding: float = 0.0,
    ) -> AABB:
        """Return the padded world-space XY bounds of a collision mesh."""
        vertices = cls._mesh_vertices(collision_path) * scale
        transform = pose.to_transformation_matrix()
        world = (transform[:3, :3] @ vertices.T).T + np.asarray(pose.p)
        return (
            float(world[:, 0].min()) - padding, float(world[:, 1].min()) - padding,
            float(world[:, 0].max()) + padding, float(world[:, 1].max()) + padding,
        )

    @classmethod
    def _mesh_world_z_extent(
        cls,
        collision_path: str,
        pose: sapien.Pose,
        scale: float,
    ) -> tuple[float, float]:
        """Return the transformed mesh's minimum and maximum world Z."""
        vertices = cls._mesh_vertices(collision_path) * scale
        transform = pose.to_transformation_matrix()
        world = (transform[:3, :3] @ vertices.T).T + np.asarray(pose.p)
        return float(world[:, 2].min()), float(world[:, 2].max())

    @classmethod
    def _footprint_offsets(
        cls,
        collision_path: str,
        qpos: Sequence[float],
        scale: float,
    ) -> AABB:
        """Return rotation-and-scale-adjusted XY bounds before translation."""
        # Fixed world-XY footprint box (vmin_x, vmin_y, vmax_x, vmax_y) of the mesh at this
        # rotation+scale BEFORE translation. Translating a rigid footprint merely shifts its AABB, so
        # AABB(x, y) = (vmin_x + x, vmin_y + y, vmax_x + x, vmax_y + y). Computing this once lets the
        # grid fallback below offset it per cell instead of re-transforming the mesh every candidate.
        vertices = cls._mesh_vertices(collision_path) * scale
        rotation = sapien.Pose([0.0, 0.0, 0.0], qpos).to_transformation_matrix()[:3, :3]
        rotated = (rotation @ vertices.T).T
        return (
            float(rotated[:, 0].min()),
            float(rotated[:, 1].min()),
            float(rotated[:, 0].max()),
            float(rotated[:, 1].max()),
        )

    @staticmethod
    def _aabb_gap(a: AABB, b: AABB) -> float:
        """Return the maximum signed gap between two XY AABBs.

        A positive result means that at least one axis separates the boxes;
        zero means that their boundaries touch; and a negative result means
        that they overlap on both axes.
        """
        gx = max(b[0] - a[2], a[0] - b[2])
        gy = max(b[1] - a[3], a[1] - b[3])
        return max(gx, gy)

    # ------------------------------------------------------ head-camera view
    # The front video the episodes are rendered from is "head_camera" (ur5-wsg static_camera_list,
    # assets/embodiments/ur5-wsg/config.yml:29-40): a D435 (task_config/_camera_config.yml -> fovy 37
    # deg, 320x240) mounted at position [-0.032,-0.45,1.35] looking forward [0,0.6,-0.8] (pitched
    # down ~53 deg) with left [-1,0,0]. So the visible patch of the table is a trapezoid; an object
    # spawned too far to the side / too near the front edge would render partly out of frame. These
    # constants + the projection below let us REJECT any spawn whose footprint leaves the image.
    # (random_head_camera_dis is 0 in both demo_dynamic and debug_dynamic, so the pose is exact.)
    _CAM_POS: ClassVar[np.ndarray] = np.array([-0.032, -0.45, 1.35])
    _CAM_FWD: ClassVar[np.ndarray] = np.array([0.0, 0.6, -0.8])
    _CAM_LEFT: ClassVar[np.ndarray] = np.array([-1.0, 0.0, 0.0])
    _CAM_FOVY_DEG: ClassVar[float] = 37.0
    _CAM_W: ClassVar[int] = 320
    _CAM_H: ClassVar[int] = 240

    @classmethod
    def _project_to_head_cam(
        cls,
        points_world: Sequence[Sequence[float]] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world-space points into the fixed head camera."""
        # World (N,3) -> (u, v, depth). Mirrors camera.py:101-120: camera-local axes are the columns
        # [forward, left, up] with up = forward x left, so p_cam = R^T (p - cam_pos) gives
        # x=depth-along-view, y=left, z=up; pinhole with square pixels from fovy. Validated to <1px
        # against the known visible band (near edge y~-0.247 -> v~H, far edge y~0.433 -> v~0).
        f = cls._CAM_FWD / np.linalg.norm(cls._CAM_FWD)
        l = cls._CAM_LEFT / np.linalg.norm(cls._CAM_LEFT)
        up = np.cross(f, l)
        displacement = np.asarray(points_world, dtype=float) - cls._CAM_POS
        depth = displacement @ f
        y_cam = displacement @ l
        z_cam = displacement @ up
        focal_length = (cls._CAM_H / 2.0) / np.tan(
            np.deg2rad(cls._CAM_FOVY_DEG) / 2.0
        )
        center_x, center_y = cls._CAM_W / 2.0, cls._CAM_H / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            u = center_x - focal_length * (y_cam / depth)
            v = center_y - focal_length * (z_cam / depth)
        return u, v, depth

    @classmethod
    def _footprint_in_head_view(
        cls,
        aabb_xy: AABB,
        z: float,
        margin_px: int = 4,
    ) -> bool:
        """Return whether all footprint corners lie inside the head-camera view."""
        # True iff all four footprint corners (at table-top height z) project strictly inside the
        # image with a small pixel margin -- i.e. the object renders fully within the front video.
        xmin, ymin, xmax, ymax = aabb_xy
        corners = np.array([[xmin, ymin, z], [xmin, ymax, z], [xmax, ymin, z], [xmax, ymax, z]])
        u, v, depth = cls._project_to_head_cam(corners)
        if np.any(depth <= 0):
            return False
        return bool(
            np.all(u >= margin_px)
            and np.all(u <= cls._CAM_W - margin_px)
            and np.all(v >= margin_px)
            and np.all(v <= cls._CAM_H - margin_px)
        )

    def _sample_clear_pose(
        self,
        xlim: Sequence[float],
        ylim: Sequence[float],
        qpos: Sequence[float],
        collision_path: str,
        scale: float,
        avoid_aabbs: Sequence[AABB],
        padding: float,
        view_z: float,
        tries: int = 400,
        grid_step: float = 0.005,
    ) -> tuple[sapien.Pose | None, AABB | None]:
        """Sample a pose whose XY footprint is clear and inside the camera frame.

        Visibility is approximated by projecting the four AABB corners at
        ``view_z``. A failed search returns ``(None, None)`` so the caller can
        reject the episode seed without expanding the configured spawn band.
        """
        def is_valid(aabb: AABB) -> bool:
            clearance = min(
                (self._aabb_gap(aabb, other) for other in avoid_aabbs),
                default=float("inf"),
            )
            return clearance > 0 and self._footprint_in_head_view(aabb, view_z)

        # Phase 1: random rejection sampling -- gives per-seed spatial diversity when a spot exists.
        for _ in range(tries):
            candidate = rand_pose(xlim=xlim, ylim=ylim, qpos=qpos)
            candidate_aabb = self._mesh_aabb_xy(
                collision_path, candidate, scale, padding=padding
            )
            if is_valid(candidate_aabb):
                return candidate, candidate_aabb

        # Phase 2: scan a 5-mm grid to reduce misses from random rejection sampling. The footprint
        # shape is translation-invariant, so offset one precomputed box per grid cell. Select a
        # valid cell at random to preserve layout diversity.
        offsets = self._footprint_offsets(collision_path, qpos, scale)

        def aabb_at(x: float, y: float) -> AABB:
            return (
                offsets[0] + x - padding,
                offsets[1] + y - padding,
                offsets[2] + x + padding,
                offsets[3] + y + padding,
            )
        xlo, xhi = min(xlim), max(xlim)
        ylo, yhi = min(ylim), max(ylim)
        valid_cells: list[tuple[float, float, AABB]] = []
        nx = max(2, int((xhi - xlo) / grid_step) + 1)
        ny = max(2, int((yhi - ylo) / grid_step) + 1)
        for gx in np.linspace(xlo, xhi, nx):
            for gy in np.linspace(ylo, yhi, ny):
                candidate_aabb = aabb_at(float(gx), float(gy))
                if is_valid(candidate_aabb):
                    valid_cells.append((float(gx), float(gy), candidate_aabb))
        if valid_cells:
            grid_x, grid_y, candidate_aabb = valid_cells[
                np.random.randint(len(valid_cells))
            ]
            return sapien.Pose([grid_x, grid_y, 0.741], qpos), candidate_aabb
        return None, None

    @classmethod
    def _sample_cook_steps(cls, config: dict[str, Any]) -> int:
        """Sample per-episode cook duration around a nominal rate ± jitter.

        Default setup: ``cook_steps`` (nominal steps for doneness 0→1) times
        ``U(1 ± cook_speed_jitter)`` with jitter defaulting to 20%.

        Legacy ``cook_steps_min`` / ``cook_steps_max`` (without ``cook_steps``)
        still sample uniformly in that inclusive range.
        """
        jitter = float(config.get("cook_speed_jitter", cls.COOK_SPEED_JITTER_DEFAULT))
        jitter = float(np.clip(abs(jitter), 0.0, 0.95))

        if "cook_steps" in config:
            nom = float(config["cook_steps"])
        elif "cook_steps_min" in config or "cook_steps_max" in config:
            lo = float(config.get("cook_steps_min", config.get("cook_steps_max", cls.COOK_STEPS_DEFAULT)))
            hi = float(config.get("cook_steps_max", lo))
            if hi < lo:
                lo, hi = hi, lo
            return max(1, int(round(float(np.random.uniform(lo, hi)))))
        else:
            nom = float(cls.COOK_STEPS_DEFAULT)

        nom = max(1.0, nom)
        lo = nom * (1.0 - jitter)
        hi = nom * (1.0 + jitter)
        return max(1, int(round(float(np.random.uniform(lo, hi)))))


    @staticmethod
    def _compose_yaw_qpos(base_qpos: Sequence[float], yaw: float) -> list[float]:
        """Lay flat with ``base_qpos``, then rotate about world +Z (table surface).

        World-Z yaw must pre-multiply the base orientation. Post-multiplying applies
        yaw in the asset's local frame and tips pans/boards off the table.
        """
        yaw_q = t3d.euler.euler2quat(0.0, 0.0, float(yaw))
        return list(t3d.quaternions.qmult(yaw_q, list(base_qpos)))

    def _aabb_clear(
        self,
        aabb: AABB,
        avoid_aabbs: Sequence[AABB],
        view_z: float,
        min_gap: float = 0.0,
    ) -> bool:
        """True when ``aabb`` clears all avoid boxes and stays in head view."""
        gap = min(
            (self._aabb_gap(aabb, other) for other in avoid_aabbs),
            default=float("inf"),
        )
        return gap > float(min_gap) and self._footprint_in_head_view(aabb, view_z)

    def _sample_mirrored_prop_pose(
        self,
        *,
        side: float,
        nom_x: float,
        nom_y: float,
        base_qpos: Sequence[float],
        collision_path: str,
        scale: float,
        avoid_aabbs: Sequence[AABB],
        padding: float,
        view_z: float,
        jitter_xy: float,
        yaw_lim: float,
        tries: int = 80,
    ) -> tuple[sapien.Pose | None, AABB | None, list[float] | None]:
        """Place a prop from a right-side nominal, mirrored when ``side < 0``.

        Applies small XY jitter and a random Z yaw, rejecting overlaps.
        """
        for _ in range(tries):
            jx = float(np.random.uniform(-jitter_xy, jitter_xy))
            jy = float(np.random.uniform(-jitter_xy, jitter_xy))
            yaw = float(np.random.uniform(-yaw_lim, yaw_lim))
            qpos = self._compose_yaw_qpos(base_qpos, yaw)
            x = float(side) * (float(nom_x) + jx)
            y = float(nom_y) + jy
            pose = sapien.Pose([x, y, 0.741], qpos)
            aabb = self._mesh_aabb_xy(collision_path, pose, scale, padding=padding)
            if self._aabb_clear(aabb, avoid_aabbs, view_z, min_gap=0.0):
                return pose, aabb, qpos
        return None, None, None

    # ---------------------------------------------------------------- actors
    def _spawn_station(
        self,
        side: float,
        tag: str,
        avoid_aabbs: list[AABB],
        bz: float,
    ) -> tuple[dict[str, Any], list[AABB]]:
        """Spawn one cook station (pan, optional key, board, steak) on ``side``.

        Layout is authored for the right side and mirrored for the left (Opt 2).
        Pan and board each get small XY jitter and a random Z yaw. Cook keys sit
        on the outer lateral side, clear of the pan bowl and board.
        """
        prop_pad = float(self.PROP_CLEARANCE)
        jitter_xy = float(getattr(self, "place_jitter_xy", self.PLACE_JITTER_XY_DEFAULT))
        yaw_lim = float(getattr(self, "place_yaw_lim", self.PLACE_YAW_LIM_DEFAULT))
        skillet_id = int(np.random.choice([0, 1, 2, 3]))
        skillet_path = f"assets/objects/106_skillet/collision/base{skillet_id}.glb"
        skillet_scale = self._applied_scale("106_skillet", skillet_id, self.pan_scale)

        # Dual: fixed |x| so bowls keep ≥10 cm; single uses the tighter right/left nominal.
        pan_nom_x = self.DUAL_PAN_NOM_X if self.dual_setup_enabled else self.PAN_NOM_X
        skillet_pose = skillet_aabb = skillet_qpos = None
        # Sample pan with bowl-vs-avoid clearance (full skillet mesh AABB includes the
        # handle and was blocking the mirrored dual station).
        for _ in range(120):
            jx = float(np.random.uniform(-jitter_xy, jitter_xy))
            jy = float(np.random.uniform(-jitter_xy, jitter_xy))
            yaw = float(np.random.uniform(-yaw_lim, yaw_lim))
            qpos = self._compose_yaw_qpos(self.SKILLET_BASE_QPOS, yaw)
            x = float(side) * (float(pan_nom_x) + jx)
            y = float(self.PAN_NOM_Y) + jy
            pose = sapien.Pose([x, y, 0.741], qpos)
            mesh_aabb = self._mesh_aabb_xy(
                skillet_path, pose, skillet_scale, padding=prop_pad
            )
            br = float(self.PAN_BOWL_HALF)
            bowl = (x - br, y - br, x + br, y + br)
            # Dual sibling avoid is already expand(bowl, DUAL_CLEARANCE) — compare the
            # raw bowl (do not also apply PROP_CLEARANCE or the 2nd pan never fits).
            if self.dual_setup_enabled:
                gap = min(
                    (self._aabb_gap(bowl, other) for other in avoid_aabbs),
                    default=float("inf"),
                )
            else:
                bowl_pad = self._expand_aabb(bowl, prop_pad)
                gap = min(
                    (self._aabb_gap(bowl_pad, other) for other in avoid_aabbs),
                    default=float("inf"),
                )
            # View-gate the real bowl (or full mesh); padded bowl often clips the
            # head-camera frame at dual-station |x| even when the pan is visible.
            in_view = self._footprint_in_head_view(
                mesh_aabb, bz
            ) or self._footprint_in_head_view(bowl, bz)
            if gap > 0.0 and in_view:
                skillet_pose, skillet_aabb, skillet_qpos = pose, mesh_aabb, qpos
                break
        if skillet_pose is None or skillet_aabb is None or skillet_qpos is None:
            raise UnStableError(
                f"cook_meat: pan not placeable ({tag}, seed {self._ep_seed}) -- skip"
            )
        skillet = create_actor(
            self,
            pose=skillet_pose,
            modelname="106_skillet",
            model_id=skillet_id,
            convex=True,
            is_static=True,
            scale_mult=self.pan_scale,
        )
        skillet_name = f"106_skillet_{tag}"
        skillet.set_name(skillet_name)
        # Mesh AABB includes the handle; board/key only need the bowl footprint clear.
        px, py = float(skillet_pose.p[0]), float(skillet_pose.p[1])
        br = float(self.PAN_BOWL_HALF)
        bowl_aabb: AABB = (px - br, py - br, px + br, py + br)
        avoid = list(avoid_aabbs) + [self._expand_aabb(bowl_aabb, prop_pad)]

        # Cutting board: nominal offset from pan (outer ±x, +y), then jitter + yaw.
        board_scale_mult = self.board_scale_mult
        board_applied_scale = self._applied_scale("104_board", 0, board_scale_mult)
        board_path = "assets/objects/104_board/collision/base0.glb"
        board_nom_x = abs(px) + self.BOARD_OFF_X
        board_nom_y = py + self.BOARD_OFF_Y
        board_pose, board_aabb, board_qpos = self._sample_mirrored_prop_pose(
            side=side,
            nom_x=board_nom_x,
            nom_y=board_nom_y,
            base_qpos=self.BOARD_BASE_QPOS,
            collision_path=board_path,
            scale=board_applied_scale,
            avoid_aabbs=avoid,
            padding=prop_pad,
            view_z=bz,
            jitter_xy=jitter_xy,
            yaw_lim=yaw_lim,
        )
        if board_pose is None or board_aabb is None or board_qpos is None:
            # Fallback: absolute right-side board nominal (still mirrored by side).
            board_pose, board_aabb, board_qpos = self._sample_mirrored_prop_pose(
                side=side,
                nom_x=pan_nom_x + self.BOARD_OFF_X,
                nom_y=self.PAN_NOM_Y + self.BOARD_OFF_Y,
                base_qpos=self.BOARD_BASE_QPOS,
                collision_path=board_path,
                scale=board_applied_scale,
                avoid_aabbs=avoid,
                padding=prop_pad,
                view_z=bz,
                jitter_xy=jitter_xy,
                yaw_lim=yaw_lim,
                tries=120,
            )
        if board_pose is None or board_aabb is None or board_qpos is None:
            raise UnStableError(
                f"cook_meat: board not placeable ({tag}, seed {self._ep_seed}) -- skip"
            )
        board_xy = board_pose.p[:2]
        probe_pose = sapien.Pose([0.0, 0.0, 0.0], board_qpos)
        board_z_min, board_z_max = self._mesh_world_z_extent(
            board_path, probe_pose, board_applied_scale
        )
        board_th = board_z_max - board_z_min
        board_spawn_z = bz - board_z_min
        board_pose = sapien.Pose(
            [float(board_xy[0]), float(board_xy[1]), board_spawn_z], board_qpos
        )
        board_aabb = self._mesh_aabb_xy(
            board_path, board_pose, board_applied_scale, padding=prop_pad
        )
        with open(
            "assets/objects/104_board/model_data0.json", encoding="utf-8"
        ) as board_data_file:
            board_data = json.load(board_data_file)
        board = create_actor(
            self,
            pose=board_pose,
            modelname="104_board",
            model_id=0,
            convex=True,
            is_static=True,
            scale_mult=board_scale_mult,
        )
        board.set_name(f"104_board_{tag}")
        board.config = {
            "scale": [board_scale_mult, board_scale_mult, board_scale_mult],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        board_top = bz + board_th
        avoid = list(avoid) + [self._expand_aabb(board_aabb, prop_pad)]

        steak_pose = rand_pose(
            xlim=[float(board_xy[0]), float(board_xy[0])],
            ylim=[float(board_xy[1]), float(board_xy[1])],
            zlim=[board_top],
            qpos=self._compose_yaw_qpos(
                self.BOARD_BASE_QPOS, float(np.random.uniform(-yaw_lim, yaw_lim))
            ),
            rotate_rand=False,
        )
        steak = create_actor(
            self,
            pose=steak_pose,
            modelname="200_steak",
            model_id=0,
            convex=True,
            is_static=False,
            scale_mult=(1.0, self.steak_thick, 1.0),
        )
        steak_name = f"200_steak_{tag}"
        steak.set_name(steak_name)
        steak.set_mass(0.05)

        # Cook key on the outer lateral side of the pan, biased toward the robot (−y)
        # so it stays clear of the board (+y) and does not overlap the bowl.
        cook_key = None
        cook_key_base = None
        key_xy = None
        key_top_z = None
        key_aabb: AABB | None = None
        if self.cook_button_enabled:
            pan_xy = np.asarray(skillet_pose.p[:2], dtype=np.float64)
            hx, hy, _ = self.KEY_BASE_HALF
            clear_r = float(self.KEY_PAN_CLEAR_HALF)
            key_gap = float(self.KEY_MIN_GAP)
            lat_sign = -1.0 if side < 0 else 1.0
            lat0 = max(clear_r + hx + key_gap, abs(self.key_offset_x))
            # Clear the bowl rim; board already sits in ``avoid``.
            key_avoid = list(avoid) + [
                (
                    float(pan_xy[0] - clear_r),
                    float(pan_xy[1] - clear_r),
                    float(pan_xy[0] + clear_r),
                    float(pan_xy[1] + clear_r),
                )
            ]
            found = False
            key_x = key_y = 0.0
            # Search outward (±x) and toward the robot (−y). Do not require head-cam
            # containment for the key — left-side outer slots often leave the frame
            # and were rejecting every dual+button seed.
            dx_vals = list(np.linspace(lat0, lat0 + 0.18, 12))
            dy_vals = list(np.linspace(float(self.key_y_bias), float(self.key_y_bias) - 0.16, 10))
            # Also try near pan y if the board blocks the −y corridor.
            dy_vals += list(np.linspace(0.02, -0.06, 5))
            for require_view in (True, False):
                for dx in dx_vals:
                    for dy in dy_vals:
                        kx = float(pan_xy[0] + lat_sign * dx)
                        ky = float(pan_xy[1] + dy)
                        # Keep key on the table footprint.
                        if abs(kx) > 0.42 or ky < -0.28 or ky > 0.22:
                            continue
                        cand: AABB = (kx - hx, ky - hy, kx + hx, ky + hy)
                        gap = min(
                            (self._aabb_gap(cand, other) for other in key_avoid),
                            default=float("inf"),
                        )
                        if gap <= key_gap:
                            continue
                        if require_view and not self._footprint_in_head_view(cand, bz):
                            continue
                        key_x, key_y = kx, ky
                        key_aabb = cand
                        found = True
                        break
                    if found:
                        break
                if found:
                    break
            if not found or key_aabb is None:
                raise UnStableError(
                    f"cook_meat: cook key not placeable ({tag}, seed {self._ep_seed}) -- skip"
                )
            base_hz = float(self.KEY_BASE_HALF[2])
            cap_hz = float(self.KEY_HALF[2])
            base_z = bz + base_hz
            cap_z = bz + 2.0 * base_hz + cap_hz
            cook_key_base = create_box(
                self,
                pose=sapien.Pose([key_x, key_y, base_z], [1, 0, 0, 0]),
                half_size=list(self.KEY_BASE_HALF),
                color=list(self.KEY_BASE_COLOR),
                name=f"cook_key_base_{tag}",
                is_static=True,
            )
            cook_key = create_box(
                self,
                pose=sapien.Pose([key_x, key_y, cap_z], [1, 0, 0, 0]),
                half_size=list(self.KEY_HALF),
                color=list(self.KEY_COLOR),
                name=f"cook_key_{tag}",
                is_static=True,
            )
            key_xy = (key_x, key_y)
            key_top_z = float(bz + 2.0 * base_hz + 2.0 * cap_hz)
            avoid.append(self._expand_aabb(key_aabb, key_gap))
            self.add_prohibit_area(cook_key_base, padding=0.02)
            self.add_prohibit_area(cook_key, padding=0.02)

        self.add_prohibit_area(skillet, padding=0.05)
        self.add_prohibit_area(board, padding=0.03)
        self.add_prohibit_area(steak, padding=0.03)

        steak_shapes: list[Any] = []
        for c in steak.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                steak_shapes = list(c.render_shapes)

        station = {
            "tag": tag,
            "side": float(side),
            "arm": ArmTag("right" if side > 0 else "left"),
            "skillet": skillet,
            "skillet_id": skillet_id,
            "skillet_name": skillet_name,
            "board": board,
            "board_xy": (float(board_xy[0]), float(board_xy[1])),
            "board_top": float(board_top),
            "steak": steak,
            "steak_name": steak_name,
            "cook_key": cook_key,
            "cook_key_base": cook_key_base,
            "key_xy": key_xy,
            "key_top_z": key_top_z,
            "steak_shapes": steak_shapes,
            "doneness": 0.0,
            "max_doneness": 0.0,
            "grasp_doneness": None,
            "cooking_active": False,
            "awaiting_return_grasp": False,
            "cook_phase_done": False,
        }
        self._set_station_meat_color(station, 0.0)
        # Sibling dual station only needs bowl↔bowl ≥10 cm. Board/key sit outward
        # (+y / outer ±x); folding them into the mid-line keep-out made the 2nd pan
        # unplaceable once XY jitter shrank.
        keep_parts = [bowl_aabb, board_aabb]
        if key_aabb is not None:
            keep_parts.append(key_aabb)
        station_keepout = self._union_aabb(keep_parts)
        out_avoid = list(avoid_aabbs)
        if self.dual_setup_enabled:
            out_avoid.append(
                self._expand_aabb(bowl_aabb, self.DUAL_STATION_CLEARANCE)
            )
        else:
            out_avoid.append(station_keepout)
        return station, out_avoid

    def load_actors(self) -> None:
        """Create the randomized task layout and reset cooking state."""
        config = self._cook_cfg
        self.cook_steps = self._sample_cook_steps(config)
        self.target_doneness = float(
            np.random.uniform(
                config.get("target_doneness_min", 0.45),
                config.get("target_doneness_max", 0.55),
            )
        )
        self.cook_doneness_tol = float(
            config.get("cook_doneness_tol", self.COOK_DONENESS_TOL_DEFAULT)
        )
        self.cook_button_enabled = bool(
            config.get("cook_button_enabled", self.COOK_BUTTON_ENABLED_DEFAULT)
        )
        self.dual_setup_enabled = bool(
            config.get("dual_setup_enabled", self.DUAL_SETUP_ENABLED_DEFAULT)
        )
        self.key_offset_x = float(config.get("key_offset_x", self.KEY_OFFSET_X_DEFAULT))
        self.key_y_bias = float(config.get("key_y_bias", self.KEY_Y_BIAS_DEFAULT))
        self.key_press_depth = float(
            config.get("key_press_depth", self.KEY_PRESS_DEPTH_DEFAULT)
        )
        self.key_press_xy = float(config.get("key_press_xy", self.KEY_PRESS_XY_DEFAULT))
        self.key_press_dz = float(config.get("key_press_dz", self.KEY_PRESS_DZ_DEFAULT))
        self.key_hover_dis = float(
            config.get("key_hover_dis", self.KEY_HOVER_DIS_DEFAULT)
        )
        self.pan_scale = float(config.get("pan_scale", 1.0))
        self.place_dx = float(config.get("place_dx", 0.0))
        self.place_dy = float(config.get("place_dy", 0.0))
        self.board_scale_mult = float(config.get("board_scale_mult", 0.07))
        self.steak_thick = float(config.get("steak_thick", 1.6))
        self.place_jitter_xy = float(
            config.get("place_jitter_xy", self.PLACE_JITTER_XY_DEFAULT)
        )
        self.place_yaw_lim = float(
            config.get("place_yaw_lim", self.PLACE_YAW_LIM_DEFAULT)
        )
        self.board_th = 0.0

        bz = 0.74 + self.table_z_bias
        avoid: list[AABB] = []
        self.stations: list[dict[str, Any]] = []

        if self.dual_setup_enabled:
            # Both sides: left then right so arms each own a full mirrored station.
            for side, tag in ((-1.0, "left"), (1.0, "right")):
                st, avoid = self._spawn_station(side, tag, avoid, bz)
                self.stations.append(st)
            self._side = 0.0
        else:
            side = -1.0 if (self._ep_seed % 2 == 0) else 1.0
            self._side = side
            tag = "left" if side < 0 else "right"
            st, avoid = self._spawn_station(side, tag, avoid, bz)
            self.stations.append(st)

        # Primary aliases for single-station helpers / dynamic path / older tests.
        primary = self.stations[0]
        self.skillet = primary["skillet"]
        self.skillet_id = primary["skillet_id"]
        self.board = primary["board"]
        self.steak = primary["steak"]
        self.cook_key = primary["cook_key"]
        self.cook_key_base = primary["cook_key_base"]
        self._key_xy = primary["key_xy"]
        self._key_top_z = primary["key_top_z"]
        self._steak_shapes = primary["steak_shapes"]
        self.doneness = primary["doneness"]
        self.max_doneness = primary["max_doneness"]
        self._grasp_doneness = primary["grasp_doneness"]
        self._cooking_active = False
        # Legacy alias used by older unit tests / success helpers.
        self.plate = primary["board"]

    # -------------------------------------------------------- cooking state
    def _set_station_meat_color(self, station: dict[str, Any], doneness: float) -> None:
        """Update one station's steak render shapes for the supplied doneness."""
        d = float(np.clip(doneness, 0.0, 1.0))
        stops = self.COLOR_STOPS
        rgb = stops[-1][1]
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        color = list(rgb) + [1.0]
        for render_shape in station.get("steak_shapes", []):
            try:
                render_shape.material.set_base_color(color)
            except Exception:
                pass

    def _set_meat_color(self, doneness: float) -> None:
        """Update the primary steak color (single-station / test helper)."""
        if getattr(self, "stations", None):
            self._set_station_meat_color(self.stations[0], doneness)
            self.stations[0]["doneness"] = float(doneness)
            return
        # Fallback for unit tests that never call load_actors.
        st = {"steak_shapes": getattr(self, "_steak_shapes", [])}
        self._set_station_meat_color(st, doneness)

    def _steak_on_pan_station(self, station: dict[str, Any]) -> bool:
        """Return whether this station's steak contacts its skillet."""
        try:
            return bool(
                self.check_actors_contact(station["steak_name"], station["skillet_name"])
            )
        except Exception:
            return False

    def _steak_on_pan(self) -> bool:
        """Primary-station contact helper."""
        if getattr(self, "stations", None):
            return self._steak_on_pan_station(self.stations[0])
        try:
            return bool(self.check_actors_contact("200_steak", "106_skillet"))
        except Exception:
            return False

    def _steak_held(self, station: dict[str, Any]) -> bool:
        """True when a gripper finger is in contact with this station's steak."""
        try:
            contacts = self.get_gripper_actor_contact_position(station["steak_name"])
            return len(contacts) > 0
        except Exception:
            return False

    def _latch_grasp_doneness(self, station: dict[str, Any], *, force: bool = False) -> None:
        """Freeze this station's cook state once its steak is actually grasped.

        Per-station: grasping one dual-station steak must not freeze the other.
        """
        if station.get("grasp_doneness") is not None:
            return
        if not force and not self._steak_held(station):
            return
        station["grasp_doneness"] = float(station["doneness"])
        station["cooking_active"] = False
        station["awaiting_return_grasp"] = False
        station["cook_phase_done"] = True

    def _button_is_pressed_station(self, station: dict[str, Any]) -> bool:
        """True when the matching arm's EE is pressing this station's cook key."""
        if not self.use_cook_button or station.get("cook_key") is None:
            return False
        key_xy = station.get("key_xy")
        key_top_z = station.get("key_top_z")
        if key_xy is None or key_top_z is None or not hasattr(self, "robot"):
            return False
        if bool(station.get("_expert_key_held")):
            return True
        arm = station["arm"]
        get_ee = (
            self.robot.get_left_ee_pose
            if arm == "left"
            else self.robot.get_right_ee_pose
        )
        try:
            ee = np.asarray(get_ee()[:3], dtype=float)
        except Exception:
            return False
        bx, by = key_xy
        return (
            abs(float(ee[0]) - bx) < self.key_press_xy
            and abs(float(ee[1]) - by) < self.key_press_xy
            and float(ee[2]) < float(key_top_z) + self.key_press_dz
        )

    def _button_is_pressed(self) -> bool:
        """True if any station's cook key is pressed (obs / tests)."""
        if not self.use_cook_button:
            return False
        for st in getattr(self, "stations", []) or []:
            if self._button_is_pressed_station(st):
                return True
        return False

    def _advance_station_cook(self, station: dict[str, Any]) -> None:
        """Advance one station's doneness by one cook tick and recolor."""
        station["doneness"] = min(
            1.0, float(station["doneness"]) + 1.0 / max(1, self.cook_steps)
        )
        station["max_doneness"] = max(
            float(station["max_doneness"]), float(station["doneness"])
        )
        self._set_station_meat_color(station, station["doneness"])

    def _update_kinematic_tasks(self) -> None:
        """Advance base dynamics and per-station cooking state by one step."""
        super()._update_kinematic_tasks()
        stations = getattr(self, "stations", None) or []
        if not stations:
            return
        for st in stations:
            # Latch cook freeze only when THIS steak is actually held (not on approach).
            if (
                st.get("awaiting_return_grasp")
                and st.get("grasp_doneness") is None
                and self._steak_held(st)
            ):
                self._latch_grasp_doneness(st, force=True)
            if st.get("grasp_doneness") is not None:
                continue
            # After the cook wait finishes, freeze further doneness changes until
            # THIS steak is grasped (stops lingering key contact from overcooking).
            if st.get("cook_phase_done"):
                continue
            if self.use_cook_button:
                if self._button_is_pressed_station(st) and self._steak_on_pan_station(st):
                    self._advance_station_cook(st)
            elif st.get("cooking_active"):
                if self._steak_on_pan_station(st):
                    self._advance_station_cook(st)
        # Keep primary aliases in sync for obs / success helpers.
        primary = stations[0]
        self.doneness = float(primary["doneness"])
        self.max_doneness = float(primary["max_doneness"])
        self._grasp_doneness = primary["grasp_doneness"]

    def _cook_idle(self) -> None:
        """Step until every station reaches target doneness (or dwell limit)."""
        max_steps = int(round(self.target_doneness * self.cook_steps)) + 30
        for i in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if all(
                float(st["doneness"]) >= self.target_doneness for st in self.stations
            ):
                break

    def _move_pair(self, left_action, right_action=None) -> None:
        """Move one or both arms (``right_action`` optional for single-station)."""
        if right_action is None:
            self.move(left_action)
        else:
            self.move(left_action, right_action)

    def _cook_key_tip_pose(
        self, station: dict[str, Any], tip_z_above_top: float
    ) -> list[float]:
        """Top-down EE pose above a station's cook key (dispense_gummy-style)."""
        key_xy = station.get("key_xy")
        key_top_z = station.get("key_top_z")
        if key_xy is None or key_top_z is None:
            raise RuntimeError("cook_meat: cook key pose missing while button mode on")
        tcp_z = float(key_top_z) + float(tip_z_above_top)
        ee_z = tcp_z + float(self.EE_TO_TCP)
        return [
            float(key_xy[0]),
            float(key_xy[1]),
            ee_z,
            *GRASP_DIRECTION_DIC["top_down"],
        ]

    def _safe_grasp_actor(self, actor: Any, arm_tag: ArmTag, **kwargs: Any):
        """``grasp_actor`` that skips the seed when no contact pose exists."""
        pre_pose, grasp_pose = self.choose_grasp_pose(
            actor,
            arm_tag=arm_tag,
            pre_dis=float(kwargs.get("pre_grasp_dis", 0.1)),
            target_dis=float(kwargs.get("grasp_dis", 0.0)),
            contact_point_id=kwargs.get("contact_point_id"),
        )
        if pre_pose is None or grasp_pose is None:
            raise UnStableError(
                f"cook_meat: no grasp pose (seed {self._ep_seed}) -- skip"
            )
        return self.grasp_actor(actor, arm_tag=arm_tag, **kwargs)

    def _press_cook_buttons(self) -> None:
        """Press and hold each station's cook key until target doneness, then release.

        Uses absolute ``move_to_pose`` hover/press targets (not ``grasp_actor``).
        CuRobo often stops a few cm above the key mesh; press detection uses a
        generous ``key_press_dz`` so a near-contact EE still counts as held.
        """
        for st in self.stations:
            if st["cook_key"] is None:
                raise RuntimeError("cook_meat: cook_key missing while button mode on")

        hover = float(getattr(self, "key_hover_dis", self.KEY_HOVER_DIS_DEFAULT))
        # Target TCP near the key top; EE frame is EE_TO_TCP above TCP.
        press_above = max(0.0, hover - float(self.key_press_depth))

        if len(self.stations) == 1:
            st = self.stations[0]
            arm = st["arm"]
            self.move(self.close_gripper(arm))
            self.move(self.move_to_pose(arm, self._cook_key_tip_pose(st, hover)))
            self.move(self.move_to_pose(arm, self._cook_key_tip_pose(st, press_above)))
            # Always latch expert hold for the cook wait (EE may sit slightly above key).
            st["_expert_key_held"] = True
            self._cook_idle()
            # Freeze cooking before releasing the key / moving away.
            st["cook_phase_done"] = True
            st["awaiting_return_grasp"] = True
            st["_expert_key_held"] = False
            self.move(self.move_to_pose(arm, self._cook_key_tip_pose(st, hover)))
            self.move(self.open_gripper(arm))
            return

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        la, ra = left["arm"], right["arm"]
        self.move(self.close_gripper(la), self.close_gripper(ra))
        self.move(
            self.move_to_pose(la, self._cook_key_tip_pose(left, hover)),
            self.move_to_pose(ra, self._cook_key_tip_pose(right, hover)),
        )
        self.move(
            self.move_to_pose(la, self._cook_key_tip_pose(left, press_above)),
            self.move_to_pose(ra, self._cook_key_tip_pose(right, press_above)),
        )
        for st in self.stations:
            st["_expert_key_held"] = True
        self._cook_idle()
        for st in self.stations:
            st["cook_phase_done"] = True
            st["awaiting_return_grasp"] = True
            st["_expert_key_held"] = False
        self.move(
            self.move_to_pose(la, self._cook_key_tip_pose(left, hover)),
            self.move_to_pose(ra, self._cook_key_tip_pose(right, hover)),
        )
        self.move(self.open_gripper(la), self.open_gripper(ra))

    # ------------------------------------------------------------- policy
    def _dbg(self, tag: str) -> None:
        """Print opt-in planner diagnostics for task tuning."""
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_meat] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self) -> dict[str, Any]:
        """Run the dynamic or static expert trajectory for one episode."""
        if self.dual_setup_enabled:
            # Dual stations need both arms; skip dynamic moving-target path.
            return self._play_once_static()
        if self.use_dynamic:
            return self._play_once_dynamic()
        return self._play_once_static()

    def _place_steaks_on_pans(self) -> None:
        """Place each held steak into its skillet bowl (parallel when dual)."""
        def _place_one(st: dict[str, Any]) -> None:
            arm = st["arm"]
            pan_target = list(st["skillet"].get_functional_point(0))
            pan_target[0] += self.place_dx
            pan_target[1] += self.place_dy
            self.move(
                self.place_actor(
                    st["steak"],
                    target_pose=pan_target,
                    arm_tag=arm,
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.02,
                    is_open=True,
                )
            )
            self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="arm"))

        if len(self.stations) == 1:
            st = self.stations[0]
            _place_one(st)
            if not self._steak_on_pan_station(st):
                # One retry — first place often drops short after yawed board grasp.
                _place_one(st)
            if not self._steak_on_pan_station(st):
                raise UnStableError(
                    f"cook_meat: steak not on pan after place (seed {self._ep_seed}) -- skip"
                )
            return

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        actions = []
        for st in (left, right):
            pan_target = list(st["skillet"].get_functional_point(0))
            pan_target[0] += self.place_dx
            pan_target[1] += self.place_dy
            actions.append(
                self.place_actor(
                    st["steak"],
                    target_pose=pan_target,
                    arm_tag=st["arm"],
                    constrain="free",
                    pre_dis=0.10,
                    dis=0.02,
                    is_open=True,
                )
            )
        self.move(actions[0], actions[1])
        self.move(
            self.move_by_displacement(arm_tag=left["arm"], z=0.10, move_axis="arm"),
            self.move_by_displacement(arm_tag=right["arm"], z=0.10, move_axis="arm"),
        )
        missing = [st for st in (left, right) if not self._steak_on_pan_station(st)]
        for st in missing:
            _place_one(st)
        if any(not self._steak_on_pan_station(st) for st in (left, right)):
            raise UnStableError(
                f"cook_meat: steak not on pan after place (seed {self._ep_seed}) -- skip"
            )

    def _return_steaks_to_boards(self) -> None:
        """Grasp cooked steaks off pans and set them back on their cutting boards.

        Dual: both arms grasp together and place together so one steak does not
        keep cooking while the other is returned.
        """
        def _board_target(st: dict[str, Any]) -> list[float]:
            # Live board pose (more reliable than cached XY after any sim drift).
            p = st["board"].get_pose().p
            return [float(p[0]), float(p[1]), float(st["board_top"]) + 0.03]

        def _d_board(st: dict[str, Any]) -> float:
            sxy = np.asarray(st["steak"].get_pose().p[:2], dtype=float)
            bxy = np.asarray(st["board"].get_pose().p[:2], dtype=float)
            return float(np.linalg.norm(sxy - bxy))

        def _latch_after_grasp(st: dict[str, Any]) -> None:
            self._latch_grasp_doneness(st, force=True)

        def _place_on_board(st: dict[str, Any]) -> None:
            arm = st["arm"]
            place_kwargs = dict(
                arm_tag=arm,
                constrain="free",
                pre_dis=0.10,
                dis=0.015,
                is_open=True,
            )
            self.move(self.place_actor(st["steak"], target_pose=_board_target(st), **place_kwargs))
            # Retries: left-arm places often undershoot (~0.17–0.21 m from board).
            for dis in (0.01, 0.0):
                if _d_board(st) < 0.12:
                    break
                self.move(self._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.08))
                self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
                self.move(
                    self.place_actor(
                        st["steak"],
                        target_pose=_board_target(st),
                        arm_tag=arm,
                        constrain="free",
                        pre_dis=0.12,
                        dis=dis,
                        is_open=True,
                    )
                )

        if len(self.stations) == 1:
            st = self.stations[0]
            arm = st["arm"]
            st["awaiting_return_grasp"] = True
            self.move(self.open_gripper(arm))
            self.move(self._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.1))
            _latch_after_grasp(st)
            self.move(self.move_by_displacement(arm_tag=arm, z=0.12, move_axis="arm"))
            _place_on_board(st)
            self.move(self.move_by_displacement(arm_tag=arm, z=0.08))
            return

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        for st in (left, right):
            st["awaiting_return_grasp"] = True
        self.move(self.open_gripper(left["arm"]), self.open_gripper(right["arm"]))
        self.move(
            self._safe_grasp_actor(left["steak"], arm_tag=left["arm"], pre_grasp_dis=0.1),
            self._safe_grasp_actor(right["steak"], arm_tag=right["arm"], pre_grasp_dis=0.1),
        )
        _latch_after_grasp(left)
        _latch_after_grasp(right)
        self.move(
            self.move_by_displacement(arm_tag=left["arm"], z=0.12, move_axis="arm"),
            self.move_by_displacement(arm_tag=right["arm"], z=0.12, move_axis="arm"),
        )
        # Place sequentially so one arm's retry cannot collide with the other.
        for st in (left, right):
            _place_on_board(st)
            self.move(self.move_by_displacement(arm_tag=st["arm"], z=0.08))

    def _pan_cook_table(self) -> None:
        """Cook held steak(s) on pan(s), then return them to the cutting board(s)."""
        self._place_steaks_on_pans()
        self._dbg("place_in_pan")

        if self.use_cook_button:
            self._press_cook_buttons()
        else:
            if len(self.stations) == 1:
                self.move(self.back_to_origin(self.stations[0]["arm"]))
            else:
                left = next(st for st in self.stations if st["arm"] == "left")
                right = next(st for st in self.stations if st["arm"] == "right")
                self.move(self.back_to_origin(left["arm"]), self.back_to_origin(right["arm"]))
            for st in self.stations:
                st["cooking_active"] = True
            self._cook_idle()
            for st in self.stations:
                st["cook_phase_done"] = True
                st["awaiting_return_grasp"] = True
            # grasp_doneness latches per-station when each steak is actually grasped.

        self._return_steaks_to_boards()
        self._dbg("returned_to_board")

    def _task_info(self, arm_tag: ArmTag | None = None) -> dict[str, str]:
        """Build the language-template substitutions recorded for this episode."""
        primary = self.stations[0] if getattr(self, "stations", None) else None
        skillet_id = (
            primary["skillet_id"]
            if primary is not None
            else getattr(self, "skillet_id", 0)
        )
        if arm_tag is None:
            if self.dual_setup_enabled:
                arm_str = "both arms"
            elif primary is not None:
                arm_str = str(primary["arm"])
            else:
                arm_str = "left"
        else:
            arm_str = str(arm_tag)
        info = {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{skillet_id}",
            "{C}": "104_board/base0",
            "{a}": arm_str,
            "{o}": self._option_label(),
        }
        if self.use_cook_button:
            info["{E}"] = "cook_key"
        if self.dual_setup_enabled:
            info["{n}"] = "2"
        return info

    def _play_once_static(self) -> dict[str, Any]:
        """Execute the expert trajectory for stationary steak(s)."""
        if len(self.stations) == 1:
            st = self.stations[0]
            arm = st["arm"]
            self.move(self._safe_grasp_actor(st["steak"], arm_tag=arm, pre_grasp_dis=0.1))
            self.move(self.move_by_displacement(arm_tag=arm, z=0.1, move_axis="arm"))
            self._pan_cook_table()
            self.info["info"] = self._task_info(arm)
            return self.info

        left = next(st for st in self.stations if st["arm"] == "left")
        right = next(st for st in self.stations if st["arm"] == "right")
        self.move(
            self._safe_grasp_actor(left["steak"], arm_tag=left["arm"], pre_grasp_dis=0.1),
            self._safe_grasp_actor(right["steak"], arm_tag=right["arm"], pre_grasp_dis=0.1),
        )
        self.move(
            self.move_by_displacement(arm_tag=left["arm"], z=0.1, move_axis="arm"),
            self.move_by_displacement(arm_tag=right["arm"], z=0.1, move_axis="arm"),
        )
        self._pan_cook_table()
        self.info["info"] = self._task_info()
        return self.info

    def get_dynamic_motion_config(self) -> dict[str, Any] | None:
        """Return the base workflow's dynamic-motion configuration when enabled."""
        if not self.use_dynamic or self.dual_setup_enabled:
            return None
        p = self.steak.get_pose().p
        return {
            "target_actor": self.steak,
            "end_position": np.array([p[0], p[1], p[2]]),
            "table_bounds": (-0.35, 0.35, -0.25, 0.15),
            "check_z_threshold": 0.03,
            "check_z_actor": self.steak,
        }

    def _play_once_dynamic(self) -> dict[str, Any]:
        """Execute moving-target acquisition followed by the cooking trajectory."""
        st = self.stations[0]
        arm_tag = st["arm"]
        p = st["steak"].get_pose().p
        self.end_position = np.array([p[0], p[1], p[2]])

        def robot_action_sequence(need_plan_mode: bool = False) -> None:
            _ = need_plan_mode
            grasp_result = self.grasp_actor(
                st["steak"], arm_tag=arm_tag, pre_grasp_dis=0.1
            )
            if (
                not grasp_result
                or grasp_result[1] is None
                or len(grasp_result[1]) == 0
            ):
                return
            self.move(grasp_result)

        table_bounds = (-0.35, 0.35, -0.25, 0.15)
        success, _ = self.execute_dynamic_workflow(
            target_actor=st["steak"],
            end_position=self.end_position,
            robot_action_sequence=robot_action_sequence,
            table_bounds=table_bounds,
        )
        if not success:
            print("Dynamic trajectory failed, fallback to static")
            return self._play_once_static()

        for c in st["steak"].actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_velocity(np.zeros(3))
                    c.set_angular_velocity(np.zeros(3))
                    c.set_linear_damping(15.0)
                    c.set_angular_damping(40.0)
                except Exception:
                    pass

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table()
        self.info["info"] = self._task_info(arm_tag)
        return self.info

    # ------------------------------------------------------------- success
    def _station_success(self, station: dict[str, Any]) -> bool:
        """Return whether one station's steak is back on the board at target doneness.

        Cook quality uses ``grasp_doneness`` (value when cooking stopped) within
        ``target_doneness ± cook_doneness_tol`` so under- and over-cooked meat fail.
        Dual episodes call this once per steak — both must pass.
        """
        if station.get("grasp_doneness") is None:
            return False
        tol = float(getattr(self, "cook_doneness_tol", self.COOK_DONENESS_TOL_DEFAULT))
        g = float(station["grasp_doneness"])
        cooked_ok = abs(g - float(self.target_doneness)) <= tol
        steak_p = station["steak"].get_pose().p
        steak_z = float(steak_p[2])
        steak_xy = np.array(steak_p[:2])
        board_xy = np.array(station.get("board_xy", station["board"].get_pose().p[:2]), dtype=float)
        pan_xy = np.array(station["skillet"].get_functional_point(0)[:2])
        d_board = float(np.linalg.norm(steak_xy - board_xy))
        d_pan = float(np.linalg.norm(steak_xy - pan_xy))
        on_board = d_board < 0.12
        off_pan = d_board < d_pan
        board_top = float(station.get("board_top", 0.74 + self.table_z_bias))
        above_board = steak_z > (board_top - 0.02)
        return bool(cooked_ok and on_board and off_pan and above_board)

    def check_success(self) -> bool:
        """Return whether every steak is correctly cooked and back on its board.

        Single station: one steak in the doneness band, on the board, off the pan.
        Dual (Opt 2 / Opt 1+2): **both** steaks must be cooked properly (each
        ``grasp_doneness`` within ``cook_doneness_tol`` of ``target_doneness``)
        and returned to their own boards — one under-/over-cooked steak fails.
        """
        stations = getattr(self, "stations", None)
        if not stations:
            # Unit-test path that builds a bare task without load_actors.
            if self._grasp_doneness is None:
                return False
            tol = float(getattr(self, "cook_doneness_tol", self.COOK_DONENESS_TOL_DEFAULT))
            cooked_ok = abs(float(self._grasp_doneness) - float(self.target_doneness)) <= tol
            steak_p = self.steak.get_pose().p
            steak_z = float(steak_p[2])
            steak_xy = np.array(steak_p[:2])
            board_xy = np.array(self.board.get_pose().p[:2])
            pan_xy = np.array(self.skillet.get_functional_point(0)[:2])
            d_board = float(np.linalg.norm(steak_xy - board_xy))
            d_pan = float(np.linalg.norm(steak_xy - pan_xy))
            return bool(
                cooked_ok
                and d_board < 0.12
                and d_board < d_pan
                and steak_z > (0.73 + self.table_z_bias)
            )
        if bool(getattr(self, "dual_setup_enabled", False)) and len(stations) != 2:
            return False
        return all(self._station_success(st) for st in stations)

    def get_obs(self) -> dict[str, Any]:
        """Extend the base observation with per-frame cooking measurements."""
        obs = super().get_obs()
        stations = getattr(self, "stations", None) or []
        obs["cooking"] = {
            "doneness": float(getattr(self, "doneness", 0.0)),
            "target_doneness": float(
                getattr(self, "target_doneness", self.TARGET_DONENESS_DEFAULT)
            ),
            "cook_steps": float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)),
            "cook_button_enabled": bool(getattr(self, "cook_button_enabled", False)),
            "dual_setup_enabled": bool(getattr(self, "dual_setup_enabled", False)),
            "use_cook_button": bool(self.use_cook_button),
            "button_pressed": bool(self._button_is_pressed()) if self.use_cook_button else False,
            "n_stations": int(len(stations)),
            "station_doneness": [float(st["doneness"]) for st in stations],
        }
        try:
            if stations:
                sxy = np.array(stations[0]["steak"].get_pose().p)[:2]
                pxy = np.array(stations[0]["skillet"].get_functional_point(0))[:2]
                bxy = np.array(stations[0].get("board_xy", stations[0]["board"].get_pose().p[:2]))
                obs["cooking"]["steak_xy"] = [float(sxy[0]), float(sxy[1])]
                obs["cooking"]["pan_xy"] = [float(pxy[0]), float(pxy[1])]
                obs["cooking"]["place_offset"] = [
                    float(sxy[0] - pxy[0]),
                    float(sxy[1] - pxy[1]),
                ]
                obs["cooking"]["board_xy"] = [float(bxy[0]), float(bxy[1])]
        except Exception:
            pass
        return obs
