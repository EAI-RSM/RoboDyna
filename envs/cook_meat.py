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
from .utils.action import ArmTag
from .utils.create_actor import UnStableError, create_actor
from .utils.rand_create_actor import rand_pose


AABB: TypeAlias = tuple[float, float, float, float]


class cook_meat(Base_Task):
    """Cook a steak to a sampled doneness and place it on a serving plate.

    Doneness advances only while the cooking phase is active and the steak
    contacts the skillet. The visible steak color is interpolated from raw red
    through medium brown to well-done dark brown.
    """

    COOK_STEPS_DEFAULT: ClassVar[int] = 1000
    TARGET_DONENESS_DEFAULT: ClassVar[float] = 0.5
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
        self._cook_cfg = kwargs.get("task_args", {}).get("cook_meat", {})
        self._ep_seed = int(kwargs.get("seed", 0))
        super()._init_task_env_(**kwargs)

    # ------------------------------------------------------- footprint checks
    # rand_pose()/create_actor() do NOT check for overlap between objects, and add_prohibit_area()
    # only feeds the eval-time trajectory-extension check (_base_task.py) -- there is no built-in
    # guard against two hand-placed static props (e.g. the pan and the plate) landing on top of
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
        # scale_mult inflates the footprint by 1/authored_scale (e.g. 40x for the 0.025-scale plate).
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

    # ---------------------------------------------------------------- actors
    def load_actors(self) -> None:
        """Create the randomized task layout and reset cooking state."""
        config = self._cook_cfg
        # KEY per-episode randomization: the COOK SPEED = how many sim steps the steak takes to go
        # from raw(0) to fully done(1.0). Smaller = browns faster. Also a tight doneness band ~medium.
        self.cook_steps = int(
            np.random.uniform(
                config.get("cook_steps_min", 600),
                config.get("cook_steps_max", 1600),
            )
        )
        self.target_doneness = float(
            np.random.uniform(
                config.get("target_doneness_min", 0.45),
                config.get("target_doneness_max", 0.55),
            )
        )
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._cooking_active = False
        self._grasp_doneness = None

        # Pan, board, plate and steak all live on the SAME side so one arm can grasp the steak off
        # its board, cook it on the pan, and set it on the plate (a cross-body reach to the far
        # side is unplannable). Balance left/right by seed parity (np.random.choice was streaky ->
        # all-left on seeds 0-3), so both hands are reliably exercised across the dataset.
        side = -1.0 if (self._ep_seed % 2 == 0) else 1.0
        self._side = side
        bz = 0.74 + self.table_z_bias

        # pan: a heavy static frying pan, kept in the arm's reliable placement zone on the side.
        # scale_mult enlarges the asset (mesh + functional point together) -> bigger bowl = easier,
        # more reliable steak placement (helps the UR5 reach/place yield).
        self.pan_scale = float(config.get("pan_scale", 1.0))
        # placement offset (world x,y) added to the pan functional point so the steak lands centered
        # in the bowl (the grasp offset otherwise lands it off-center) -- tune from the measured offset
        self.place_dx = float(config.get("place_dx", 0.0))
        self.place_dy = float(config.get("place_dy", 0.0))
        self.skillet_id = int(np.random.choice([0, 1, 2, 3]))
        # real collision-mesh footprint (see the footprint-checks note above -- model_data's
        # "extents" is an OMBB that overestimates the skillet's true world footprint by ~35% in x),
        # used both to keep the pan in the camera frame and to keep the plate/board clear of it.
        skillet_path = f"assets/objects/106_skillet/collision/base{self.skillet_id}.glb"
        skillet_scale = self._applied_scale("106_skillet", self.skillet_id, self.pan_scale)
        # pan: view-gated only (nothing to avoid yet) so the frying pan itself always renders fully
        # inside the front (head_camera) video. Handle orientation preserved (rotate_rand stays off).
        skillet_pose, skillet_aabb = self._sample_clear_pose(
            xlim=sorted([side * 0.02, side * 0.13]), ylim=[-0.17, -0.03], qpos=[0, 0, 0.707, 0.707],
            collision_path=skillet_path, scale=skillet_scale, avoid_aabbs=[], padding=0.02, view_z=bz,
        )
        if skillet_pose is None:
            raise UnStableError(f"cook_meat: pan not placeable in head-camera view (seed {self._ep_seed}) -- skip")
        self.skillet = create_actor(
            self, pose=skillet_pose, modelname="106_skillet",
            model_id=self.skillet_id, convex=True, is_static=True, scale_mult=self.pan_scale,
        )

        # serving plate: rejection-sampled clear of the pan's real (mesh-derived) footprint.
        # Shrunk via scale_mult (003_plate is a shared/stock asset -- do NOT edit its model_data).
        self.plate_scale_mult = float(config.get("plate_scale_mult", 0.55))
        plate_path = "assets/objects/003_plate/collision/base0.glb"
        plate_qpos = [0.5, 0.5, 0.5, 0.5]
        plate_scale = self._applied_scale("003_plate", 0, self.plate_scale_mult)
        plate_pose, plate_aabb = self._sample_clear_pose(
            xlim=sorted([side * 0.05, side * 0.32]), ylim=[-0.18, 0.15], qpos=plate_qpos,
            collision_path=plate_path, scale=plate_scale, avoid_aabbs=[skillet_aabb], padding=0.02,
            view_z=bz,
        )
        if plate_pose is None:
            raise UnStableError(f"cook_meat: plate not placeable clear of pan & in view (seed {self._ep_seed}) -- skip")
        self.plate = create_actor(
            self, pose=plate_pose, modelname="003_plate",
            model_id=0, convex=False, is_static=True, scale_mult=self.plate_scale_mult,
        )

        # cutting board: rejection-sampled clear of BOTH the pan and the plate, in the FRONT-ish
        # band (positive y) where the steak can be reliably lifted off it (pick_ripe_apple.py
        # pitfall note). The steak spawns ON TOP of the board (sampled once, board and steak share
        # the same xy), like the apple-on-board pattern in pick_ripe_apple.py.
        # 104_board's model_data has no "scale" key, so create_actor drops the whole config
        # (Actor.config = None) -> add_prohibit_area / point getters would crash on None. Give the
        # actor a minimal in-memory config (the on-disk asset is untouched), same fix toast_bread.py
        # applies to 067_steamer.
        self.board_scale_mult = float(config.get("board_scale_mult", 0.07))
        board_scale_mult = self.board_scale_mult
        board_applied_scale = self._applied_scale(
            "104_board", 0, board_scale_mult
        )
        board_path = "assets/objects/104_board/collision/base0.glb"
        board_qpos = [0.707, 0.707, 0, 0]
        board_pose, board_aabb = self._sample_clear_pose(
            xlim=sorted([side * 0.05, side * 0.32]), ylim=[0.0, 0.18], qpos=board_qpos,
            collision_path=board_path,
            scale=board_applied_scale,
            avoid_aabbs=[skillet_aabb, plate_aabb],
            padding=0.02, view_z=bz,
        )
        if board_pose is None:
            raise UnStableError(f"cook_meat: board not placeable clear of pan/plate & in view (seed {self._ep_seed}) -- skip")
        board_xy = board_pose.p[:2]
        # true board thickness/rest-height from the ACTUAL mesh (probed at the origin), not
        # model_data's extents -- more accurate, and accounts for the mesh's own bottom face not
        # necessarily sitting exactly at its local-origin z (its model_data "center" is off-center).
        probe_pose = sapien.Pose([0.0, 0.0, 0.0], board_qpos)
        board_z_min, board_z_max = self._mesh_world_z_extent(
            board_path, probe_pose, board_applied_scale
        )
        self.board_th = board_z_max - board_z_min
        board_spawn_z = bz - board_z_min   # the mesh's true bottom face touches the table surface
        board_pose = sapien.Pose(
            [float(board_xy[0]), float(board_xy[1]), board_spawn_z], board_qpos,
        )
        with open(
            "assets/objects/104_board/model_data0.json", encoding="utf-8"
        ) as board_data_file:
            board_data = json.load(board_data_file)
        self.board = create_actor(
            self, pose=board_pose, modelname="104_board",
            model_id=0, convex=True, is_static=True, scale_mult=board_scale_mult,
        )
        self.board.config = {
            "scale": [board_scale_mult, board_scale_mult, board_scale_mult],
            "extents": board_data["extents"],
            "center": board_data["center"],
        }
        board_top = bz + self.board_th

        # steak: laid flat (qpos lays the thickness axis vertical, like 075_bread) on the board,
        # at the board's xy -- lifted by the board's thickness relative to the original
        # directly-on-table resting height (which was already tuned/validated at 0.74+table_z_bias).
        steak_pose = rand_pose(
            xlim=[float(board_xy[0]), float(board_xy[0])], ylim=[float(board_xy[1]), float(board_xy[1])],
            zlim=[board_top], qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True, rotate_lim=[0, np.pi / 6, 0],
        )
        # thicken the steak along its thin axis (model-y, which qpos maps to world-z) so it stands
        # taller in the pan -> the gripper gets a clean bite to lift it back OUT of the bowl
        self.steak_thick = float(config.get("steak_thick", 1.6))
        self.steak = create_actor(
            self, pose=steak_pose, modelname="200_steak",
            model_id=0, convex=True, is_static=False, scale_mult=(1.0, self.steak_thick, 1.0),
        )
        self.steak.set_mass(0.05)

        self.add_prohibit_area(self.skillet, padding=0.05)
        self.add_prohibit_area(self.plate, padding=0.04)
        self.add_prohibit_area(self.board, padding=0.03)
        self.add_prohibit_area(self.steak, padding=0.03)

        # cache the steak's render shapes so the cooking timer can recolor it
        self._steak_shapes = []
        for c in self.steak.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._steak_shapes = list(c.render_shapes)
        self._set_meat_color(0.0)

    # -------------------------------------------------------- cooking state
    def _set_meat_color(self, doneness: float) -> None:
        """Update every steak render shape for the supplied doneness."""
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
        for render_shape in self._steak_shapes:
            try:
                render_shape.material.set_base_color(color)
            except Exception:
                # Rendering color is cosmetic and must not invalidate an episode.
                pass

    def _update_kinematic_tasks(self) -> None:
        """Advance base dynamics and contact-gated cooking state by one step."""
        super()._update_kinematic_tasks()
        if getattr(self, "_cooking_active", False):
            try:
                on_pan = self.check_actors_contact("200_steak", "106_skillet")
            except Exception:
                # Treat transient simulator contact-query failures as no contact.
                on_pan = False
            if on_pan:
                self.doneness = min(1.0, self.doneness + 1.0 / max(1, self.cook_steps))
                self.max_doneness = max(self.max_doneness, self.doneness)
                self._set_meat_color(self.doneness)

    def _cook_idle(self) -> None:
        """Step the scene until the target doneness or dwell limit is reached."""
        max_steps = int(round(self.target_doneness * self.cook_steps)) + 30
        for i in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self.doneness >= self.target_doneness:
                break

    # ------------------------------------------------------------- policy
    def _dbg(self, tag: str) -> None:
        """Print opt-in planner diagnostics for task tuning."""
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_meat] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self) -> dict[str, Any]:
        """Run the dynamic or static expert trajectory for one episode."""
        if self.use_dynamic:
            return self._play_once_dynamic()
        return self._play_once_static()

    def _pan_cook_table(self, arm_tag: ArmTag) -> None:
        """Cook the held steak, then transfer it from the pan to the plate."""
        # place_actor aligns the steak into the pan bowl reliably (a bare drop misses the small bowl).
        pan_target = list(self.skillet.get_functional_point(0))
        pan_target[0] += self.place_dx   # offset so the steak lands centered in the bowl
        pan_target[1] += self.place_dy
        self.move(
            self.place_actor(
                self.steak, target_pose=pan_target, arm_tag=arm_tag,
                constrain="free", pre_dis=0.08, dis=0.03, is_open=True,
            ))
        self._dbg("place_in_pan")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._dbg("lift_clear_of_pan")
        self.move(self.back_to_origin(arm_tag))
        self._dbg("back_to_origin")

        self._cooking_active = True
        self._cook_idle()
        self._grasp_doneness = self.doneness
        self._cooking_active = False  # stop the timer before the grasp-off motion

        # grasp the cooked steak off the pan
        self.move(self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._dbg("grasp_off_pan")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self._dbg("lift_off_pan")

        # set the cooked steak down on the serving plate (not back on the table). The plate is
        # flat (not concave), so a relative set-down -- shift the held steak horizontally over the
        # plate center, lower, open -- plans far more reliably than an absolute place onto the
        # plate's functional frame (whose authored orientation yields an unreachable gripper pose;
        # this is the same cook_meat lesson toast_bread's plate hand-off already applies). Split
        # into single-axis displacements: a diagonal one-shot move is much harder to plan.
        steak_xy = np.array(self.steak.get_pose().p[:2])
        plate_xy = np.array(self.plate.get_functional_point(0)[:2])
        dx, dy = float(plate_xy[0] - steak_xy[0]), float(plate_xy[1] - steak_xy[1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx))
        self._dbg("over_plate_x")
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=dy))
        self._dbg("over_plate_y")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.06))
        self._dbg("lower_to_plate")
        self.move(self.open_gripper(arm_tag))
        self._dbg("release_on_plate")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))
        self._dbg("retreat")

    def _task_info(self, arm_tag: ArmTag) -> dict[str, str]:
        """Build the language-template substitutions recorded for this episode."""
        return {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{C}": "104_board/base0",
            "{D}": "003_plate/base0",
            "{a}": str(arm_tag),
        }

    def _play_once_static(self) -> dict[str, Any]:
        """Execute the expert trajectory for a stationary steak."""
        arm_tag = ArmTag("right" if self.steak.get_pose().p[0] > 0 else "left")
        # grasp the raw steak off the table
        self.move(self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table(arm_tag)
        self.info["info"] = self._task_info(arm_tag)
        return self.info

    def get_dynamic_motion_config(self) -> dict[str, Any] | None:
        """Return the base workflow's dynamic-motion configuration when enabled."""
        if not self.use_dynamic:
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
        arm_tag = ArmTag("right" if self.steak.get_pose().p[0] > 0 else "left")
        p = self.steak.get_pose().p
        self.end_position = np.array([p[0], p[1], p[2]])

        def robot_action_sequence(_need_plan_mode: bool) -> None:
            grasp_result = self.grasp_actor(
                self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1
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
            target_actor=self.steak,
            end_position=self.end_position,
            robot_action_sequence=robot_action_sequence,
            table_bounds=table_bounds,
        )
        if not success:
            print("Dynamic trajectory failed, fallback to static")
            return self._play_once_static()

        for c in self.steak.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_velocity(np.zeros(3))
                    c.set_angular_velocity(np.zeros(3))
                    c.set_linear_damping(15.0)
                    c.set_angular_damping(40.0)
                except Exception:
                    # Damping is best-effort; the original dynamic workflow remains valid.
                    pass

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table(arm_tag)
        self.info["info"] = self._task_info(arm_tag)
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self) -> bool:
        """Return whether a sufficiently cooked steak is positioned near the plate."""
        if self._grasp_doneness is None:
            return False
        cooked_ok = self.max_doneness >= self.target_doneness - 0.05
        steak_p = self.steak.get_pose().p
        steak_z = float(steak_p[2])
        steak_xy = np.array(steak_p[:2])
        plate_xy = np.array(self.plate.get_functional_point(0)[:2])
        pan_xy = np.array(self.skillet.get_functional_point(0)[:2])
        d_plate = float(np.linalg.norm(steak_xy - plate_xy))
        d_pan = float(np.linalg.norm(steak_xy - pan_xy))
        # Approximate plate placement by requiring a plate-sized XY radius, greater proximity to
        # the plate than the pan, and a position above the table surface.
        on_plate = d_plate < 0.12
        off_pan = d_plate < d_pan
        above_table = steak_z > (0.73 + self.table_z_bias)
        return bool(cooked_ok and on_plate and off_pan and above_table)

    def get_obs(self) -> dict[str, Any]:
        """Extend the base observation with per-frame cooking measurements."""
        obs = super().get_obs()
        obs["cooking"] = {
            "doneness": float(getattr(self, "doneness", 0.0)),
            "target_doneness": float(getattr(self, "target_doneness", self.TARGET_DONENESS_DEFAULT)),
            "cook_steps": float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)),
        }
        # record steak-vs-pan and steak-vs-plate offsets to measure/center the placements
        try:
            sxy = np.array(self.steak.get_pose().p)[:2]
            pxy = np.array(self.skillet.get_functional_point(0))[:2]
            platexy = np.array(self.plate.get_functional_point(0))[:2]
            obs["cooking"]["steak_xy"] = [float(sxy[0]), float(sxy[1])]
            obs["cooking"]["pan_xy"] = [float(pxy[0]), float(pxy[1])]
            obs["cooking"]["place_offset"] = [float(sxy[0] - pxy[0]), float(sxy[1] - pxy[1])]
            obs["cooking"]["plate_xy"] = [float(platexy[0]), float(platexy[1])]
        except Exception:
            # Actors may be unavailable during early setup-time observations.
            pass
        return obs
