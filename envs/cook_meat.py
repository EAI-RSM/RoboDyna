from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
import json
import trimesh


class cook_meat(Base_Task):
    """Pick up a raw steak from a cutting board, cook it on a pan (a per-step timer browns it
    gradually), then once it reaches a target doneness, grasp it off and set it on a serving
    plate. Introduces a time-evolving, rendered object state on top of the standard pick-place
    primitives; mirrors toast_bread's board-source / plate-destination layout."""

    COOK_STEPS_DEFAULT = 1000         # on-pan sim steps for doneness 0 -> 1 (longer = slower, clearer change)
    TARGET_DONENESS_DEFAULT = 0.5     # medium: cook to the brown midpoint, then grasp off
    # cooking gradient: raw red -> MEDIUM warm red-brown (at the 0.5 target) -> well-done dark brown.
    # 0.5 is deliberately a clear MEDIUM (still reddish), not full brown, so picking at the 0.5
    # target reads visually as "medium", not "fully cooked".
    COLOR_STOPS = [
        (0.0, [1.00, 0.12, 0.09]),    # raw: vivid saturated red
        (0.5, [0.66, 0.30, 0.14]),    # medium: warm red-brown (clearly transitional)
        (1.0, [0.16, 0.08, 0.04]),    # well done: dark brown
    ]

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's `task_args` block
        self._cook_cfg = kwags.get("task_args", {}).get("cook_meat", {})
        self._ep_seed = int(kwags.get("seed", 0))   # used to balance left/right hand per episode
        super()._init_task_env_(**kwags)

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
    _MESH_CACHE = {}

    @staticmethod
    def _applied_scale(modelname, model_id, scale_mult):
        # The scale create_actor() actually bakes into the mesh is model_data["scale"] * scale_mult
        # (create_actor.py:542,552,564). model_data's raw collision-glb vertices are in UN-scaled
        # mesh units, so the footprint helpers below MUST be fed this same product -- feeding only
        # scale_mult inflates the footprint by 1/authored_scale (e.g. 40x for the 0.025-scale plate).
        # Assets with no "scale" key (e.g. 104_board) get authored scale 1.0, matching create_actor's
        # (1,1,1) default for that case.
        try:
            d = json.load(open(f"assets/objects/{modelname}/model_data{model_id}.json"))
            authored = float(d["scale"][0])
        except (KeyError, TypeError, FileNotFoundError, ValueError):
            authored = 1.0
        return authored * float(scale_mult)

    @classmethod
    def _mesh_vertices(cls, collision_path):
        if collision_path not in cls._MESH_CACHE:
            mesh = trimesh.load(collision_path, force="mesh")
            cls._MESH_CACHE[collision_path] = np.asarray(mesh.vertices, dtype=float)
        return cls._MESH_CACHE[collision_path]

    @classmethod
    def _mesh_aabb_xy(cls, collision_path, pose, scale, padding=0.0):
        # world-space XY footprint of the actual (scaled, rotated, translated) collision mesh.
        v = cls._mesh_vertices(collision_path) * scale
        M = pose.to_transformation_matrix()
        world = (M[:3, :3] @ v.T).T + np.asarray(pose.p)
        return (
            float(world[:, 0].min()) - padding, float(world[:, 1].min()) - padding,
            float(world[:, 0].max()) + padding, float(world[:, 1].max()) + padding,
        )

    @classmethod
    def _mesh_world_z_extent(cls, collision_path, pose, scale):
        # (min, max) world-Z of the transformed mesh -- used for the board's true thickness
        # (instead of trusting model_data's OMBB-derived "extents", see note above).
        v = cls._mesh_vertices(collision_path) * scale
        M = pose.to_transformation_matrix()
        world = (M[:3, :3] @ v.T).T + np.asarray(pose.p)
        return float(world[:, 2].min()), float(world[:, 2].max())

    @classmethod
    def _footprint_offsets(cls, collision_path, qpos, scale):
        # Fixed world-XY footprint box (vmin_x, vmin_y, vmax_x, vmax_y) of the mesh at this
        # rotation+scale BEFORE translation. Translating a rigid footprint merely shifts its AABB, so
        # AABB(x, y) = (vmin_x + x, vmin_y + y, vmax_x + x, vmax_y + y). Computing this once lets the
        # grid fallback below offset it per cell instead of re-transforming the mesh every candidate.
        v = cls._mesh_vertices(collision_path) * scale
        R = sapien.Pose([0.0, 0.0, 0.0], qpos).to_transformation_matrix()[:3, :3]
        w = (R @ v.T).T
        return float(w[:, 0].min()), float(w[:, 1].min()), float(w[:, 0].max()), float(w[:, 1].max())

    @staticmethod
    def _aabb_overlap(a, b):
        return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]

    @staticmethod
    def _aabb_gap(a, b):
        # >0: separated by this much clearance (whichever axis actually separates them);
        # <=0: overlapping (closer to 0 = shallower penetration). Used to score candidates.
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
    _CAM_POS = np.array([-0.032, -0.45, 1.35])
    _CAM_FWD = np.array([0.0, 0.6, -0.8])     # already unit-norm
    _CAM_LEFT = np.array([-1.0, 0.0, 0.0])
    _CAM_FOVY_DEG = 37.0
    _CAM_W, _CAM_H = 320, 240

    @classmethod
    def _project_to_head_cam(cls, pts_world):
        # World (N,3) -> (u, v, depth). Mirrors camera.py:101-120: camera-local axes are the columns
        # [forward, left, up] with up = forward x left, so p_cam = R^T (p - cam_pos) gives
        # x=depth-along-view, y=left, z=up; pinhole with square pixels from fovy. Validated to <1px
        # against the known visible band (near edge y~-0.247 -> v~H, far edge y~0.433 -> v~0).
        f = cls._CAM_FWD / np.linalg.norm(cls._CAM_FWD)
        l = cls._CAM_LEFT / np.linalg.norm(cls._CAM_LEFT)
        up = np.cross(f, l)
        d = np.asarray(pts_world, dtype=float) - cls._CAM_POS
        depth = d @ f
        y_cam = d @ l
        z_cam = d @ up
        fpix = (cls._CAM_H / 2.0) / np.tan(np.deg2rad(cls._CAM_FOVY_DEG) / 2.0)
        cx, cy = cls._CAM_W / 2.0, cls._CAM_H / 2.0
        with np.errstate(divide="ignore", invalid="ignore"):
            u = cx - fpix * (y_cam / depth)   # +y_cam (left) -> smaller u (image left)
            v = cy - fpix * (z_cam / depth)   # +z_cam (up)   -> smaller v (image top)
        return u, v, depth

    @classmethod
    def _footprint_in_head_view(cls, aabb_xy, z, margin_px=4):
        # True iff all four footprint corners (at table-top height z) project strictly inside the
        # image with a small pixel margin -- i.e. the object renders fully within the front video.
        xmin, ymin, xmax, ymax = aabb_xy
        corners = np.array([[xmin, ymin, z], [xmin, ymax, z], [xmax, ymin, z], [xmax, ymax, z]])
        u, v, depth = cls._project_to_head_cam(corners)
        if np.any(depth <= 0):
            return False
        return bool(np.all(u >= margin_px) and np.all(u <= cls._CAM_W - margin_px)
                    and np.all(v >= margin_px) and np.all(v <= cls._CAM_H - margin_px))

    def _sample_clear_pose(self, xlim, ylim, qpos, collision_path, scale, avoid_aabbs, padding,
                           view_z, tries=400, grid_step=0.005):
        # Find a placement whose real mesh-derived world AABB BOTH (a) clears every box in
        # `avoid_aabbs` and (b) lies fully inside the head-camera frame. Returns (pose, aabb) on
        # success, or (None, None) if NO such placement exists in the band -- in which case the
        # caller raises UnStableError to SKIP the whole seed. This is the "perfect sampling" policy:
        # never contort a placement (no pushing off-screen, no band expansion) -- if a seed can't be
        # sampled cleanly, drop it and let the collector try the next one (good seeds are common).
        def ok(aabb):
            clr = min((self._aabb_gap(aabb, a) for a in avoid_aabbs), default=1e9)
            return clr > 0 and self._footprint_in_head_view(aabb, view_z)

        # Phase 1: random rejection sampling -- gives per-seed spatial diversity when a spot exists.
        for _ in range(tries):
            cand = rand_pose(xlim=xlim, ylim=ylim, qpos=qpos)
            aabb = self._mesh_aabb_xy(collision_path, cand, scale, padding=padding)
            if ok(aabb):
                return cand, aabb

        # Phase 2: exhaustive grid -- random can miss a small feasible pocket; a grid cannot, so this
        # avoids skipping a seed that IS sampleable. Footprint shape is translation-invariant, so
        # offset a once-computed box per cell (cheap). Collect all clear+in-view cells, pick one at
        # random (keeps diversity). Only if the grid ALSO finds nothing do we declare the seed bad.
        offs = self._footprint_offsets(collision_path, qpos, scale)
        def aabb_at(x, y):
            return (offs[0] + x - padding, offs[1] + y - padding, offs[2] + x + padding, offs[3] + y + padding)
        xlo, xhi = min(xlim), max(xlim)
        ylo, yhi = min(ylim), max(ylim)
        good = []
        nx = max(2, int((xhi - xlo) / grid_step) + 1)
        ny = max(2, int((yhi - ylo) / grid_step) + 1)
        for gx in np.linspace(xlo, xhi, nx):
            for gy in np.linspace(ylo, yhi, ny):
                ab = aabb_at(gx, gy)
                if ok(ab):
                    good.append((gx, gy, ab))
        if good:
            gx, gy, ab = good[np.random.randint(len(good))]
            return sapien.Pose([float(gx), float(gy), 0.741], qpos), ab
        return None, None   # seed is unsampleable for this object -> caller skips it

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cook_cfg
        # KEY per-episode randomization: the COOK SPEED = how many sim steps the steak takes to go
        # from raw(0) to fully done(1.0). Smaller = browns faster. Also a tight doneness band ~medium.
        self.cook_steps = int(np.random.uniform(c.get("cook_steps_min", 600), c.get("cook_steps_max", 1600)))
        self.target_doneness = float(np.random.uniform(c.get("target_doneness_min", 0.45),
                                                       c.get("target_doneness_max", 0.55)))
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
        self.pan_scale = float(self._cook_cfg.get("pan_scale", 1.0))   # original size (config-tunable)
        # placement offset (world x,y) added to the pan functional point so the steak lands centered
        # in the bowl (the grasp offset otherwise lands it off-center) -- tune from the measured offset
        self.place_dx = float(self._cook_cfg.get("place_dx", 0.0))
        self.place_dy = float(self._cook_cfg.get("place_dy", 0.0))
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
        self.plate_scale_mult = float(self._cook_cfg.get("plate_scale_mult", 0.55))
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
        self.board_scale_mult = float(self._cook_cfg.get("board_scale_mult", 0.07))
        bm = self.board_scale_mult                                        # per-spawn mult passed to create_actor
        board_applied = self._applied_scale("104_board", 0, bm)           # mesh-baked scale (== bm; board has no authored scale)
        board_path = "assets/objects/104_board/collision/base0.glb"
        board_qpos = [0.707, 0.707, 0, 0]
        board_pose, board_aabb = self._sample_clear_pose(
            xlim=sorted([side * 0.05, side * 0.32]), ylim=[0.0, 0.18], qpos=board_qpos,
            collision_path=board_path, scale=board_applied, avoid_aabbs=[skillet_aabb, plate_aabb],
            padding=0.02, view_z=bz,
        )
        if board_pose is None:
            raise UnStableError(f"cook_meat: board not placeable clear of pan/plate & in view (seed {self._ep_seed}) -- skip")
        board_xy = board_pose.p[:2]
        # true board thickness/rest-height from the ACTUAL mesh (probed at the origin), not
        # model_data's extents -- more accurate, and accounts for the mesh's own bottom face not
        # necessarily sitting exactly at its local-origin z (its model_data "center" is off-center).
        _probe_pose = sapien.Pose([0.0, 0.0, 0.0], board_qpos)
        board_z_min, board_z_max = self._mesh_world_z_extent(board_path, _probe_pose, board_applied)
        self.board_th = board_z_max - board_z_min
        board_spawn_z = bz - board_z_min   # the mesh's true bottom face touches the table surface
        board_pose = sapien.Pose(
            [float(board_xy[0]), float(board_xy[1]), board_spawn_z], board_qpos,
        )
        _bd = json.load(open("assets/objects/104_board/model_data0.json"))
        self.board = create_actor(
            self, pose=board_pose, modelname="104_board",
            model_id=0, convex=True, is_static=True, scale_mult=bm,
        )
        self.board.config = {
            "scale": [bm, bm, bm],
            "extents": _bd["extents"],
            "center": _bd["center"],
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
        self.steak_thick = float(self._cook_cfg.get("steak_thick", 1.6))
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
    def _set_meat_color(self, doneness):
        d = float(np.clip(doneness, 0.0, 1.0))
        # piecewise-linear interpolation across the multi-stop gradient
        stops = self.COLOR_STOPS
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        col = rgb + [1.0]
        for s in self._steak_shapes:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step
        super()._update_kinematic_tasks()
        if getattr(self, "_cooking_active", False):
            try:
                on_pan = self.check_actors_contact("200_steak", "106_skillet")
            except Exception:
                on_pan = False
            if on_pan:
                self.doneness = min(1.0, self.doneness + 1.0 / max(1, self.cook_steps))
                self.max_doneness = max(self.max_doneness, self.doneness)
                self._set_meat_color(self.doneness)

    def _cook_idle(self):
        # dwell on the pan, recording frames, until the target doneness is reached
        max_steps = int(round(self.target_doneness * self.cook_steps)) + 30
        for i in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self.doneness >= self.target_doneness:
                break

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        import os
        if os.environ.get("COOK_DEBUG"):
            print(f"[cook_meat] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self):
        if self.use_dynamic:
            return self._play_once_dynamic()
        return self._play_once_static()

    def _pan_cook_table(self, arm_tag):
        # with the steak held: seat it in the pan, cook it, lift it off, then set it back on the table.
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

    def _play_once_static(self):
        arm_tag = ArmTag("right" if self.steak.get_pose().p[0] > 0 else "left")
        # grasp the raw steak off the table
        self.move(self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table(arm_tag)
        self.info["info"] = {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{C}": "104_board/base0",
            "{D}": "003_plate/base0",
            "{a}": str(arm_tag),
        }
        return self.info

    def get_dynamic_motion_config(self) -> dict:
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

    def _play_once_dynamic(self):
        arm_tag = ArmTag("right" if self.steak.get_pose().p[0] > 0 else "left")
        p = self.steak.get_pose().p
        self.end_position = np.array([p[0], p[1], p[2]])

        def robot_action_sequence(need_plan_mode):
            gr = self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1)
            if not gr or gr[1] is None or len(gr[1]) == 0:
                return
            self.move(gr)

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
                    pass

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table(arm_tag)
        self.info["info"] = {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{C}": "104_board/base0",
            "{D}": "003_plate/base0",
            "{a}": str(arm_tag),
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
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
        # success now needs the cooked steak resting ON THE SERVING PLATE (the task's intent), not
        # merely back on the table -- judged by being clearly closer to the plate than to the pan,
        # within a plate-sized radius (mirrors toast_bread's on_plate/off_steamer check).
        on_plate = d_plate < 0.12
        off_pan = d_plate < d_pan
        above_table = steak_z > (0.73 + self.table_z_bias)
        return bool(cooked_ok and on_plate and off_pan and above_table)

    # record the cooking state into the trajectory (per-frame)
    def get_obs(self):
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
            pass
        return obs
