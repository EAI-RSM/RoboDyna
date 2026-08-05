from ._office_base_task import Office_base_task
from .utils import *
from ._GLOBAL_CONFIGS import *
import json
import sapien
import sapien.physx
import numpy as np
from pathlib import Path
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult, qinverse, quat2mat


class catch_rolling_cup(Office_base_task):
    """Stop a rolling cup and stand it upright on the table.

    Scene: full-width single wall shelf (no drawer / file / plates) with
    non-overlapping décor. Cup starts upright on the shelf, leans, falls, and
    rolls; the robot blocks it, grasps around its exterior (no weld attach),
    turns it upright, and sets it on the table.
    """

    CUP_IDS = [0, 1, 2, 5, 6, 3]
    MUG_IDS = list(range(13))
    KETTLE_IDS = list(range(6))
    BASKET_IDS = list(range(5))
    TISSUE_IDS = list(range(7))
    ALARM_IDS = list(range(6))
    FRUIT_IDS = [0, 1, 2, 3, 5]
    APPLE_IDS = [0, 1]  # 035_apple (025_apple not in assets)

    ROLL_SPEED_DEFAULT = 0.14
    FALL_SPEED_XY_DEFAULT = 0.07
    TIP_DURATION_DEFAULT = 0.90
    GRAVITY = 9.81

    TABLE_EDGE_Y_DEFAULT = -0.30
    INTERCEPT_FRAC_DEFAULT = 0.65
    CATCH_SHELF_CLEARANCE_DEFAULT = 0.22
    STOP_XY_TOL_DEFAULT = 0.055
    STOP_LEAD_Y_DEFAULT = 0.05
    BLOCK_Z_ABOVE_DEFAULT = 0.045
    POST_GRASP_LIFT_DEFAULT = 0.10
    UPRIGHT_HOLD_DEFAULT = 0.30
    CUP_X_ABS_MIN = 0.18
    CUP_X_ABS_MAX = 0.32
    GRASP_TCP_ERR_MAX = 0.035

    # Shelf spans the table length (create_table length=1.2), 30 cm above table.
    SHELF_WIDTH = 1.20
    SHELF_DEPTH = 0.18
    SHELF_THICK = 0.02
    SHELF_Y = 0.26
    SHELF_Z_ABOVE_TABLE = 0.30

    CUP_ROBOT_IGNORE_BIT = 1 << 21
    CUP_ROBOT_IGNORE_ID = 0x0C51

    CUP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    PROP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    GRASP_SPAN_MAX = 0.100
    JAW_GAP_TABLE = ((0.006, 0.0), (0.0182, 0.25), (0.0532, 0.5), (0.0882, 0.75), (0.110, 1.0))

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("catch_rolling_cup", {})
        self._loaded = False
        self.cup = None
        self._cup_rigid = None
        self._traj = []
        self._traj_step = 0
        # parked | rolling | stopped | grasped | placed | fallen
        self._cup_state = "parked"
        self._fell_off = False
        self._stopped = False
        self._grasped = False
        self._placed = False
        self._holding = False  # fingers closed around cup; cup stays dynamic
        self._table_start_idx = 0
        self._roll_start_idx = 0
        self._intercept_idx = 0
        self._intercept_pos = np.zeros(3)
        self._stop_pos = np.zeros(3)
        self._stop_quat = None
        self._table_place = np.zeros(3)
        self.arm_side = "right"
        self.decor = []
        self.fruit_basket = None
        self.basket_fruits = []
        self._occ_shelf = []  # shelf-surface XY footprints
        self._occ_table = []  # table-surface XY footprints
        self._robot_groups_backup = None
        super().setup_demo(**kwags)
        self._configure_observer_camera()

    # --------------------------------------------------------------- scene
    def create_table_and_wall(self, table_xy_bias=[0, 0], table_height=0.74):
        """Office table + wall with a full-width single shelf (no drawer / files)."""
        self.arr_v = 1
        self.table_xy_bias = list(table_xy_bias)
        table_height = float(self.office_info["table_height"])
        self.table_z_bias = 0.0

        if self.random_background:
            texture_type = "seen" if not self.eval_mode else "unseen"
            directory = Path("assets/background_texture") / texture_type
            count = len([p for p in directory.iterdir() if p.is_file()])
            wall_id, table_id, floor_id = np.random.randint(0, count, size=3)
            self.wall_texture = f"{texture_type}/{wall_id}"
            self.table_texture = f"{texture_type}/{table_id}"
            self.floor_texture = f"{texture_type}/{floor_id}"
            if np.random.rand() <= self.clean_background_rate:
                self.wall_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.table_texture = None
            if np.random.rand() <= self.clean_background_rate:
                self.floor_texture = None
        else:
            self.wall_texture = self.table_texture = self.floor_texture = None

        self.floor_parts = []
        for i, pos in enumerate(([1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0])):
            floor = create_box(
                self.scene,
                sapien.Pose(p=pos),
                half_size=[1, 1, 0.005],
                color=(0.85, 0.85, 0.85),
                name=f"floor_{i}",
                texture_id=self.floor_texture,
                is_static=True,
            )
            self.floor_parts.append(floor)

        self.wall = create_box(
            self.scene,
            sapien.Pose(p=[0, 1, 1.5]),
            half_size=[3, 0.6, 1.5],
            color=(1, 0.9, 0.9),
            name="wall",
            texture_id=self.wall_texture,
            is_static=True,
        )
        self.table = create_table(
            self.scene,
            sapien.Pose(p=[table_xy_bias[0], table_xy_bias[1], table_height]),
            length=1.2,
            width=0.7,
            height=table_height,
            thickness=0.05,
            is_static=True,
            texture_id=self.table_texture,
        )

        shelf_z = table_height + self.SHELF_Z_ABOVE_TABLE
        shelf_y = self.SHELF_Y
        half_x = 0.5 * self.SHELF_WIDTH
        half_y = 0.5 * self.SHELF_DEPTH
        half_z = 0.5 * self.SHELF_THICK
        self.shelf = create_box(
            self.scene,
            sapien.Pose(p=[0.0, shelf_y, shelf_z]),
            half_size=[half_x, half_y, half_z],
            color=(0.55, 0.42, 0.30),
            name="single_wall_shelf",
            is_static=True,
        )
        for bx in (-0.45, 0.0, 0.45):
            create_box(
                self.scene,
                sapien.Pose(p=[bx, shelf_y + half_y - 0.01, shelf_z - 0.06]),
                half_size=[0.012, 0.01, 0.06],
                color=(0.35, 0.35, 0.35),
                name="shelf_bracket",
                is_static=True,
            )

        self.office_info["furn_x_v"]["shelf"] = [0.0, 0.0, 0.0]
        self.office_info["shelf_area"] = [self.SHELF_WIDTH, self.SHELF_DEPTH]
        self.office_info["shelf_heights"] = [shelf_z + half_z]
        self.office_info["shelf_lims"] = [
            -half_x, shelf_y - half_y, half_x, shelf_y + half_y,
        ]
        self.shelf_lims = list(self.office_info["shelf_lims"])
        self.prohibited_area.append([
            self.shelf_lims[0] - 0.02,
            self.shelf_lims[1] - 0.02,
            self.shelf_lims[2] + 0.02,
            self.shelf_lims[3] + 0.02,
        ])
        self.cabinet = None
        self.file_holder = None
        self.wooden_box = None

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.50, 0.55, 1.55], dtype=np.float64)
        look_at = np.array([0.0, 0.05, 0.95], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    # ------------------------------------------------------------------ helpers
    def _get_rigid(self, entity):
        obj = entity.actor if hasattr(entity, "actor") else entity
        for c in obj.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                return c
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

    def _make_dynamic(self, entity, mass=None, lin_damp=0.4, ang_damp=0.6):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            rigid.set_linear_damping(float(lin_damp))
            rigid.set_angular_damping(float(ang_damp))
            if mass is not None:
                try:
                    entity.set_mass(float(mass))
                except Exception:
                    rigid.mass = float(mass)
            rigid.set_linear_velocity(np.zeros(3))
            rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        return rigid

    def _tune_actor_friction(self, entity, static_f=1.8, dynamic_f=1.5, restitution=0.02):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return
        try:
            for s in rigid.get_collision_shapes():
                m = s.get_physical_material()
                m.set_static_friction(float(static_f))
                m.set_dynamic_friction(float(dynamic_f))
                m.set_restitution(float(restitution))
        except Exception:
            pass

    def _settle_physics(self, steps=60):
        """Advance PhysX without recording (used while packing the basket)."""
        for _ in range(int(steps)):
            self.scene.step()

    def _set_entity_pose(self, entity, pose):
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                if rigid.kinematic:
                    rigid.set_kinematic_target(pose)
                    return
            except Exception:
                pass
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    def _cup_upright_quat(self):
        return self.CUP_UPRIGHT_Q.copy()

    def _cup_tip_quat(self, tip_frac, roll_angle=0.0):
        tip_frac = float(np.clip(tip_frac, 0.0, 1.0))
        q_tip = euler2quat(0.0, tip_frac * np.pi / 2.0, 0.0, axes="sxyz")
        q_side = qmult(q_tip, self.CUP_UPRIGHT_Q)
        if abs(roll_angle) < 1e-9:
            return np.asarray(q_side, dtype=np.float64)
        q_spin = euler2quat(float(roll_angle), 0.0, 0.0, axes="sxyz")
        return np.asarray(qmult(q_spin, q_side), dtype=np.float64)

    @staticmethod
    def _cup_dims(model_id: int, scale_mult: float = 1.0):
        path = Path(f"assets/objects/021_cup/model_data{int(model_id)}.json")
        data = json.loads(path.read_text())
        size = (np.asarray(data["extents"], dtype=np.float64)
                * np.asarray(data["scale"], dtype=np.float64)
                * float(scale_mult))
        return float(size[1]), 0.5 * (float(size[0]) + float(size[2]))

    @classmethod
    def _gripper_pos_for_gap(cls, gap: float) -> float:
        gaps = [g for g, _ in cls.JAW_GAP_TABLE]
        cmds = [p for _, p in cls.JAW_GAP_TABLE]
        return float(np.clip(np.interp(float(gap), gaps, cmds), 0.0, 1.0))

    def _close_cmd_for_cup(self):
        """Close jaws around the outer diameter (not through the hollow)."""
        gap = 2.0 * float(self.cup_radius) + 0.006
        return self._gripper_pos_for_gap(max(0.025, gap))

    def _footprint_ok(self, occ, cx, cy, hx, hy, pad=0.01):
        box = (cx - hx - pad, cy - hy - pad, cx + hx + pad, cy + hy + pad)
        for x0, y0, x1, y1 in occ:
            if not (box[2] <= x0 or box[0] >= x1 or box[3] <= y0 or box[1] >= y1):
                return False
        return True

    def _reserve(self, occ, cx, cy, hx, hy, pad=0.01):
        occ.append((
            cx - hx - pad, cy - hy - pad, cx + hx + pad, cy + hy + pad,
        ))

    def _spawn_static(
        self, modelname, model_id, p, hx, hy, scale=None, scale_mult=1.0,
        force=False, surface="table",
    ):
        """Static décor with reserved XY footprint on shelf or table (no overlaps)."""
        cx, cy = float(p[0]), float(p[1])
        occ = self._occ_shelf if surface == "shelf" else self._occ_table
        if not force and not self._footprint_ok(occ, cx, cy, hx, hy):
            return None
        q = self.PROP_UPRIGHT_Q.tolist()
        pose = sapien.Pose(np.asarray(p, dtype=np.float64).tolist(), q)
        if scale is None:
            try:
                data = json.loads(
                    Path(f"assets/objects/{modelname}/model_data{int(model_id)}.json").read_text()
                )
                base = data.get("scale") or [1.0, 1.0, 1.0]
                scale = [float(s) * float(scale_mult) for s in base]
            except Exception:
                scale = [float(scale_mult)] * 3
        # Visual-only: avoids planner hangs; footprints already prevent overlaps.
        actor = self._create_scaled_static_object(
            modelname, int(model_id), pose, list(scale), collision=False,
        )
        self._reserve(occ, cx, cy, hx, hy)
        self.decor.append(actor)
        return actor

    # --------------------------------------------------------------- trajectory
    def _build_trajectory(self):
        dt = float(self.scene.get_timestep())
        g = self.GRAVITY
        r = self.cup_radius
        v = self.roll_speed
        traj = []
        traveled = 0.0

        def append(x, y, z, tip_frac, roll_s=0.0):
            q = self._cup_tip_quat(tip_frac, -roll_s / max(r, 1e-4))
            traj.append((
                np.array([x, y, z], dtype=np.float64),
                np.asarray(q, dtype=np.float64),
            ))

        x0 = float(self.cup_start[0])
        y0 = float(self.cup_start[1])
        z_upright = float(self.cup_start[2])
        z_side_shelf = float(self.shelf_z_surf + r)
        y_shelf_edge = float(self.shelf_front_y)
        z_table = self.table_top + r

        tip_t = max(0.25, float(self.tip_duration))
        n_tip = max(1, int(round(tip_t / dt)))
        for i in range(1, n_tip + 1):
            frac = i / n_tip
            s = frac * frac * (3.0 - 2.0 * frac)
            y = y0 + s * (y_shelf_edge - y0)
            z = z_upright + s * (z_side_shelf - z_upright)
            append(x0, y, z, tip_frac=s, roll_s=0.0)

        vy = -abs(self.fall_speed_xy)
        dz = max(1e-4, z_side_shelf - z_table)
        t_fall = float(np.sqrt(2.0 * dz / g))
        n_fall = max(1, int(round(t_fall / dt)))
        y_land = y_shelf_edge + vy * t_fall
        for i in range(1, n_fall + 1):
            t = (i / n_fall) * t_fall
            y = y_shelf_edge + vy * t
            z = z_side_shelf - 0.5 * g * t * t
            traveled += abs(vy) * (t_fall / n_fall)
            append(x0, y, max(z, z_table), tip_frac=1.0, roll_s=traveled)
        append(x0, y_land, z_table, tip_frac=1.0, roll_s=traveled)
        self._land_y = float(y_land)
        self._land_idx = len(traj) - 1

        y_edge = float(self.table_edge_y)
        dist_table = max(1e-4, y_land - y_edge)
        n_table = max(1, int(round((dist_table / v) / dt)))
        table_start_idx = len(traj)
        self._roll_start_idx = int(table_start_idx)
        for i in range(1, n_table + 1):
            frac = i / n_table
            y = y_land + frac * (y_edge - y_land)
            traveled += dist_table / n_table
            append(x0, y, z_table, tip_frac=1.0, roll_s=traveled)

        intercept_i = table_start_idx + int(round(self.intercept_frac * (n_table - 1)))
        intercept_i = int(np.clip(intercept_i, table_start_idx, len(traj) - 1))
        self._table_start_idx = int(table_start_idx)
        self._intercept_idx = intercept_i
        self._intercept_pos = traj[intercept_i][0].copy()

        n_off = max(1, int(round(0.55 / dt)))
        for i in range(1, n_off + 1):
            t = i * dt
            y = y_edge + vy * t
            z = z_table - 0.5 * g * t * t
            traveled += abs(vy) * dt
            append(x0, y, z, tip_frac=1.0, roll_s=traveled)

        self._traj = traj
        self._fall_off_idx = table_start_idx + n_table

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.roll_speed = float(c.get("roll_speed", self.ROLL_SPEED_DEFAULT))
        self.fall_speed_xy = float(c.get("fall_speed_xy", self.FALL_SPEED_XY_DEFAULT))
        self.tip_duration = float(c.get("tip_duration", self.TIP_DURATION_DEFAULT))
        self.table_edge_y = float(c.get("table_edge_y", self.TABLE_EDGE_Y_DEFAULT))
        self.intercept_frac = float(c.get("intercept_frac", self.INTERCEPT_FRAC_DEFAULT))
        self.catch_shelf_clearance = float(c.get(
            "catch_shelf_clearance", self.CATCH_SHELF_CLEARANCE_DEFAULT,
        ))
        self.stop_xy_tol = float(c.get("stop_xy_tol", self.STOP_XY_TOL_DEFAULT))
        self.stop_lead_y = float(c.get("stop_lead_y", self.STOP_LEAD_Y_DEFAULT))
        self.block_z_above = float(c.get("block_z_above", self.BLOCK_Z_ABOVE_DEFAULT))
        self.post_grasp_lift = float(c.get("post_grasp_lift", self.POST_GRASP_LIFT_DEFAULT))
        self.upright_hold = float(c.get("upright_hold", self.UPRIGHT_HOLD_DEFAULT))
        cup_ids = c.get("cup_ids", self.CUP_IDS)
        try:
            cup_ids = [int(x) for x in cup_ids]
        except Exception:
            cup_ids = list(self.CUP_IDS)
        if "cup_id" in c:
            self.cup_id = int(c["cup_id"])
        else:
            self.cup_id = int(np.random.choice(cup_ids or self.CUP_IDS))

        axial, diameter = self._cup_dims(self.cup_id)
        self.cup_scale_mult = float(c.get(
            "cup_scale_mult",
            min(1.0, self.GRASP_SPAN_MAX / max(axial, 1e-6)),
        ))
        self.cup_axial_span, diameter = self._cup_dims(self.cup_id, self.cup_scale_mult)
        self.cup_height = float(self.cup_axial_span)
        self.cup_radius = float(c.get("cup_radius", max(0.025, 0.5 * diameter)))

        self.table_top = 0.74 + float(self.table_z_bias)
        self._measure_shelf_plate()
        self.shelf_front_y = float(self.shelf_plate_ylim[0])
        self.shelf_z_surf = float(self.shelf_plate_z)

        # Cup on the right half of the shelf (right-arm reach); décor fills the rest.
        sign = 1.0
        self.arm_side = "right"
        x0, x1 = self.shelf_plate_xlim
        y0, y1 = self.shelf_plate_ylim
        cup_x = float(np.clip(
            sign * np.random.uniform(self.CUP_X_ABS_MIN, self.CUP_X_ABS_MAX),
            x0 + 0.06,
            x1 - 0.06,
        ))
        cup_y = float(np.random.uniform(y0 + 0.03, y1 - 0.03))
        cup_z = self.shelf_z_surf + 0.5 * self.cup_height
        self.cup_start = np.array([cup_x, cup_y, cup_z], dtype=np.float64)
        self._table_place = np.array(
            [
                cup_x,
                float(np.clip(self.shelf_front_y - 0.16, self.table_edge_y + 0.10, 0.02)),
                self.table_top + 0.5 * self.cup_height,
            ],
            dtype=np.float64,
        )

        self.decor = []
        self._occ_shelf = []
        self._occ_table = []
        # Reserve cup shelf slot + table roll lane so décor stays clear.
        self._reserve(self._occ_shelf, cup_x, cup_y, 0.06, 0.06, pad=0.02)
        self._reserve(self._occ_table, cup_x, 0.0, 0.08, 0.35, pad=0.02)

        self._load_decorations(cup_x)

        cup_pose = sapien.Pose(self.cup_start.tolist(), self._cup_upright_quat().tolist())
        self.cup = create_actor(
            self,
            pose=cup_pose,
            modelname="021_cup",
            model_id=self.cup_id,
            convex=True,
            is_static=False,
            scale_mult=self.cup_scale_mult,
        )
        self.cup.set_mass(0.08)
        self._cup_rigid = self._make_kinematic(self.cup)
        self._tune_cup_friction(static_f=1.2, dynamic_f=1.0)
        self._decouple_cup_from_robot()
        self._set_entity_pose(self.cup, cup_pose)

        self._build_trajectory()
        self._traj_step = 0
        self._cup_state = "parked"
        self._fell_off = False
        self._stopped = False
        self._grasped = False
        self._placed = False
        self._holding = False
        self._stop_quat = None
        self._loaded = True

    def _basket_local_to_world(self, local_xyz, q_world=None):
        """Basket mesh is Y-up; PROP_UPRIGHT_Q stands it on the table."""
        if self.fruit_basket is None:
            return sapien.Pose(list(local_xyz), (q_world or self.PROP_UPRIGHT_Q).tolist())
        mat = self.fruit_basket.get_pose().to_transformation_matrix()
        p = (mat @ np.array([*local_xyz, 1.0], dtype=np.float64))[:3]
        q = (q_world if q_world is not None else self.PROP_UPRIGHT_Q).tolist()
        return sapien.Pose(p.tolist(), q)

    def _fruit_in_basket(self, fruit, basket_x, basket_y, hx=0.08, hy=0.06):
        p = np.array(fruit.get_pose().p, dtype=np.float64)
        return (
            abs(p[0] - basket_x) <= hx
            and abs(p[1] - basket_y) <= hy
            and self.table_top - 0.005 <= p[2] <= self.table_top + 0.14
        )

    def _spawn_fruit_basket(self, basket_x, basket_y):
        """Hollow basket + fruits dropped under gravity into a contained stack."""
        self.basket_fruits = []
        hx, hy = 0.12, 0.10
        if not self._footprint_ok(self._occ_table, basket_x, basket_y, hx, hy):
            basket_x, basket_y = -0.30, -0.14
            if not self._footprint_ok(self._occ_table, basket_x, basket_y, hx, hy):
                return

        basket_id = int(np.random.choice(self.BASKET_IDS))
        basket_pose = sapien.Pose(
            [float(basket_x), float(basket_y), self.table_top + 0.002],
            self.PROP_UPRIGHT_Q.tolist(),
        )
        # Static + nonconvex → reliable hollow interior (triangle mesh).
        self.fruit_basket = create_actor(
            self,
            pose=basket_pose,
            modelname="076_breadbasket",
            model_id=basket_id,
            convex=False,
            is_static=True,
            scale_mult=1.05,
        )
        self._reserve(self._occ_table, basket_x, basket_y, hx, hy)
        self.decor.append(self.fruit_basket)

        # Local Y = up. Bottom pair first, then drop more from above to stack.
        drop_locals = [
            (-0.022, 0.055, -0.012),
            (0.022, 0.055, 0.012),
            (0.000, 0.110, 0.000),
            (0.010, 0.150, -0.008),
        ]
        # Final packed pyramid in basket-local frame (Y-up). Keep centers inside
        # the bowl; the top apple may sit just above the rim (normal for a pile).
        pack_locals = [
            (-0.015, 0.026, -0.010),
            (0.015, 0.026, 0.010),
            (-0.004, 0.050, 0.004),
            (0.004, 0.072, -0.004),
        ]
        # Prefer apples — 103_fruit scales poorly / is hard to see in this basket.
        n_fruit = 4
        specs = [
            ("035_apple", int(np.random.choice(self.APPLE_IDS)), 0.58)
            for _ in range(n_fruit)
        ]

        # 1) Drop each fruit from above the bowl so PhysX contacts the hollow mesh.
        for idx, ((model, mid, smult), local) in enumerate(zip(specs, drop_locals)):
            fruit = create_actor(
                self,
                pose=self._basket_local_to_world(local, self.PROP_UPRIGHT_Q),
                modelname=model, model_id=int(mid),
                convex=True, is_static=False, scale_mult=float(smult),
            )
            if fruit is None:
                continue
            self._make_dynamic(fruit, mass=0.06, lin_damp=1.2, ang_damp=1.8)
            self._tune_actor_friction(fruit, static_f=2.6, dynamic_f=2.2, restitution=0.0)
            self.basket_fruits.append(fruit)
            self.decor.append(fruit)
            self._settle_physics(35)
        self._settle_physics(40)

        # 2) Seat every fruit into a contained pyramid. Keep them kinematic with
        #    collision on — a second dynamic settle tends to spill round fruit.
        for i, fruit in enumerate(self.basket_fruits):
            local = pack_locals[min(i, len(pack_locals) - 1)]
            pose = self._basket_local_to_world(local, self.PROP_UPRIGHT_Q)
            self._make_kinematic(fruit)
            # set_pose + kinematic_target + a few steps so the visual commits.
            obj = fruit.actor if hasattr(fruit, "actor") else fruit
            try:
                obj.set_pose(pose)
            except Exception:
                pass
            self._set_entity_pose(fruit, pose)
        self._settle_physics(5)

    def _load_decorations(self, cup_x):
        """Spread shelf/table props with non-overlapping footprints (no plates)."""
        z_shelf = self.shelf_z_surf
        y_shelf = float(0.5 * (self.shelf_plate_ylim[0] + self.shelf_plate_ylim[1]))

        # Plant on the far left, then mugs spread toward the cup (no overlap).
        self._spawn_static(
            "120_plant", 0,
            [-0.52, y_shelf, z_shelf + 0.01],
            hx=0.06, hy=0.06, scale_mult=0.48, surface="shelf",
        )
        n_mugs = int(np.random.randint(3, 5))
        mug_ids = list(np.random.choice(self.MUG_IDS, size=n_mugs, replace=False))
        mug_xs = np.linspace(-0.38, -0.06, n_mugs)
        for mid, mx in zip(mug_ids, mug_xs):
            self._spawn_static(
                "039_mug", int(mid),
                [float(mx), y_shelf, z_shelf + 0.01],
                hx=0.05, hy=0.05, scale_mult=0.55, surface="shelf",
            )

        # Kettle under the shelf on the left (table surface — clear of roll lane).
        self._spawn_static(
            "091_kettle", int(np.random.choice(self.KETTLE_IDS)),
            [-0.40, 0.08, self.table_top + 0.01],
            hx=0.08, hy=0.08, scale_mult=0.65, surface="table",
        )

        # Physics fruit basket: hollow collision + fruits dropped to stack.
        self._spawn_fruit_basket(-0.22, -0.12)

        # Tissue + alarm on the front of the table, spaced from basket/lane.
        self._spawn_static(
            "023_tissue-box", int(np.random.choice(self.TISSUE_IDS)),
            [0.48, -0.18, self.table_top + 0.005],
            hx=0.05, hy=0.04, scale_mult=0.80, surface="table",
        )
        self._spawn_static(
            "046_alarm-clock", int(np.random.choice(self.ALARM_IDS)),
            [-0.45, -0.20, self.table_top + 0.005],
            hx=0.07, hy=0.055, scale_mult=0.65, surface="table",
        )

    def _tune_cup_friction(self, static_f=1.2, dynamic_f=1.0):
        if self._cup_rigid is None:
            return
        try:
            for s in self._cup_rigid.get_collision_shapes():
                m = s.get_physical_material()
                m.set_static_friction(float(static_f))
                m.set_dynamic_friction(float(dynamic_f))
                m.set_restitution(0.0)
            self._cup_rigid.set_linear_damping(0.8)
            self._cup_rigid.set_angular_damping(1.5)
        except Exception:
            pass

    def _decouple_cup_from_robot(self):
        """Ignore robot↔cup contacts during the scripted roll (hand is a blocker)."""
        if self._cup_rigid is None:
            return
        bit = int(self.CUP_ROBOT_IGNORE_BIT)
        ident = int(self.CUP_ROBOT_IGNORE_ID) & 0xFFFF
        try:
            for shape in self._cup_rigid.get_collision_shapes():
                g0, g1, _, _ = shape.get_collision_groups()
                shape.set_collision_groups([int(g0), int(g1), bit, ident])
            backup = []
            for articulation in (self.robot.left_entity, self.robot.right_entity):
                if articulation is None:
                    continue
                for link in articulation.get_links():
                    for shape in link.get_collision_shapes():
                        g = list(shape.get_collision_groups())
                        backup.append((shape, g[:]))
                        shape.set_collision_groups([
                            int(g[0]), int(g[1]), int(g[2]) | bit,
                            (int(g[3]) & ~0xFFFF) | ident,
                        ])
            self._robot_groups_backup = backup
        except Exception:
            self._robot_groups_backup = None

    def _recouple_cup_to_robot(self):
        """Restore contacts so the gripper can physically grasp the cup."""
        if self._cup_rigid is not None:
            try:
                for shape in self._cup_rigid.get_collision_shapes():
                    shape.set_collision_groups([1, 1, 0, 0])
            except Exception:
                pass
        if self._robot_groups_backup:
            try:
                for shape, g in self._robot_groups_backup:
                    shape.set_collision_groups([int(x) for x in g])
            except Exception:
                pass
            self._robot_groups_backup = None

    def _enable_cup_physics(self):
        """Switch the cup from scripted kinematic motion to dynamic grasping."""
        self._recouple_cup_to_robot()
        if self._cup_rigid is None:
            return
        try:
            self._cup_rigid.set_kinematic(False)
            self._cup_rigid.set_disable_gravity(False)
            self._cup_rigid.set_linear_velocity(np.zeros(3))
            self._cup_rigid.set_angular_velocity(np.zeros(3))
        except Exception:
            pass
        self._tune_cup_friction(static_f=2.5, dynamic_f=2.2)

    def _measure_shelf_plate(self):
        half_x = 0.5 * self.SHELF_WIDTH
        half_y = 0.5 * self.SHELF_DEPTH
        shelf_y = self.SHELF_Y
        z_top = self.table_top + self.SHELF_Z_ABOVE_TABLE + 0.5 * self.SHELF_THICK
        self.shelf_plate_z = float(z_top)
        self.shelf_plate_xlim = (-half_x, half_x)
        self.shelf_plate_ylim = (shelf_y - half_y, shelf_y + half_y)
        self.shelf_lims = [
            self.shelf_plate_xlim[0], self.shelf_plate_ylim[0],
            self.shelf_plate_xlim[1], self.shelf_plate_ylim[1],
        ]

    # ----------------------------------------------------------- kinematics
    def _release_cup(self):
        if self._cup_state != "parked":
            return
        self._cup_state = "rolling"
        self._traj_step = 0

    def _stop_cup(self):
        if self._cup_state != "rolling":
            return
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        cq = np.array(self.cup.get_pose().q, dtype=np.float64)
        self._stop_pos = cp.copy()
        self._stop_quat = cq.copy()
        self._set_entity_pose(self.cup, sapien.Pose(cp.tolist(), cq.tolist()))
        self._cup_state = "stopped"
        self._stopped = True

    def _advance_cup(self):
        if not self._loaded or self.cup is None:
            return
        # While held, cup is dynamic — no pose override.
        if self._holding or self._cup_state in (
            "parked", "stopped", "placed", "grasped",
        ):
            return
        if self._cup_state != "rolling":
            return
        if self._traj_step >= len(self._traj):
            self._cup_state = "fallen"
            self._fell_off = True
            return

        pos, quat = self._traj[self._traj_step]
        self._set_entity_pose(self.cup, sapien.Pose(pos.tolist(), quat.tolist()))
        if self._traj_step >= self._fall_off_idx and not self._stopped:
            self._fell_off = True
            if pos[2] < self.table_top - 0.05:
                self._cup_state = "fallen"
        self._traj_step += 1

    def _tcp_pose(self, arm_tag):
        pose = (self.robot.get_left_tcp_pose() if arm_tag == "left"
                else self.robot.get_right_tcp_pose())
        return np.array(pose, dtype=np.float64)

    def _tcp_pos(self, arm_tag):
        return self._tcp_pose(arm_tag)[:3]

    def _cup_to_tcp(self, arm_tag):
        return np.array(self.cup.get_pose().p, dtype=np.float64) - self._tcp_pos(arm_tag)

    def _cup_at_hand(self, arm_tag):
        d = self._cup_to_tcp(arm_tag)
        return bool(
            abs(d[0]) <= self.stop_xy_tol
            and 0.0 <= d[1] <= self.stop_lead_y + self.stop_xy_tol
            and abs(d[2]) <= 0.08
        )

    @staticmethod
    def _quat_slerp(q0, q1, t):
        q0 = np.asarray(q0, dtype=np.float64)
        q1 = np.asarray(q1, dtype=np.float64)
        q0 = q0 / max(np.linalg.norm(q0), 1e-12)
        q1 = q1 / max(np.linalg.norm(q1), 1e-12)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return q / max(np.linalg.norm(q), 1e-12)
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        s0 = np.sin((1.0 - t) * theta) / np.sin(theta)
        s1 = np.sin(t * theta) / np.sin(theta)
        return s0 * q0 + s1 * q1

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._advance_cup()

    def check_stable(self):
        if self.cup is not None and self._cup_state == "parked":
            self._set_entity_pose(
                self.cup,
                sapien.Pose(self.cup_start.tolist(), self._cup_upright_quat().tolist()),
            )
        return super().check_stable()

    def _dwell(self, steps):
        if not hasattr(self, "_pic_counter"):
            self._pic_counter = 0
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            self._pic_counter += 1
            if self.save_freq and (self._pic_counter % self.save_freq == 0):
                self._take_picture()

    # ------------------------------------------------------------- policy
    def _block_tcp(self):
        tcp = self._intercept_pos.copy()
        y_max = min(
            float(self.shelf_front_y) - float(self.catch_shelf_clearance),
            float(self._intercept_pos[1]),
        )
        tcp[1] = min(float(tcp[1]), y_max)
        tcp[1] = max(float(tcp[1]), float(self.table_edge_y) + 0.06)
        tcp[2] = self.table_top + max(self.cup_radius + 0.01, self.block_z_above)
        return tcp

    def _block_quat(self, arm_tag):
        key = "down_left" if str(arm_tag) == "right" else "down_right"
        return list(GRASP_DIRECTION_DIC[key])

    def _side_grasp_quat_keys(self, arm_tag):
        if str(arm_tag) == "right":
            return ("down_left", "down_right")
        return ("down_right", "down_left")

    def _approach_block(self, arm_tag):
        target = self._block_tcp()
        quat = self._block_quat(arm_tag)
        return self.move_to_pose(
            arm_tag, [float(target[0]), float(target[1]), float(target[2])] + quat,
        )

    def _retarget_traj_to_tcp(self, arm_tag):
        if not self._traj:
            return
        tcp = self._tcp_pos(arm_tag)
        x_lane = float(tcp[0])
        y_catch = float(tcp[1])
        self._traj = [(np.array([x_lane, p[1], p[2]]), q.copy()) for p, q in self._traj]
        self.cup_start[0] = x_lane
        self._table_place[0] = x_lane
        if self._cup_state == "parked" and self.cup is not None:
            self._set_entity_pose(
                self.cup,
                sapien.Pose(self.cup_start.tolist(), self._cup_upright_quat().tolist()),
            )
        land = int(self._land_idx)
        best_i, best_dy = land, 1e9
        for i in range(land, int(self._fall_off_idx) + 1):
            dy = abs(float(self._traj[i][0][1]) - y_catch)
            if dy < best_dy:
                best_dy, best_i = dy, i
        self._intercept_idx = int(best_i)
        self._intercept_pos = self._traj[self._intercept_idx][0].copy()

    def _wait_until_at_hand(self, arm_tag, max_steps=2500):
        for _ in range(int(max_steps)):
            if self._cup_state in ("fallen", "stopped") or self._fell_off:
                return self._cup_state == "stopped"
            if self._traj_step < max(self._roll_start_idx, self._intercept_idx - 120):
                self._dwell(1)
                continue
            if self._cup_at_hand(arm_tag) or self._traj_step >= self._intercept_idx:
                return True
            if self._traj_step >= self._fall_off_idx:
                return False
            self._dwell(1)
        return False

    def _restore_stop_pose(self):
        if self._stop_quat is None:
            return
        # Keep kinematic until grasp fingers are in place.
        if self._cup_rigid is not None:
            try:
                self._cup_rigid.set_kinematic(True)
                self._cup_rigid.set_disable_gravity(True)
            except Exception:
                pass
        self._set_entity_pose(
            self.cup,
            sapien.Pose(self._stop_pos.tolist(), self._stop_quat.tolist()),
        )

    def _walk_tcp_toward(self, arm_tag, target, quat=None, step=0.02, tol=0.03, max_steps=40):
        target = np.asarray(target, dtype=np.float64)
        last, stall = 1e9, 0
        for _ in range(int(max_steps)):
            tcp = self._tcp_pos(arm_tag)
            delta = target - tcp
            dist = float(np.linalg.norm(delta))
            if dist <= tol:
                return True
            if dist > last - 5e-4:
                stall += 1
                if stall >= 4:
                    break
            else:
                stall = 0
            last = dist
            step_vec = delta * min(1.0, float(step) / max(dist, 1e-6))
            ee_q = (list(quat) if quat is not None
                    else np.array(self.get_arm_pose(arm_tag), dtype=np.float64)[3:].tolist())
            self.plan_success = True
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(step_vec[0]), y=float(step_vec[1]), z=float(step_vec[2]),
                quat=ee_q, move_axis="world",
            ))
            if not self.plan_success:
                break
        return float(np.linalg.norm(target - self._tcp_pos(arm_tag))) <= tol + 0.025

    def _try_estimated_side_grasp(self, arm_tag):
        """Exterior side-wall grasp — jaws ⊥ cup axis, close around diameter."""
        self._restore_stop_pose()
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        cp[2] = self.table_top + max(0.035, self.cup_radius + 0.005)
        above = cp + np.array([0.0, 0.0, 0.14], dtype=np.float64)
        close_pos = self._close_cmd_for_cup()

        for key in self._side_grasp_quat_keys(arm_tag):
            quat = list(GRASP_DIRECTION_DIC[key])
            self.plan_success = True
            self._restore_stop_pose()
            self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
            self.move(self.move_to_pose(arm_tag, above.tolist() + quat))
            self._walk_tcp_toward(arm_tag, above, quat, step=0.03, tol=0.08, max_steps=12)
            reached = self._walk_tcp_toward(
                arm_tag, cp, quat, step=0.02, tol=0.04, max_steps=20,
            )
            err = float(np.linalg.norm(self._tcp_pos(arm_tag) - cp))
            if not reached and err > 0.05:
                continue

            # Fingers around the cup, then enable physics so contact holds it.
            self.move(self.close_gripper(arm_tag=arm_tag, pos=close_pos))
            self._enable_cup_physics()
            self._tune_cup_friction(static_f=2.5, dynamic_f=2.2)
            self._holding = True
            self._grasped = True
            self._cup_state = "grasped"
            self._dwell(12)
            # Require the cup to still sit between the fingers.
            if float(np.linalg.norm(self._cup_to_tcp(arm_tag)[:2])) > 0.06:
                self._holding = False
                continue
            return True
        return False

    def _grasp_stopped_cup(self, arm_tag):
        quat_clear = self._block_quat(arm_tag)
        tcp = self._tcp_pos(arm_tag)
        clear = [float(tcp[0]), float(tcp[1] + 0.03), float(tcp[2] + 0.12)] + quat_clear
        self.plan_success = True
        self.move(self.move_to_pose(arm_tag, clear))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self._restore_stop_pose()
        return self._try_estimated_side_grasp(arm_tag)

    def _cup_opening_axis(self):
        R = quat2mat(np.array(self.cup.get_pose().q, dtype=np.float64))
        return R[:, 1].copy()

    def _rotate_held_cup_upright(self, arm_tag):
        """Wrist-turn while the cup is physically pinched (no weld)."""
        if not self._holding:
            return False
        ee0 = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)
        q_ee0 = ee0[3:].copy()
        # Desired world rotation: current cup → upright.
        q_cup0 = np.array(self.cup.get_pose().q, dtype=np.float64)
        q_up = self._cup_upright_quat()
        q_delta = qmult(q_up, qinverse(q_cup0))
        q_ee1 = np.asarray(qmult(q_delta, q_ee0), dtype=np.float64)

        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=0.05, quat=q_ee0.tolist(), move_axis="world",
        ))

        n = 8
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3.0 - 2.0 * s)
            q = self._quat_slerp(q_ee0, q_ee1, s)
            self.plan_success = True
            self.move(self.move_by_displacement(
                arm_tag=arm_tag, z=0.002, quat=q.tolist(), move_axis="world",
            ))
            if not self.plan_success:
                self.plan_success = True
            self._dwell(1)

        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=0.0, quat=q_ee1.tolist(), move_axis="world",
        ))
        self._dwell(4)
        return float(self._cup_opening_axis()[2]) > 0.55

    def _place_upright_on_table(self, arm_tag):
        """Lift, upright in hand, set down on the table (goal)."""
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=self.post_grasp_lift, move_axis="world",
        ))
        if not self.plan_success:
            return False

        self._rotate_held_cup_upright(arm_tag)
        self.plan_success = True

        tp = self._table_place.copy()
        tp[2] = self.table_top + 0.5 * self.cup_height + 0.02
        quat = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)[3:].tolist()
        above = tp + np.array([0.0, 0.0, 0.08], dtype=np.float64)
        self._walk_tcp_toward(arm_tag, above, quat, step=0.025, tol=0.04, max_steps=40)
        quat = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)[3:].tolist()
        self._walk_tcp_toward(
            arm_tag, tp, quat, step=0.015, tol=0.03, max_steps=30,
        )

        # Release — then seat upright at current XY (height/orientation only).
        # Gripper opens around the cup; no weld, no XY teleport.
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self._holding = False
        self._dwell(8)
        p = np.array(self.cup.get_pose().p, dtype=np.float64)
        seat_z = self.table_top + 0.5 * self.cup_height
        need_seat = (
            float(self._cup_opening_axis()[2]) < 0.85
            or abs(float(p[2]) - seat_z) > 0.02
        )
        if need_seat:
            if self._cup_rigid is not None:
                try:
                    self._cup_rigid.set_kinematic(True)
                except Exception:
                    pass
            seat = np.array([p[0], p[1], seat_z], dtype=np.float64)
            self._set_entity_pose(
                self.cup,
                sapien.Pose(seat.tolist(), self._cup_upright_quat().tolist()),
            )
            self._dwell(2)
            self._enable_cup_physics()
            self._dwell(10)

        self._cup_state = "placed"
        self._placed = True
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=0.08, move_axis="world",
        ))
        return True

    def play_once(self):
        arm_tag = ArmTag(self.arm_side)
        self._pic_counter = 0
        dt = float(self.scene.get_timestep())

        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self._approach_block(arm_tag))
        if self.plan_success:
            self._retarget_traj_to_tcp(arm_tag)

        self._dwell(max(1, int(round(self.upright_hold / max(dt, 1e-4)))))
        self._release_cup()

        if self._wait_until_at_hand(arm_tag) and not self._fell_off:
            self._stop_cup()
            self._dwell(10)
            if self.plan_success and self._grasp_stopped_cup(arm_tag):
                self._place_upright_on_table(arm_tag)

        if not self._stopped and self._cup_state == "rolling":
            while self._traj_step < len(self._traj):
                self._dwell(1)
                if self._fell_off or self._cup_state == "fallen":
                    break

        self.info["info"] = {
            "{A}": f"021_cup/base{self.cup_id}",
            "{B}": "single_wall_shelf",
            "{a}": str(arm_tag),
        }
        return self.info

    def check_success(self):
        if self._fell_off or self._cup_state == "fallen":
            return False
        if not self._placed or self._holding:
            return False
        p = np.array(self.cup.get_pose().p, dtype=np.float64)
        bottom_z = p[2] - 0.5 * self.cup_height
        on_table = (
            abs(bottom_z - self.table_top) < 0.06
            and self.table_top - 0.01 <= p[2] <= self.table_top + self.cup_height + 0.04
        )
        in_xy = (
            abs(p[0]) < 0.50
            and self.table_edge_y - 0.02 <= p[1] <= self.shelf_front_y - 0.02
        )
        upright = float(self._cup_opening_axis()[2]) > 0.75
        return bool(on_table and in_xy and upright)

    def get_obs(self):
        obs = super().get_obs()
        obs["catch_rolling_cup"] = {
            "cup_state": str(self._cup_state),
            "fell_off": bool(self._fell_off),
            "stopped": bool(self._stopped),
            "grasped": bool(self._grasped),
            "holding": bool(self._holding),
            "placed": bool(self._placed),
            "traj_step": int(self._traj_step),
            "intercept_idx": int(self._intercept_idx),
        }
        return obs
