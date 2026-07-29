"""Pour beer from a bottle into a glass, pausing so foam can settle (KitchenS).

Expert flow (kept visually grounded):
  1) freeze bottle, side-grasp, seat TCP on the contact, weld
  2) lift + carry toward the glass
  3) tip the bottle *around the TCP* (stays in the fingers)
  4) liquid flows only while held + tipped + mouth-over-glass
  5) untilt, place back, release
"""
from __future__ import annotations

import numpy as np
import sapien
import sapien.render
import transforms3d as t3d

from ._kitchens_base_task import KitchenS_base_task
from .utils import *
from .utils.create_actor import create_actor


class pour_beer(KitchenS_base_task):
    """Pour bottle beer into a glass without foam overflow."""

    BOTTLE_MODEL = "255_beer_bottle"
    GLASS_MODEL = "257_beer_glass"

    BOTTLE_BEER_FRAC = 0.90
    FULL_LIQUID_TOL = 0.08
    TARGET_LIQUID = 0.80

    POUR_RATE = 0.016
    FOAM_GAIN = 1.5
    FOAM_DECAY = 0.0025
    OVERFLOW_LEVEL = 1.0
    EXPERT_FOAM_RESUME = 0.20
    MAX_POUR_CYCLES = 8

    BEER_COLOR = [0.78, 0.52, 0.10, 0.78]
    FOAM_COLOR = [0.97, 0.95, 0.90, 0.88]
    GLASS_RGBA = [0.82, 0.93, 0.98, 0.22]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]
    BOTTLE_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    GLASS_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]

    # Bottle mesh long-axis is local +Y; upright → world +Z.
    BOTTLE_HEIGHT = 0.220
    CONTACT_ALONG = 0.124  # mid-body contact from origin (scaled model)
    POUR_TILT_DOT = 0.50
    MOUTH_POUR_XY_TOL = 0.10
    GRASP_TCP_TOL = 0.035
    EE_TO_TCP = 0.12

    def setup_demo(self, **kwags):
        self._cfg = dict(kwags.get("task_args", {}).get("pour_beer", {}))
        if kwags.get("scene_id") is None:
            kwags["scene_id"] = int(self._cfg.get("scene_id", 0))
        # Bar counter: solid top, no sink / stove / microwave.
        self.clear_sink_and_range = True
        self.replace_sink_with_range = False

        self._loaded = False
        self.pouring = False
        self.overflowed = False
        self.liquid_level = 0.0
        self.foam_level = 0.0
        self.bottle_beer = float(self.BOTTLE_BEER_FRAC)
        self._liquid_entity = None
        self._foam_entity = None
        self._liquid_half_h_cached = -1.0
        self._foam_half_h_cached = -1.0
        self.cup = None
        self.bottle = None
        self.table_top = 0.74
        self._bottle_rigid = None
        self._bottle_welded = False
        self._bottle_weld_offset = None
        self._grasp_ee_quat = None
        self._bottle_dropped = False
        self._bottle_spawn_pose = None
        self._pin_bottle_spawn = False  # lock bottle on table during approach
        self._bar_props = []

        super().setup_demo(**kwags)
        self._style_bar_wall()
        self._configure_observer_camera()

    def _load_microwave(self, table_height, table_xy_bias):
        """Bar scene — leave the back counter free of kitchen appliances."""
        return

    def _style_bar_wall(self):
        """Warm the back wall so the solid counter reads more like a bar."""
        wall = getattr(self, "wall", None)
        if wall is None:
            return
        try:
            ent = wall.actor if hasattr(wall, "actor") else wall
            for c in ent.get_components():
                if not isinstance(c, sapien.render.RenderBodyComponent):
                    continue
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color([0.22, 0.12, 0.10, 1.0])
                        s.material.set_roughness(0.85)
                    except Exception:
                        try:
                            s.material.base_color = [0.22, 0.12, 0.10, 1.0]
                        except Exception:
                            pass
        except Exception:
            pass

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.10, -0.50, 1.35], dtype=np.float64)
        look_at = np.array([0.10, -0.02, 0.90], dtype=np.float64)
        forward = look_at - camera_pos
        forward /= np.linalg.norm(forward)
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        left /= np.linalg.norm(left)
        up = np.cross(forward, left)
        m = np.eye(4)
        m[:3, :3] = np.stack([forward, left, up], axis=1)
        m[:3, 3] = camera_pos
        camera.entity.set_pose(sapien.Pose(m))

    def _fluid_material(self, rgba):
        mat = sapien.render.RenderMaterial(base_color=list(rgba))
        try:
            mat.set_roughness(0.18)
            mat.set_metallic(0.0)
        except Exception:
            mat.roughness = 0.18
            mat.metallic = 0.0
        return mat

    def _make_glass_transparent(self, actor):
        try:
            for c in actor.actor.get_components():
                if not isinstance(c, sapien.render.RenderBodyComponent):
                    continue
                for s in c.render_shapes:
                    try:
                        s.material.set_base_color(list(self.GLASS_RGBA))
                        s.material.set_transmission(0.88)
                        s.material.set_transmission_roughness(0.03)
                        s.material.set_roughness(0.05)
                        s.material.set_metallic(0.0)
                        s.material.set_ior(1.45)
                    except Exception:
                        try:
                            s.material.base_color = list(self.GLASS_RGBA)
                            s.material.transmission = 0.88
                        except Exception:
                            pass
        except Exception:
            pass

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.table_top = float(self.kitchens_info["table_height"]) + float(self.table_z_bias)

        self.bottle_beer_frac = float(cfg.get("bottle_beer_frac", self.BOTTLE_BEER_FRAC))
        self.pour_rate = float(cfg.get("pour_rate", self.POUR_RATE))
        self.foam_gain = float(cfg.get("foam_gain", self.FOAM_GAIN))
        self.foam_decay = float(cfg.get("foam_decay", self.FOAM_DECAY))
        self.overflow_level = float(cfg.get("overflow_level", self.OVERFLOW_LEVEL))
        self.expert_foam_resume = float(cfg.get("expert_foam_resume", self.EXPERT_FOAM_RESUME))
        self.target_liquid = float(cfg.get("target_liquid", self.TARGET_LIQUID))
        self.full_liquid_tol = float(cfg.get("full_liquid_tol", self.FULL_LIQUID_TOL))
        self.max_pour_cycles = int(cfg.get("max_pour_cycles", self.MAX_POUR_CYCLES))

        self.liquid_level = 0.0
        self.foam_level = 0.0
        self.bottle_beer = float(self.bottle_beer_frac)
        self.pouring = False
        self.overflowed = False
        self._liquid_entity = None
        self._foam_entity = None
        self._liquid_half_h_cached = -1.0
        self._foam_half_h_cached = -1.0
        self._bottle_rigid = None
        self._bottle_welded = False
        self._bottle_weld_offset = None
        self._grasp_ee_quat = None
        self._bottle_dropped = False
        self._bottle_spawn_pose = None
        self._pin_bottle_spawn = False

        # Pour station in front of the right arm; bar props stay along the back.
        side = float(cfg.get("station_x", 0.10))
        self.arm = ArmTag("right" if side >= 0 else "left")
        cup_y = float(cfg.get("cup_y", -0.08))
        bottle_y = float(cfg.get("bottle_y", 0.04))
        bottle_dx = float(cfg.get("bottle_dx", 0.09 if side >= 0 else -0.09))
        self.cup_xy = np.array([side, cup_y], dtype=float)
        self.bottle_xy = np.array([side + bottle_dx, bottle_y], dtype=float)

        self._bar_props = []
        self._build_bar_props()
        self._spawn_glass()
        self._spawn_bottle()
        self._rebuild_fluids(force=True)
        self.add_prohibit_area(self.cup, padding=0.04)
        self.add_prohibit_area(self.bottle, padding=0.04)
        self._loaded = True
        print(
            f"[pour_beer] scene={self.scene_id} arm={self.arm} "
            f"cup={self.cup_xy} bottle={self.bottle_xy} bar_props={len(self._bar_props)}"
        )

    def _yaw_upright(self, yaw_deg: float = 0.0) -> list:
        """Y-up mesh upright quat with an optional world yaw (degrees)."""
        base = np.array(self.BOTTLE_UPRIGHT_Q, dtype=float)
        if abs(yaw_deg) < 1e-6:
            return base.tolist()
        yaw_q = t3d.euler.euler2quat(0.0, 0.0, np.deg2rad(yaw_deg), axes="sxyz")
        return t3d.quaternions.qmult(yaw_q, base).tolist()

    def _spawn_static_prop(
        self,
        modelname: str,
        xy,
        model_id: int = 0,
        yaw_deg: float = 0.0,
        z_off: float = 0.001,
        scale_mult: float = 1.0,
        prohibit: float = 0.03,
    ):
        pose = sapien.Pose(
            [float(xy[0]), float(xy[1]), self.table_top + float(z_off)],
            self._yaw_upright(yaw_deg),
        )
        actor = create_actor(
            self,
            pose=pose,
            modelname=modelname,
            model_id=int(model_id),
            convex=True,
            is_static=True,
            scale_mult=scale_mult,
        )
        if actor is None:
            print(f"[pour_beer] skip prop {modelname}/base{model_id}")
            return None
        actor.set_name(f"bar_{modelname}_{model_id}")
        self._bar_props.append(actor)
        try:
            self.add_prohibit_area(actor, padding=float(prohibit))
        except Exception:
            pass
        return actor

    def _build_bar_props(self):
        """Decorate the solid counter like a bar: back bottles/glasses + snacks."""
        # Back shelf row (near backsplash) — bottles and glasses.
        back_y = 0.24
        back_bottles = [
            ("255_beer_bottle", 0, -0.48, -12),
            ("001_bottle", 2, -0.40, 8),
            ("114_bottle", 1, -0.32, -5),
            ("255_beer_bottle", 0, -0.24, 15),
            ("001_bottle", 5, 0.28, -10),
            ("114_bottle", 2, 0.36, 6),
            ("255_beer_bottle", 0, 0.44, -8),
            ("001_bottle", 8, 0.52, 12),
        ]
        for model, mid, x, yaw in back_bottles:
            self._spawn_static_prop(model, [x, back_y], model_id=mid, yaw_deg=yaw, prohibit=0.025)

        # Wine glasses + cups nestled among the bottles.
        glass_row = [
            ("088_wineglass", 0, -0.16, back_y - 0.02, 0),
            ("088_wineglass", 1, -0.10, back_y - 0.02, 20),
            ("088_wineglass", 2, 0.18, back_y - 0.02, -15),
            ("021_cup", 0, 0.08, back_y - 0.04, 30),
            ("021_cup", 1, 0.14, back_y - 0.04, -25),
            ("039_mug", 0, -0.02, back_y - 0.03, 40),
        ]
        for model, mid, x, y, yaw in glass_row:
            self._spawn_static_prop(model, [x, y], model_id=mid, yaw_deg=yaw, prohibit=0.02)

        # Snacks along the sides — leave the pour station clear.
        self._spawn_static_prop("025_chips-tub", [-0.42, 0.02], model_id=0, yaw_deg=-25, prohibit=0.05)
        self._spawn_static_prop("025_chips-tub", [0.48, -0.06], model_id=2, yaw_deg=35, prohibit=0.05)
        self._spawn_static_prop("068_boxdrink", [-0.48, -0.10], model_id=0, yaw_deg=15, prohibit=0.03)
        self._spawn_static_prop("071_can", [-0.38, -0.12], model_id=0, yaw_deg=-40, prohibit=0.02)
        self._spawn_static_prop("071_can", [0.40, 0.08], model_id=0, yaw_deg=50, prohibit=0.02)
        self._spawn_static_prop("105_sauce-can", [0.50, 0.10], model_id=0, yaw_deg=-20, prohibit=0.03)
        self._spawn_static_prop("019_coaster", [0.02, -0.18], model_id=0, yaw_deg=0, prohibit=0.015)
        self._spawn_static_prop("019_coaster", [0.18, -0.20], model_id=0, yaw_deg=25, prohibit=0.015)
        self._spawn_static_prop("054_baguette", [-0.50, 0.14], model_id=2, yaw_deg=70, prohibit=0.04)

    def _spawn_glass(self):
        pose = sapien.Pose(
            [float(self.cup_xy[0]), float(self.cup_xy[1]), self.table_top + 0.001],
            self.GLASS_UPRIGHT_Q,
        )
        self.cup = create_actor(
            self, pose=pose, modelname=self.GLASS_MODEL, model_id=0, convex=True, is_static=True
        )
        self.cup.set_name("beer_glass")
        self._make_glass_transparent(self.cup)

        cfg = getattr(self.cup, "config", {}) or {}
        ext = np.array(cfg.get("extents", [0.08, 0.13, 0.08]), dtype=float)
        sc = cfg.get("scale", [1, 1, 1])
        sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
        world = ext * sc
        self.cup_height = float(world[1])
        self.cup_inner_r = 0.36 * float(max(world[0], world[2]))
        self.cup_bottom_z = self.table_top + 0.010
        self.cup_fillable_h = max(0.08, self.cup_height - 0.025)
        self.cup_rim_z = self.cup_bottom_z + self.cup_fillable_h

    def _spawn_bottle(self):
        pose = sapien.Pose(
            [float(self.bottle_xy[0]), float(self.bottle_xy[1]), self.table_top + 0.002],
            self.BOTTLE_UPRIGHT_Q,
        )
        self.bottle = create_actor(
            self, pose=pose, modelname=self.BOTTLE_MODEL, model_id=0, convex=True, is_static=False
        )
        self.bottle.set_mass(0.15)
        self.bottle.set_name("beer_bottle")
        self.bottle_id = 0
        self._bottle_spawn_pose = pose
        self._bottle_rigid = None
        for c in self.bottle.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_damping(2.0)
                    c.set_angular_damping(2.0)
                except Exception:
                    pass
                self._bottle_rigid = c
                break

    # ------------------------------------------------------------------ fluids
    def _cup_center_xy(self) -> np.ndarray:
        try:
            return np.asarray(self.cup.get_pose().p[:2], dtype=float)
        except Exception:
            return self.cup_xy.copy()

    def _total_fill(self) -> float:
        return float(self.liquid_level + self.foam_level)

    def _remove_entity(self, ent):
        if ent is None:
            return None
        try:
            self.scene.remove_entity(ent)
        except Exception:
            pass
        return None

    def _make_column(self, half_h, radius, z_center, rgba, name):
        xy = self._cup_center_xy()
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=radius,
            half_length=max(0.002, half_h),
            material=self._fluid_material(rgba),
        )
        builder.set_initial_pose(sapien.Pose(p=[float(xy[0]), float(xy[1]), float(z_center)]))
        return builder.build(name=name)

    def _rebuild_fluids(self, force: bool = False):
        if not getattr(self, "cup_fillable_h", None):
            return
        liq_h = max(0.0, float(self.liquid_level)) * self.cup_fillable_h
        foam_h = max(0.0, float(self.foam_level)) * self.cup_fillable_h
        liq_half = max(0.002, 0.5 * liq_h) if self.liquid_level > 1e-4 else 0.0
        foam_half = max(0.002, 0.5 * foam_h) if self.foam_level > 1e-4 else 0.0
        if (
            not force
            and abs(liq_half - self._liquid_half_h_cached) < 0.003
            and abs(foam_half - self._foam_half_h_cached) < 0.003
        ):
            return
        self._liquid_half_h_cached = liq_half
        self._foam_half_h_cached = foam_half
        self._liquid_entity = self._remove_entity(self._liquid_entity)
        self._foam_entity = self._remove_entity(self._foam_entity)
        r = self.cup_inner_r
        if self.liquid_level > 1e-4:
            self._liquid_entity = self._make_column(
                liq_half, r, self.cup_bottom_z + liq_half, self.BEER_COLOR, "beer_liquid"
            )
        if self.foam_level > 1e-4:
            self._foam_entity = self._make_column(
                foam_half, r * 0.98, self.cup_bottom_z + liq_h + foam_half, self.FOAM_COLOR, "beer_foam"
            )

    # ------------------------------------------------------------------ hold / weld
    def _ee_pose(self, arm: ArmTag) -> sapien.Pose:
        p = self.get_arm_pose(str(arm))
        return sapien.Pose(list(p[:3]), list(p[3:7]))

    def _tcp_pos(self, arm: ArmTag) -> np.ndarray:
        p = (
            self.robot.get_right_tcp_pose()
            if str(arm) == "right"
            else self.robot.get_left_tcp_pose()
        )
        return np.asarray(p[:3], dtype=float)

    def _contact_world(self, contact_id: int = 1) -> np.ndarray:
        try:
            fp = self.bottle.get_contact_point(contact_id, "pose")
            return np.asarray(fp.p if hasattr(fp, "p") else fp[:3], dtype=float)
        except Exception:
            p = np.asarray(self.bottle.get_pose().p, dtype=float)
            return p + self._bottle_axis() * self.CONTACT_ALONG

    def _set_bottle_pose(self, pose: sapien.Pose) -> None:
        self.bottle.actor.set_pose(pose)
        if self._bottle_rigid is not None:
            try:
                self._bottle_rigid.set_kinematic_target(pose)
            except Exception:
                pass

    def _freeze_bottle(self, pose: sapien.Pose | None = None) -> None:
        if pose is not None:
            self._set_bottle_pose(pose)
        if self._bottle_rigid is not None:
            try:
                self._bottle_rigid.set_disable_gravity(True)
                self._bottle_rigid.set_kinematic(True)
                self._bottle_rigid.set_linear_velocity(np.zeros(3))
                self._bottle_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _ignore_bottle_robot_collision(self) -> None:
        ignore_bit, ignore_id = 1 << 16, 0xBE
        ents = [self.bottle.actor]
        try:
            ents += list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
        except Exception:
            pass
        for ent in ents:
            try:
                shapes = []
                if hasattr(ent, "get_collision_shapes"):
                    shapes = list(ent.get_collision_shapes())
                else:
                    for c in ent.get_components():
                        if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                            shapes.extend(c.get_collision_shapes())
                for shape in shapes:
                    g0, g1, g2, g3 = shape.get_collision_groups()
                    shape.set_collision_groups(
                        [int(g0), int(g1), int(g2) | ignore_bit, (int(g3) & 0xFFFF0000) | ignore_id]
                    )
            except Exception:
                pass

    def _weld_bottle_to_ee(self, arm: ArmTag) -> bool:
        """Lock bottle at its current pose relative to the EE — no teleport to cup."""
        tcp = self._tcp_pos(arm)
        contact = self._contact_world(1)
        dist = float(np.linalg.norm(tcp - contact))
        if dist > 0.08:
            print(f"[pour_beer] refuse weld — TCP far from contact ({dist:.3f} m)")
            return False
        self._freeze_bottle()
        self._bottle_weld_offset = self._ee_pose(arm).inv() * self.bottle.get_pose()
        self._grasp_ee_quat = np.asarray(self._ee_pose(arm).q, dtype=float).copy()
        self._bottle_welded = True
        self._bottle_dropped = False
        self._ignore_bottle_robot_collision()
        print(
            f"[pour_beer] welded tcp-contact={dist:.3f} "
            f"ee-origin={float(np.linalg.norm(np.asarray(self._ee_pose(arm).p) - np.asarray(self.bottle.get_pose().p))):.3f}"
        )
        return True

    def _release_bottle_weld(self) -> None:
        self._bottle_welded = False
        self._bottle_weld_offset = None
        self._grasp_ee_quat = None
        if self._bottle_rigid is not None:
            try:
                self._bottle_rigid.set_kinematic(False)
                self._bottle_rigid.set_disable_gravity(False)
            except Exception:
                pass

    def _sync_welded_bottle(self) -> None:
        if not self._bottle_welded or self._bottle_weld_offset is None:
            return
        pose = self._ee_pose(self.arm) * self._bottle_weld_offset
        self._set_bottle_pose(pose)

    def _is_bottle_held(self) -> bool:
        return (not self._bottle_dropped) and self._bottle_welded

    def _bottle_axis(self) -> np.ndarray:
        R = self.bottle.get_pose().to_transformation_matrix()[:3, :3]
        axis = np.asarray(R[:, 1], dtype=float)
        return axis / (float(np.linalg.norm(axis)) + 1e-9)

    def _is_bottle_tilted(self) -> bool:
        return float(self._bottle_axis()[2]) < float(self.POUR_TILT_DOT)

    def _mouth_world(self) -> np.ndarray:
        self._sync_welded_bottle()
        try:
            fp = self.bottle.get_functional_point(0, "pose")
            return np.asarray(fp.p if hasattr(fp, "p") else fp[:3], dtype=float)
        except Exception:
            p = np.asarray(self.bottle.get_pose().p, dtype=float)
            return p + self._bottle_axis() * self.BOTTLE_HEIGHT

    def _mouth_over_cup(self, tol: float | None = None) -> bool:
        tol = self.MOUTH_POUR_XY_TOL if tol is None else float(tol)
        return float(np.linalg.norm(self._mouth_world()[:2] - self._cup_center_xy())) < tol

    def _can_pour(self) -> bool:
        if self._bottle_dropped or self.bottle_beer <= 1e-6:
            return False
        if not self._is_bottle_held() or not self._is_bottle_tilted():
            return False
        return self._mouth_over_cup()

    def _walk_tcp_to_contact(self, arm: ArmTag, tol: float | None = None) -> float:
        """Move the hand to the bottle — bottle stays on the table (no teleport)."""
        tol = self.GRASP_TCP_TOL if tol is None else float(tol)
        for _ in range(16):
            if self._bottle_spawn_pose is not None:
                self._freeze_bottle(self._bottle_spawn_pose)
            contact = self._contact_world(1)
            tcp = self._tcp_pos(arm)
            delta = contact - tcp
            err = float(np.linalg.norm(delta))
            if err < tol:
                break
            self._move_ok(
                dx=float(np.clip(delta[0], -0.04, 0.04)),
                dy=float(np.clip(delta[1], -0.04, 0.04)),
                dz=float(np.clip(delta[2], -0.04, 0.04)),
            )
        if self._bottle_spawn_pose is not None:
            self._freeze_bottle(self._bottle_spawn_pose)
        err = float(np.linalg.norm(self._tcp_pos(arm) - self._contact_world(1)))
        print(f"[pour_beer] walk tcp-contact={err:.3f}")
        return err

    # ------------------------------------------------------------------ dynamics
    def _step_fluids(self):
        if self.pouring and self._can_pour():
            d = min(self.pour_rate, self.bottle_beer)
            self.bottle_beer = max(0.0, self.bottle_beer - d)
            self.liquid_level = min(1.0, self.liquid_level + d)
            self.foam_level = min(1.2, self.foam_level + d * self.foam_gain)
        else:
            if self.pouring and not self._is_bottle_held():
                self.pouring = False
                self._bottle_dropped = True
                print("[pour_beer] bottle dropped — pouring stopped")
            if self.foam_level > 1e-6:
                self.foam_level = max(0.0, self.foam_level - self.foam_decay)
        if self._total_fill() >= self.overflow_level + 1e-4:
            self.overflowed = True
        self._rebuild_fluids(force=False)

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        # Keep the bottle planted on the table while the hand approaches —
        # never let it tip/fall and later "teleport" into the gripper.
        if self._pin_bottle_spawn and self._bottle_spawn_pose is not None and not self._bottle_welded:
            self._freeze_bottle(self._bottle_spawn_pose)
        self._sync_welded_bottle()
        self._step_fluids()

    def _idle_steps(self, n_steps: int):
        save_freq = self.save_freq if self.save_freq is not None else 15
        for i in range(int(n_steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.render_freq and i % max(1, int(self.render_freq)) == 0:
                self._update_render()
                if hasattr(self, "viewer") and self.viewer is not None:
                    self.viewer.render()
            if self.save_freq is not None and i % save_freq == 0:
                self._take_picture()

    # ------------------------------------------------------------------ motion
    def _move_world(self, dx=0.0, dy=0.0, dz=0.0, quat=None):
        return self.move(
            self.move_by_displacement(self.arm, x=float(dx), y=float(dy), z=float(dz), quat=quat)
        )

    def _move_ok(self, dx=0.0, dy=0.0, dz=0.0, quat=None) -> bool:
        # A prior failed plan leaves plan_success=False, which makes later
        # move() calls no-op — always clear it before an intentional retry.
        self.plan_success = True
        self._move_world(dx=dx, dy=dy, dz=dz, quat=quat)
        ok = bool(self.plan_success)
        if not ok:
            self.plan_success = True
        self._sync_welded_bottle()
        return ok

    def _slide_mouth_to_cup(self, z_clear: float = 0.03, tol: float = 0.04) -> float:
        """Translate EE (bottle welded) so the mouth sits over the glass."""
        err = 1.0
        cup = self._cup_center_xy()
        for _ in range(12):
            if not self._is_bottle_held():
                break
            mouth = self._mouth_world()
            dx, dy = float(cup[0] - mouth[0]), float(cup[1] - mouth[1])
            dz = float(self.cup_rim_z + z_clear - mouth[2])
            err = float(np.hypot(dx, dy))
            if err < tol and abs(dz) < 0.025:
                break
            step = 0.06 if err > 0.06 else 0.035
            self._move_ok(
                dx=float(np.clip(dx, -step, step)),
                dy=float(np.clip(dy, -step, step)),
                dz=float(np.clip(dz, -0.04, 0.04)),
            )
        err = float(np.linalg.norm(self._mouth_world()[:2] - cup))
        print(
            f"[pour_beer] mouth_err={err:.3f} tilt_z={self._bottle_axis()[2]:.2f} "
            f"held={self._is_bottle_held()}"
        )
        return err

    def _desired_tip_bottle_pose(self, frac: float) -> sapien.Pose:
        """Bottle pose tipped by ``frac`` around the current TCP (contact stays put)."""
        frac = float(np.clip(frac, 0.0, 1.0))
        ang = frac * 0.48 * np.pi  # up to ~86°
        sign = -1.0 if str(self.arm) == "right" else 1.0
        tip_q = t3d.quaternions.qmult(
            t3d.euler.euler2quat(0.0, sign * ang, 0.0),
            np.array(self.BOTTLE_UPRIGHT_Q, dtype=float),
        )
        axis = t3d.quaternions.quat2mat(tip_q)[:, 1]
        axis = axis / (float(np.linalg.norm(axis)) + 1e-9)
        tcp = self._tcp_pos(self.arm)
        origin = tcp - axis * self.CONTACT_ALONG
        return sapien.Pose(origin.tolist(), tip_q.tolist())

    def _tip_bottle_in_hand(self, frac: float) -> None:
        """Tip bottle about TCP, then best-effort match the wrist (empty_bag style).

        Absolute tipped EE IK often hangs CuRobo, so we:
          1) set the bottle tip kinematically about the TCP (stays in the fingers)
          2) ask for a small EE quat tip via displacement (ignore failures)
          3) re-assert the bottle tip / weld after the wrist move
        """
        if not self._bottle_welded:
            return
        bottle_pose = self._desired_tip_bottle_pose(frac)
        self._set_bottle_pose(bottle_pose)
        self._bottle_weld_offset = self._ee_pose(self.arm).inv() * bottle_pose

        # Best-effort wrist tip from the upright grasp quat.
        if self._grasp_ee_quat is not None:
            ang = float(np.clip(frac, 0.0, 1.0)) * 0.48 * np.pi
            sign = -1.0 if str(self.arm) == "right" else 1.0
            pour_q = t3d.quaternions.qmult(
                t3d.euler.euler2quat(0.0, sign * ang, 0.0),
                self._grasp_ee_quat,
            )
            self.plan_success = True
            self.move(
                self.move_by_displacement(
                    self.arm, x=0.0, y=0.0, z=0.0, quat=pour_q.tolist(), move_axis="world"
                )
            )
            self.plan_success = True

        # Re-assert tip — EE motion / sync may have drifted the bottle.
        bottle_pose = self._desired_tip_bottle_pose(frac)
        self._set_bottle_pose(bottle_pose)
        self._bottle_weld_offset = self._ee_pose(self.arm).inv() * bottle_pose
        self._idle_steps(1)

    def _pour_burst(self) -> bool:
        if not self._is_bottle_held():
            return False

        # Upright carry → mouth above glass → tip in fingers → re-center → pour → untilt.
        self._slide_mouth_to_cup(z_clear=0.08, tol=0.035)
        # ~60–75° is enough to pour; avoid full 90° wrist tips (often unreachable).
        for frac in (0.55, 0.85):
            self._tip_bottle_in_hand(frac)
            self._slide_mouth_to_cup(z_clear=0.02, tol=0.05)
        if not self._is_bottle_tilted():
            print("[pour_beer] tip failed")
            self._tip_bottle_in_hand(0.0)
            return False
        # Align to the same tolerance the pour gate uses (not a looser pre-check).
        self._slide_mouth_to_cup(z_clear=0.015, tol=self.MOUTH_POUR_XY_TOL * 0.7)
        if not self._can_pour():
            print("[pour_beer] tip/align failed")
            self._tip_bottle_in_hand(0.0)
            return False

        self.pouring = True
        peak = 0.0
        flowed = False
        for _ in range(70):
            if not self._can_pour() or self.overflowed or self.bottle_beer <= 1e-6:
                break
            headroom = self.overflow_level - self.liquid_level
            foam_cap = min(0.26, max(0.08, headroom - 0.08))
            if self.foam_level >= foam_cap or self.liquid_level >= self.target_liquid + 0.02:
                break
            prev = self.liquid_level
            self._idle_steps(1)
            flowed = flowed or (self.liquid_level > prev + 1e-6)
            peak = max(peak, float(self.foam_level))
        self.pouring = False

        for frac in (0.5, 0.0):
            self._tip_bottle_in_hand(frac)
        self._slide_mouth_to_cup(z_clear=0.08, tol=0.05)
        for _ in range(35):
            if self.foam_level <= self.expert_foam_resume or self.overflowed:
                break
            self._idle_steps(1)
        print(
            f"[pour_beer] burst flowed={flowed} peak_foam={peak:.2f} "
            f"liq={self.liquid_level:.2f} bottle={self.bottle_beer:.2f}"
        )
        return flowed

    def _grasp_and_weld(self, arm: ArmTag) -> bool:
        """Approach a planted bottle, close on it, weld — never teleport the bottle."""
        spawn = self._bottle_spawn_pose
        if spawn is None:
            spawn = self.bottle.get_pose()
            self._bottle_spawn_pose = spawn

        self._ignore_bottle_robot_collision()
        self._freeze_bottle(spawn)
        # Pin every sim step so grasp collisions cannot knock it over on camera.
        self._pin_bottle_spawn = True

        self.move(self.open_gripper(arm))
        self.plan_success = True
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
                contact_point_id=[1, 0, 2],
            )
        )
        if not self.plan_success:
            print("[pour_beer] grasp plan failed")
            self._pin_bottle_spawn = False
            return False

        # Hand walks to the bottle; bottle stays at spawn (no snap-into-gripper).
        self._freeze_bottle(spawn)
        err = self._walk_tcp_to_contact(arm, tol=self.GRASP_TCP_TOL)
        if err > 0.055:
            print(f"[pour_beer] grasp seating failed (tcp-contact={err:.3f})")
            self._pin_bottle_spawn = False
            return False

        self.move(self.close_gripper(arm, pos=0.0))
        self._idle_steps(3)
        self._freeze_bottle(spawn)  # still on the table under the closed fingers
        ok = self._weld_bottle_to_ee(arm)
        self._pin_bottle_spawn = False
        return ok

    def play_once(self):
        arm = self.arm
        self.plan_success = True

        if not self._grasp_and_weld(arm):
            self.plan_success = False
            return self.info

        self.move(self.move_by_displacement(arm, z=0.12, move_axis="arm"))
        if not self.plan_success:
            print("[pour_beer] lift failed")
            self._release_bottle_weld()
            return self.info
        self._sync_welded_bottle()

        # Carry toward the glass (translations only — no unreachable pour quats).
        bp = np.asarray(self.bottle.get_pose().p, dtype=float)
        cup = self._cup_center_xy()
        self._move_ok(
            dx=float(cup[0] - bp[0]) * 0.55,
            dy=float(cup[1] - bp[1]) * 0.70,
            dz=0.04,
        )

        for cycle in range(int(self.max_pour_cycles)):
            if self._bottle_dropped or not self._is_bottle_held():
                self.plan_success = False
                break
            if self.overflowed or self.bottle_beer <= 0.05:
                break
            if self.liquid_level >= self.target_liquid - 1e-3:
                break
            ok = self._pour_burst()
            print(
                f"[pour_beer] cycle={cycle} ok={ok} liq={self.liquid_level:.2f} "
                f"bottle={self.bottle_beer:.2f} held={self._is_bottle_held()}"
            )
            if not ok and cycle == 0:
                self.plan_success = False
                break

        self.pouring = False
        for _ in range(40):
            if self.foam_level < 0.05:
                break
            self._idle_steps(1)

        if self._is_bottle_held():
            self._tip_bottle_in_hand(0.0)
            bp = np.asarray(self.bottle.get_pose().p, dtype=float)
            self._move_ok(
                dx=float(self.bottle_xy[0] - bp[0]),
                dy=float(self.bottle_xy[1] - bp[1]),
                dz=0.02,
            )
            self._move_ok(dz=-0.10)
            self._release_bottle_weld()
            self.move(self.open_gripper(arm))
        else:
            self._release_bottle_weld()

        if self.check_success() and not self._bottle_dropped:
            self.plan_success = True

        self.info["info"] = {
            "{A}": f"{self.BOTTLE_MODEL}/base{self.bottle_id}",
            "{B}": f"{self.GLASS_MODEL}/base0",
            "{a}": str(arm),
        }
        return self.info

    def check_success(self):
        if self.overflowed or self._bottle_dropped:
            return False
        liquid_ok = self.liquid_level >= self.target_liquid - self.full_liquid_tol
        bottle_empty = self.bottle_beer <= 0.20
        foam_ok = self._total_fill() < self.overflow_level - 0.02
        return bool(liquid_ok and bottle_empty and foam_ok)

    def get_obs(self):
        obs = super().get_obs()
        obs["beer_pour"] = {
            "liquid_level": float(self.liquid_level),
            "foam_level": float(self.foam_level),
            "total_fill": float(self._total_fill()),
            "bottle_beer": float(self.bottle_beer),
            "pouring": bool(self.pouring),
            "overflowed": bool(self.overflowed),
            "held": bool(self._is_bottle_held()),
            "scene_id": int(getattr(self, "scene_id", 0)),
        }
        return obs
