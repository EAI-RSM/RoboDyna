"""Pour beer from a bottle into a glass, pausing so foam can settle (KitchenS).

Expert flow (right-arm station):
  1) mid-body side-grasp — jaws close around the bottle; weld follows the hand
  2) lift, then carry upright to the RIGHT of the glass with clearance
  3) wrist-tip LEFT so the mouth arcs over the glass opening
  4) liquid flows only while held + tipped + mouth-over-glass
  5) untilt, open gripper, seat bottle back on its coaster
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

    BOTTLE_BEER_FRAC = 1.0
    FULL_LIQUID_TOL = 0.05
    TARGET_LIQUID = 0.95  # fill to ~95% of glass height (−5% from full)

    POUR_RATE = 0.022
    FOAM_GAIN = 0.85
    FOAM_DECAY = 0.0045
    OVERFLOW_LEVEL = 1.0
    EXPERT_FOAM_RESUME = 0.14
    MAX_POUR_CYCLES = 14

    BEER_COLOR = [0.78, 0.52, 0.10, 0.78]
    FOAM_COLOR = [0.97, 0.95, 0.90, 0.88]
    STAIN_COLOR = [0.95, 0.82, 0.10, 0.95]
    GLASS_RGBA = [0.82, 0.93, 0.98, 0.22]
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]
    BOTTLE_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]
    GLASS_UPRIGHT_Q = [0.70710678, 0.70710678, 0.0, 0.0]

    # WSG jaw gap (m) → normalized gripper command (from empty_bag calibration).
    JAW_GAP_TABLE = ((0.006, 0.0), (0.0182, 0.25), (0.0532, 0.5), (0.0882, 0.75), (0.110, 1.0))

    # Bottle mesh long-axis is local +Y; upright → world +Z.
    BOTTLE_HEIGHT = 0.220
    CONTACT_ALONG = 0.124  # mid-body side grasp along local Y, scaled
    BOTTLE_RADIUS = 0.0325
    POUR_TILT_DOT = 0.70  # axis·ẑ below this counts as tipped (≈45° from upright)
    TIP_MAX_RAD = 0.30 * np.pi  # ~54° max tip via small EE steps (no 360° unwind)
    MOUTH_POUR_XY_TOL = 0.050
    GRASP_TCP_TOL = 0.035
    EE_TO_TCP = 0.12
    # Upright hold: bottle clearly to the right of / above the glass (ref: pour_beer.webp).
    POUR_HOLD_DX = 0.20
    POUR_HOLD_Z_ABOVE_RIM = 0.14
    # Mouth above the lip with a visible gap — not jammed onto the rim.
    MOUTH_CLEAR_Z = 0.040
    # Aim mouth at the right half of the opening so the bottle body stays farther away.
    MOUTH_AIM_DX = 0.018

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
        self._stain_entity = None
        self._liquid_half_h_cached = -1.0
        self._foam_half_h_cached = -1.0
        self.cup = None
        self.bottle = None
        self.table_top = 0.74
        self._bottle_rigid = None
        self._bottle_welded = False
        self._bottle_weld_offset = None
        self._grasp_ee_quat = None
        self._tip_frac = 0.0
        self._pour_tip_axis_fixed = None
        self._pour_to_cup_fixed = None
        self._bottle_dropped = False
        self._bottle_spawn_pose = None
        self._pin_bottle_spawn = False  # lock bottle on table during approach
        self._bar_props = []
        self._grasp_gripper_pos = 0.55
        self.glass_coaster = None
        self.bottle_coaster = None
        self.coaster_top_z = 0.75

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
        self._stain_entity = self._remove_entity(getattr(self, "_stain_entity", None))
        self._liquid_half_h_cached = -1.0
        self._foam_half_h_cached = -1.0
        self._bottle_rigid = None
        self._bottle_welded = False
        self._bottle_weld_offset = None
        self._grasp_ee_quat = None
        self._tip_frac = 0.0
        self._pour_tip_axis_fixed = None
        self._pour_to_cup_fixed = None
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
        self.glass_coaster = None
        self.bottle_coaster = None
        self.coaster_top_z = self.table_top + 0.008
        self._build_bar_props()
        self._spawn_station_coasters()
        self._spawn_glass()
        self._spawn_bottle()
        self._rebuild_fluids(force=True)
        self.add_prohibit_area(self.cup, padding=0.04)
        self.add_prohibit_area(self.bottle, padding=0.04)
        self._loaded = True
        print(
            f"[pour_beer] scene={self.scene_id} arm={self.arm} "
            f"cup={self.cup_xy} bottle={self.bottle_xy} "
            f"coaster_z={self.coaster_top_z:.3f} bar_props={len(self._bar_props)}"
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
        # Decor only — skip prohibit pads so they don't crowd the pour grasp.
        return actor

    def _build_bar_props(self):
        """Sparse bar décor (~10 props), spaced so nothing overlaps the pour station."""
        back_y = 0.24
        # Four bottles across the back, ≥14 cm centers.
        for model, mid, x, yaw in [
            ("255_beer_bottle", 0, -0.50, -10),
            ("001_bottle", 2, -0.32, 8),
            ("001_bottle", 5, 0.34, -8),
            ("255_beer_bottle", 0, 0.52, 12),
        ]:
            self._spawn_static_prop(model, [x, back_y], model_id=mid, yaw_deg=yaw)

        # One glass + one mug in the mid gap (wineglass mesh is huge → shrink).
        self._spawn_static_prop(
            "088_wineglass", [0.00, back_y - 0.02], model_id=0, yaw_deg=15, scale_mult=0.38
        )
        self._spawn_static_prop(
            "039_mug", [0.16, back_y - 0.02], model_id=0, yaw_deg=30, scale_mult=0.65
        )

        # Side snacks only — pour station (x≈0.10, y≈-0.08..0.04) stays clear.
        self._spawn_static_prop("025_chips-tub", [-0.46, -0.10], model_id=0, yaw_deg=-20)
        self._spawn_static_prop("025_chips-tub", [0.50, -0.12], model_id=2, yaw_deg=30)
        self._spawn_static_prop("071_can", [-0.50, -0.22], model_id=0, yaw_deg=-35)
        self._spawn_static_prop("054_baguette", [-0.52, 0.14], model_id=2, yaw_deg=65)

    def _spawn_station_coasters(self):
        """One coaster under the glass, one under the bottle (home spot)."""
        # Coaster mesh is thin (~8 mm). Glass/bottle sit on its top face.
        self.glass_coaster = self._spawn_static_prop(
            "019_coaster", self.cup_xy, model_id=0, yaw_deg=0.0, z_off=0.001
        )
        self.bottle_coaster = self._spawn_static_prop(
            "019_coaster", self.bottle_xy, model_id=0, yaw_deg=15.0, z_off=0.001
        )
        if self.glass_coaster is not None:
            self.glass_coaster.set_name("glass_coaster")
        if self.bottle_coaster is not None:
            self.bottle_coaster.set_name("bottle_coaster")

        # Prefer authored height; fall back to a thin disc.
        top_z = self.table_top + 0.008
        for coaster in (self.glass_coaster, self.bottle_coaster):
            if coaster is None:
                continue
            cfg = getattr(coaster, "config", {}) or {}
            ext = np.array(cfg.get("extents", [0.15, 0.008, 0.15]), dtype=float)
            sc = cfg.get("scale", [1, 1, 1])
            sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
            # Local Y is up when upright.
            top_z = max(top_z, self.table_top + float(ext[1] * sc) + 0.001)
        self.coaster_top_z = float(top_z)

    def _spawn_glass(self):
        pose = sapien.Pose(
            [float(self.cup_xy[0]), float(self.cup_xy[1]), self.coaster_top_z + 0.001],
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
        center = np.array(cfg.get("center", [0.0, 0.0, 0.0]), dtype=float)
        world = ext * sc
        self.cup_height = float(world[1])
        self.cup_inner_r = 0.30 * float(max(world[0], world[2]))

        # Mesh AABB / authored center+extents (local Y → world Z when upright).
        # Previous fillable rim sat ~2.5 cm above the glass top — beer looked spilled.
        pose_z = float(self.cup.get_pose().p[2])
        half_y = 0.5 * float(ext[1]) * sc
        cy = float(center[1]) * sc
        mesh_bottom_z = pose_z + cy - half_y
        mesh_top_z = pose_z + cy + half_y
        try:
            for c in self.cup.actor.get_components():
                if hasattr(c, "get_global_aabb_fast"):
                    aabb = np.asarray(c.get_global_aabb_fast(), dtype=float)
                    mesh_bottom_z = float(aabb[0, 2])
                    mesh_top_z = float(aabb[1, 2])
                    break
                if hasattr(c, "get_global_aabb"):
                    aabb = np.asarray(c.get_global_aabb(), dtype=float)
                    mesh_bottom_z = float(aabb[0, 2])
                    mesh_top_z = float(aabb[1, 2])
                    break
        except Exception:
            pass
        # Inner floor / lip clearance — beer+foam must stay inside the glass mesh.
        self.cup_bottom_z = mesh_bottom_z + 0.012
        self.cup_rim_z = mesh_top_z - 0.008
        self.cup_fillable_h = max(0.05, float(self.cup_rim_z - self.cup_bottom_z))
        print(
            f"[pour_beer] glass mesh z=[{mesh_bottom_z:.3f},{mesh_top_z:.3f}] "
            f"fill z=[{self.cup_bottom_z:.3f},{self.cup_rim_z:.3f}] h={self.cup_fillable_h:.3f}"
        )

    def _spawn_bottle(self):
        pose = sapien.Pose(
            [float(self.bottle_xy[0]), float(self.bottle_xy[1]), self.coaster_top_z + 0.002],
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

        # Mid-body grasp height + jaw gap from the authored mesh extents.
        cfg = getattr(self.bottle, "config", {}) or {}
        ext = np.array(cfg.get("extents", [0.45, 1.56, 0.45]), dtype=float)
        sc = cfg.get("scale", [1, 1, 1])
        sc = float(sc[0] if isinstance(sc, (list, tuple)) else sc)
        center = np.array(cfg.get("center", [0.0, 0.78, 0.0]), dtype=float)
        world = ext * sc
        self.BOTTLE_RADIUS = 0.48 * float(max(world[0], world[2]))
        # Prefer authored contact height (mid-body); fall back to mesh center.
        try:
            cpose = self.bottle.get_contact_point(1, "pose")
            cp = np.asarray(cpose.p if hasattr(cpose, "p") else cpose[:3], dtype=float)
            origin = np.asarray(self.bottle.get_pose().p, dtype=float)
            self.CONTACT_ALONG = float(np.dot(cp - origin, self._bottle_axis()))
        except Exception:
            self.CONTACT_ALONG = float(center[1] * sc)
        try:
            mouth = self.bottle.get_functional_point(0, "pose")
            mp = np.asarray(mouth.p if hasattr(mouth, "p") else mouth[:3], dtype=float)
            origin = np.asarray(self.bottle.get_pose().p, dtype=float)
            self.BOTTLE_HEIGHT = float(np.linalg.norm(mp - origin))
        except Exception:
            self.BOTTLE_HEIGHT = float(world[1] * 0.98)
        # Jaw command that closes around the body (not through it).
        # Calibrated WSG gap is a bit optimistic vs the visual mesh — leave slack.
        self._grasp_gripper_pos = self._gripper_pos_for_gap(2.0 * self.BOTTLE_RADIUS + 0.022)
        print(
            f"[pour_beer] bottle r={self.BOTTLE_RADIUS:.3f} mid={self.CONTACT_ALONG:.3f} "
            f"h={self.BOTTLE_HEIGHT:.3f} grip_pos={self._grasp_gripper_pos:.2f}"
        )

    @classmethod
    def _gripper_pos_for_gap(cls, gap: float) -> float:
        gaps = [g for g, _ in cls.JAW_GAP_TABLE]
        cmds = [p for _, p in cls.JAW_GAP_TABLE]
        return float(np.clip(np.interp(float(gap), gaps, cmds), 0.0, 1.0))

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

    def _clamped_fill_fracs(self):
        """Beer + foam fractions that never render above the glass rim."""
        liq = max(0.0, min(1.0, float(self.liquid_level)))
        foam = max(0.0, float(self.foam_level))
        # Foam only occupies remaining headroom inside the glass.
        foam = min(foam, max(0.0, 1.0 - liq))
        return liq, foam

    def _rebuild_fluids(self, force: bool = False):
        if not getattr(self, "cup_fillable_h", None):
            return
        liq_frac, foam_frac = self._clamped_fill_fracs()
        liq_h = liq_frac * self.cup_fillable_h
        foam_h = foam_frac * self.cup_fillable_h
        # Hard cap so the column top never crosses the rim.
        top = liq_h + foam_h
        if top > self.cup_fillable_h:
            scale = self.cup_fillable_h / max(top, 1e-6)
            liq_h *= scale
            foam_h *= scale
        liq_half = max(0.002, 0.5 * liq_h) if liq_frac > 1e-4 else 0.0
        foam_half = max(0.002, 0.5 * foam_h) if foam_frac > 1e-4 else 0.0
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
        if liq_frac > 1e-4:
            self._liquid_entity = self._make_column(
                liq_half, r, self.cup_bottom_z + liq_half, self.BEER_COLOR, "beer_liquid"
            )
        if foam_frac > 1e-4:
            self._foam_entity = self._make_column(
                foam_half, r * 0.98, self.cup_bottom_z + liq_h + foam_half, self.FOAM_COLOR, "beer_foam"
            )

    def _spawn_overflow_stain(self):
        """Yellow beer stain around the glass base — visible overflow failure cue."""
        if getattr(self, "_stain_entity", None) is not None:
            return
        xy = self._cup_center_xy()
        outer_r = max(0.07, float(self.cup_inner_r) * 3.2)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("static")
        builder.add_cylinder_visual(
            pose=sapien.Pose([0, 0, 0], self.VERTICAL_CYL_Q),
            radius=outer_r,
            half_length=0.0016,
            material=self._fluid_material(self.STAIN_COLOR),
        )
        # Sit just above the coaster / counter so the spill reads clearly.
        z = float(getattr(self, "coaster_top_z", self.table_top)) + 0.0035
        builder.set_initial_pose(sapien.Pose(p=[float(xy[0]), float(xy[1]), z]))
        self._stain_entity = builder.build(name="beer_overflow_stain")
        print("[pour_beer] OVERFLOW — yellow stain around glass")

    def _mark_overflow(self):
        if self.overflowed:
            return
        self.overflowed = True
        # Keep contents at the rim for the visual; stain shows the spill.
        total = self._total_fill()
        if total > 1.0:
            # Prefer keeping liquid; trim foam first.
            self.foam_level = max(0.0, 1.0 - float(self.liquid_level))
            if self.liquid_level > 1.0:
                self.liquid_level = 1.0
                self.foam_level = 0.0
        self._spawn_overflow_stain()
        self._rebuild_fluids(force=True)

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
        """Mid-body side-grasp contact (authored on the bottle body)."""
        try:
            fp = self.bottle.get_contact_point(contact_id, "pose")
            return np.asarray(fp.p if hasattr(fp, "p") else fp[:3], dtype=float)
        except Exception:
            p = np.asarray(self.bottle.get_pose().p, dtype=float)
            return p + self._bottle_axis() * float(self.CONTACT_ALONG)

    def _mouth_along(self) -> float:
        """Distance along bottle axis from grasp contact to mouth."""
        return max(0.04, float(self.BOTTLE_HEIGHT) - float(self.CONTACT_ALONG))

    def _pour_tip_dirs(self):
        """Tip toward the glass: bottle on the right → tip left (about −Y)."""
        if self._pour_tip_axis_fixed is not None and self._pour_to_cup_fixed is not None:
            return self._pour_tip_axis_fixed, self._pour_to_cup_fixed
        cup = self._cup_center_xy()
        try:
            src = np.asarray(self.bottle.get_pose().p[:2], dtype=float)
        except Exception:
            src = self._tcp_pos(self.arm)[:2]
        to_cup = np.array([cup[0] - src[0], cup[1] - src[1], 0.0], dtype=float)
        n = float(np.linalg.norm(to_cup))
        if n < 1e-3:
            # Default station: glass is to the left of a right-arm hold.
            to_cup = np.array([-1.0, 0.0, 0.0], dtype=float)
            n = 1.0
        to_cup /= n
        tip_axis = np.cross(np.array([0.0, 0.0, 1.0]), to_cup)
        tip_axis /= float(np.linalg.norm(tip_axis)) + 1e-9
        return tip_axis, to_cup

    def _lock_pour_tip_dirs(self) -> None:
        """Freeze tip axis at the upright hold so it doesn't drift mid-pour."""
        tip_axis, to_cup = self._pour_tip_dirs()
        # Force a clean left tip when the bottle is clearly to the right of the glass.
        cup = self._cup_center_xy()
        bp = np.asarray(self.bottle.get_pose().p[:2], dtype=float)
        if bp[0] > cup[0] + 0.03:
            to_cup = np.array([-1.0, 0.0, 0.0], dtype=float)
            tip_axis = np.array([0.0, -1.0, 0.0], dtype=float)
        self._pour_tip_axis_fixed = tip_axis
        self._pour_to_cup_fixed = to_cup
        print(f"[pour_beer] tip_dir to_cup={to_cup} axis={tip_axis}")

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
        if self.pouring and self._can_pour() and not self.overflowed:
            d = min(self.pour_rate, self.bottle_beer)
            self.bottle_beer = max(0.0, self.bottle_beer - d)
            # Proposed fill; any crest over the rim = overflow fail + stain.
            add_liq = d
            add_foam = d * self.foam_gain
            new_liq = float(self.liquid_level) + add_liq
            new_foam = float(self.foam_level) + add_foam
            if new_liq + new_foam >= float(self.overflow_level) - 1e-6:
                room = max(0.0, float(self.overflow_level) - self._total_fill())
                if room > 1e-6:
                    # Fill remaining headroom, then spill.
                    liq_share = add_liq / max(add_liq + add_foam, 1e-9)
                    self.liquid_level = min(1.0, self.liquid_level + room * liq_share)
                    self.foam_level = max(0.0, float(self.overflow_level) - self.liquid_level)
                self._mark_overflow()
            else:
                self.liquid_level = min(1.0, new_liq)
                self.foam_level = max(0.0, new_foam)
        else:
            if self.pouring and not self._is_bottle_held():
                self.pouring = False
                self._bottle_dropped = True
                print("[pour_beer] bottle dropped — pouring stopped")
            if (not self.overflowed) and self.foam_level > 1e-6:
                self.foam_level = max(0.0, self.foam_level - self.foam_decay)
        if (not self.overflowed) and self._total_fill() >= self.overflow_level + 1e-4:
            self._mark_overflow()
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

    def _nearest_quat(self, q, ref) -> np.ndarray:
        """Pick q or −q so the rotation stays on the short arc from ``ref``."""
        q = np.asarray(q, dtype=float)
        ref = np.asarray(ref, dtype=float)
        if float(np.dot(q, ref)) < 0.0:
            q = -q
        return q

    def _arm_tip(self, frac: float, step: float = 0.12) -> None:
        """Tip/untilt by rotating the arm in small steps; bottle follows the fixed weld.

        The grasp weld is never rewritten here — that was making the bottle tip
        while the hand stayed still (ghost motion). Absolute EE quats are also
        avoided: each step is a small relative rotation about the tip axis.
        """
        if not self._bottle_welded:
            return
        target = float(np.clip(frac, 0.0, 1.0))
        cur = float(getattr(self, "_tip_frac", 0.0))
        if abs(target - cur) < 1e-3:
            self._sync_welded_bottle()
            return
        n = max(1, int(round(abs(target - cur) / max(step, 1e-3))))
        tip_axis, to_cup = self._pour_tip_dirs()
        prev = cur
        for i in range(1, n + 1):
            f = cur + (target - cur) * (i / n)
            d_ang = (f - prev) * float(self.TIP_MAX_RAD)
            prev = f
            ee_q = np.asarray(self._ee_pose(self.arm).q, dtype=float)
            q = t3d.quaternions.qmult(
                t3d.quaternions.axangle2quat(tip_axis, d_ang),
                ee_q,
            )
            q = self._nearest_quat(q, ee_q)
            # Tiny approach only on the first tip-down from upright.
            pull = 0.010 if (target > cur and cur < 0.05 and i == 1) else 0.0
            self._move_ok(
                dx=float(to_cup[0] * pull),
                dy=float(to_cup[1] * pull),
                dz=0.0,
                quat=q.tolist(),
            )
            self._tip_frac = float(f)
            self._sync_welded_bottle()
        print(
            f"[pour_beer] tip frac={self._tip_frac:.2f} "
            f"ang_deg={self._tip_frac * np.degrees(self.TIP_MAX_RAD):.1f} "
            f"tilt_z={self._bottle_axis()[2]:.2f} mouth_over={self._mouth_over_cup()} "
            f"tcp={self._tcp_pos(self.arm)}"
        )

    def _carry_upright_to_pour_hold(self) -> None:
        """Carry upright bottle to the right of / above the glass (translation only)."""
        if not self._is_bottle_held():
            return
        cup = self._cup_center_xy()
        hold_dx = float(getattr(self, "POUR_HOLD_DX", 0.20))
        # Weld stays as grasped — bottle only moves because the arm moves.
        self._tip_frac = 0.0
        self._sync_welded_bottle()

        target_tcp_z = float(self.cup_rim_z) + float(self.POUR_HOLD_Z_ABOVE_RIM)
        target_bottle_xy = np.array([cup[0] + hold_dx, cup[1]], dtype=float)

        for _ in range(16):
            if not self._is_bottle_held():
                break
            self._sync_welded_bottle()
            bp = np.asarray(self.bottle.get_pose().p, dtype=float)
            tcp = self._tcp_pos(self.arm)
            dx = float(target_bottle_xy[0] - bp[0])
            dy = float(target_bottle_xy[1] - bp[1])
            dz = float(target_tcp_z - tcp[2])
            if abs(dx) < 0.025 and abs(dy) < 0.025 and abs(dz) < 0.025:
                break
            step = 0.06
            self._move_ok(
                dx=float(np.clip(dx, -step, step)),
                dy=float(np.clip(dy, -step, step)),
                dz=float(np.clip(dz, -0.06, 0.06)),
            )
        self._sync_welded_bottle()
        self._lock_pour_tip_dirs()
        bp = np.asarray(self.bottle.get_pose().p, dtype=float)
        tcp = self._tcp_pos(self.arm)
        print(
            f"[pour_beer] pour_hold bottle={bp} cup={cup} "
            f"right_clear={bp[0] - cup[0]:.3f} tcp_z={tcp[2]:.3f} "
            f"above_rim={tcp[2] - self.cup_rim_z:.3f}"
        )

    def _nudge_mouth_over_cup(self, tol: float = 0.030) -> float:
        """Translate the arm so the welded mouth hangs over the glass (no bottle teleport)."""
        cup = self._cup_center_xy()
        aim = cup + np.array([float(getattr(self, "MOUTH_AIM_DX", 0.018)), 0.0], dtype=float)
        clear = float(getattr(self, "MOUTH_CLEAR_Z", 0.040))
        target_z = float(self.cup_rim_z) + clear
        tip = float(getattr(self, "_tip_frac", 1.0))
        err = 1.0
        for _ in range(10):
            if not self._is_bottle_held():
                break
            self._sync_welded_bottle()
            mouth = self._mouth_world()
            dx, dy = float(aim[0] - mouth[0]), float(aim[1] - mouth[1])
            dz = float(target_z - mouth[2])
            err = float(np.hypot(dx, dy))
            if err < tol and abs(dz) < 0.03:
                break
            self._move_ok(
                dx=float(np.clip(dx, -0.03, 0.03)),
                dy=float(np.clip(dy, -0.03, 0.03)),
                dz=float(np.clip(dz, -0.02, 0.04)),
            )
        self._sync_welded_bottle()
        mouth = self._mouth_world()
        err = float(np.linalg.norm(mouth[:2] - cup))
        tcp = self._tcp_pos(self.arm)
        print(
            f"[pour_beer] mouth_err={err:.3f} mouth_z={mouth[2]:.3f} rim_z={self.cup_rim_z:.3f} "
            f"tcp_above={tcp[2] - self.cup_rim_z:.3f} "
            f"tilt_z={self._bottle_axis()[2]:.2f} ang_deg={tip * np.degrees(self.TIP_MAX_RAD):.1f} "
            f"held={self._is_bottle_held()}"
        )
        return err

    def _pour_burst(self) -> bool:
        """Assume already tipped over the glass; open the pour gate for one foam cycle."""
        if not self._is_bottle_held():
            return False
        if not self._is_bottle_tilted():
            self._arm_tip(1.0)
            self._nudge_mouth_over_cup()
        if not self._can_pour():
            self._arm_tip(1.0)
            self._nudge_mouth_over_cup(tol=0.04)
        if not self._can_pour():
            print(
                f"[pour_beer] tip/align failed tilt_z={self._bottle_axis()[2]:.2f} "
                f"mouth_over={self._mouth_over_cup()} held={self._is_bottle_held()}"
            )
            return False

        self.pouring = True
        peak = 0.0
        flowed = False
        for _ in range(55):
            if not self._can_pour() or self.overflowed or self.bottle_beer <= 1e-6:
                break
            headroom = float(self.overflow_level) - self._total_fill()
            # Keep foam low near the rim so liquid can reach ~95% without overflow.
            foam_cap = min(0.12, max(0.03, headroom - 0.08))
            if self.foam_level >= foam_cap or self.liquid_level >= self.target_liquid - 0.01:
                break
            if headroom < 0.05:
                break
            prev = self.liquid_level
            self._idle_steps(1)
            flowed = flowed or (self.liquid_level > prev + 1e-6)
            peak = max(peak, float(self.foam_level))
        self.pouring = False

        # Ease upright a bit so foam can settle, then re-tip next cycle.
        self._arm_tip(0.35)
        for _ in range(40):
            if self.foam_level <= self.expert_foam_resume or self.overflowed:
                break
            self._idle_steps(1)
        print(
            f"[pour_beer] burst flowed={flowed} peak_foam={peak:.2f} "
            f"liq={self.liquid_level:.2f} bottle={self.bottle_beer:.2f} "
            f"overflow={self.overflowed}"
        )
        return flowed and (not self.overflowed)

    def _grasp_and_weld(self, arm: ArmTag) -> bool:
        """Mid-body grasp with jaws closed around the bottle, then weld."""
        spawn = self._bottle_spawn_pose
        if spawn is None:
            spawn = self.bottle.get_pose()
            self._bottle_spawn_pose = spawn

        grip_pos = float(getattr(self, "_grasp_gripper_pos", 0.55))
        self._ignore_bottle_robot_collision()
        self._freeze_bottle(spawn)
        self._pin_bottle_spawn = True

        self.move(self.open_gripper(arm, pos=1.0))
        self.plan_success = True
        self.move(
            self.grasp_actor(
                self.bottle,
                arm_tag=arm,
                pre_grasp_dis=0.10,
                grasp_dis=0.0,
                gripper_pos=grip_pos,
                contact_point_id=[1, 0, 2],
            )
        )
        if not self.plan_success:
            print("[pour_beer] grasp plan failed")
            self._pin_bottle_spawn = False
            return False

        # Seat TCP on the mid-body contact; bottle stays planted.
        self._freeze_bottle(spawn)
        err = self._walk_tcp_to_contact(arm, tol=self.GRASP_TCP_TOL)
        if err > 0.055:
            print(f"[pour_beer] grasp seating failed (tcp-contact={err:.3f})")
            self._pin_bottle_spawn = False
            return False

        # Nudge a bit down the bottle axis toward the label / true mid-body.
        self._freeze_bottle(spawn)
        self._move_ok(dz=-0.018)
        self._freeze_bottle(spawn)

        # Close around the body — never drive jaws to 0 (would pierce the mesh).
        self.move(self.close_gripper(arm, pos=grip_pos))
        self._idle_steps(4)
        self._freeze_bottle(spawn)
        ok = self._weld_bottle_to_ee(arm)
        # Weld pivot for tip should match where the fingers actually sat.
        try:
            tcp = self._tcp_pos(arm)
            origin = np.asarray(self.bottle.get_pose().p, dtype=float)
            self.CONTACT_ALONG = float(np.dot(tcp - origin, self._bottle_axis()))
        except Exception:
            pass
        self._pin_bottle_spawn = False
        return ok

    def _place_bottle_on_coaster(self) -> None:
        """Seat the bottle back on its coaster (kinematic — no CuRobo place path)."""
        home = self._bottle_spawn_pose
        if home is None:
            home = sapien.Pose(
                [float(self.bottle_xy[0]), float(self.bottle_xy[1]), self.coaster_top_z + 0.002],
                self.BOTTLE_UPRIGHT_Q,
            )
        # Planner place-back often hangs after the pour; snap home after release.
        self._release_bottle_weld()
        self._freeze_bottle(home)
        self._set_bottle_pose(home)
        self._tip_frac = 0.0
        self._idle_steps(2)
        print(
            f"[pour_beer] place_on_coaster bottle={self.bottle.get_pose().p} "
            f"home={home.p}"
        )

    def play_once(self):
        arm = self.arm
        self.plan_success = True

        if not self._grasp_and_weld(arm):
            self.plan_success = False
            return self.info

        # World-frame lift — arm axis "z" was sliding the bottle sideways.
        self.plan_success = True
        self.move(self.move_by_displacement(arm, z=0.12, move_axis="world"))
        if not self.plan_success:
            print("[pour_beer] lift failed")
            self.plan_success = True
            self._move_ok(dz=0.08)
        self._sync_welded_bottle()

        # Hold upright to the RIGHT of the glass, then tip LEFT over the rim.
        self._carry_upright_to_pour_hold()
        self._arm_tip(1.0)
        self._nudge_mouth_over_cup(tol=self.MOUTH_POUR_XY_TOL * 0.7)

        for cycle in range(int(self.max_pour_cycles)):
            if self._bottle_dropped or not self._is_bottle_held():
                self.plan_success = False
                break
            if self.overflowed or self.bottle_beer <= 0.05:
                break
            # Stop with a little headroom so foam can't crest the rim.
            if self.liquid_level >= self.target_liquid - 0.02:
                break
            if cycle > 0:
                self._arm_tip(1.0)
                self._nudge_mouth_over_cup()
            ok = self._pour_burst()
            print(
                f"[pour_beer] cycle={cycle} ok={ok} liq={self.liquid_level:.2f} "
                f"bottle={self.bottle_beer:.2f} held={self._is_bottle_held()} "
                f"overflow={self.overflowed}"
            )
            if self.overflowed:
                break
            if not ok and cycle == 0:
                self.plan_success = False
                break

        self.pouring = False
        for _ in range(40):
            if self.foam_level < 0.05 or self.overflowed:
                break
            self._idle_steps(1)

        if self._is_bottle_held():
            self._arm_tip(0.0)
            self.move(self.open_gripper(arm))
            self._place_bottle_on_coaster()
        else:
            self._release_bottle_weld()
            if self._bottle_spawn_pose is not None:
                self._set_bottle_pose(self._bottle_spawn_pose)

        if self.overflowed:
            self.plan_success = False
        elif self.check_success() and not self._bottle_dropped:
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
