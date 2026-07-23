from ._base_task import Base_Task
from .utils import *
from .utils.actor_utils import Actor
import os
import sapien
import sapien.render
import sapien.physx
import numpy as np


class pick_ripe_apple(Base_Task):
    """Pick a ripening apple hanging from a tree into a basket.

    A simple decorative tree sits at the table center (x=0, toward the back). One apple hangs from a
    left or right branch and ripens green -> red -> black while attached. The matching arm waits for
    red, grasps the apple (which then detaches from the tree and freezes its color), and drops it
    into a breadbasket in front of the tree.

    Metric: ripeness_score = clamp(1 - |r_grasp - red_window| / 0.5, 0, 1), latched at detach;
    episode succeeds only if the apple comes to rest in the basket.
    """

    # ---- task params (class defaults; override via task_args.pick_ripe_apple) ----
    RIPEN_STEPS_DEFAULT = 2500
    RED_WINDOW_MIN_DEFAULT = 0.48
    RED_WINDOW_MAX_DEFAULT = 0.52
    RED_TOLERANCE_DEFAULT = 0.12
    GRASP_LEAD_STEPS = 700            # lead so r_grasp lands near red (~0.5) after grasp motion
    # Table y spans ~[-0.35, +0.35]; the back wall is at y=1. Default tree sits near the back edge.
    TREE_X_DEFAULT = 0.0
    TREE_Y_DEFAULT = 0.05             # toward back wall (wall at y=1; table ~±0.35)
    BASKET_Y_DEFAULT = -0.20          # in front of the tree — clearance so the grasp misses the rim
    BASKET_X_DEFAULT = 0.0
    BASKET_SCALE_DEFAULT = 1.0
    APPLE_HANG_DX_DEFAULT = 0.18      # |x| of hanging apple (past longer branch tip)
    APPLE_SCALE_DEFAULT = 0.78
    # Branch heights (m above table). One side is high, the other low (randomized).
    BRANCH_Z_HIGH = 0.25              # was 0.15; +10 cm
    BRANCH_Z_LOW = 0.20               # was 0.10; +10 cm
    TRUNK_HEIGHT = 0.50               # was 0.30; +20 cm
    # Visual top (texture pole / local +Z) faces up under the branch stem (identity quat).
    # Mesh origin is the local-Y tip; bbox center is ~0.0374·scale along +Y — offset hang
    # so the sphere center (not the tip) sits on the branch/stem Y.
    # Partial Y offset: full model_data center overshoots; ~half centers the branch on the fruit.
    APPLE_MESH_CENTER_Y = 0.0374355
    APPLE_Y_CENTER_FRAC = 0.45
    APPLE_DROP_BELOW_BRANCH = 0.042   # lower so ~½ of the black stem shows below the branch
    STEM_HALF_THICK = 0.002           # was 0.004; diameter halved → 4 mm
    # Left-shift stem+apple by the pre-halving stem diameter so the inset is obvious.
    STEM_INSET_X = 0.008
    # Hang X is sampled in [hang_x_hi - APPLE_X_JITTER, hang_x_hi] (hi = tip inset).
    APPLE_X_JITTER = 0.02            # m; lower bound = current tip position − 2 cm
    APPLE_HANG_Q = [1.0, 0.0, 0.0, 0.0]
    # Packing upright [0.5,0.5,0.5,0.5] then +90° about world Z.
    BASKET_Q = [0.0, 0.0, 0.70710678, 0.70710678]
    COLOR_STOPS = [
        (0.0, [0.20, 0.62, 0.18]),     # unripe: green
        (0.5, [0.92, 0.10, 0.08]),     # ripe: vivid red
        (1.0, [0.05, 0.04, 0.03]),     # overripe: near-black
    ]
    TRUNK_COLOR = [0.45, 0.28, 0.12]
    LEAF_COLOR = [0.22, 0.58, 0.20]
    BRANCH_COLOR = [0.40, 0.24, 0.10]
    STEM_COLOR = [0.0, 0.0, 0.0]      # black
    # Cylinder local axis is +X; this quat maps it onto world +Z (vertical trunk).
    VERTICAL_CYL_Q = [0.70710678, 0.0, 0.70710678, 0.0]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("pick_ripe_apple", {})
        super()._init_task_env_(**kwags)
        # Start ripening only AFTER setup settle (~2000 steps in _init_task_env_).
        self._ripen_started = True

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.ripen_steps = int(cfg.get("ripen_steps", self.RIPEN_STEPS_DEFAULT))
        self.red_window = float(np.random.uniform(
            cfg.get("red_window_min", self.RED_WINDOW_MIN_DEFAULT),
            cfg.get("red_window_max", self.RED_WINDOW_MAX_DEFAULT)))
        self.red_tol = float(cfg.get("red_tolerance", self.RED_TOLERANCE_DEFAULT))

        self.tree_x = float(cfg.get("tree_x", self.TREE_X_DEFAULT))
        self.tree_y = float(cfg.get("tree_y", self.TREE_Y_DEFAULT))
        self.basket_x = float(cfg.get("basket_x", self.BASKET_X_DEFAULT))
        self.basket_y = float(cfg.get("basket_y", self.BASKET_Y_DEFAULT))
        self.basket_scale = float(cfg.get("basket_scale", self.BASKET_SCALE_DEFAULT))
        hang_dx = float(cfg.get("apple_hang_dx", self.APPLE_HANG_DX_DEFAULT))
        apple_sm = float(os.environ.get(
            "PRA_APPLE_SCALE", str(cfg.get("apple_scale_mult", self.APPLE_SCALE_DEFAULT))))

        side_cfg = str(cfg.get("apple_side", "random")).lower()
        if side_cfg in ("left", "l", "-1"):
            self.apple_side = -1.0
        elif side_cfg in ("right", "r", "+1", "1"):
            self.apple_side = 1.0
        else:
            self.apple_side = float(np.random.choice([-1.0, 1.0]))

        z0 = 0.74 + self.table_z_bias
        self._z0 = z0

        # Staggered branch heights: randomly pick which side is the higher branch.
        z_high = float(cfg.get("branch_z_high", self.BRANCH_Z_HIGH))
        z_low = float(cfg.get("branch_z_low", self.BRANCH_Z_LOW))
        high_side_cfg = str(cfg.get("high_branch_side", "random")).lower()
        if high_side_cfg in ("left", "l", "-1"):
            self.high_branch_side = -1.0
        elif high_side_cfg in ("right", "r", "+1", "1"):
            self.high_branch_side = 1.0
        else:
            self.high_branch_side = float(np.random.choice([-1.0, 1.0]))
        self.branch_z = {
            -1.0: z_high if self.high_branch_side < 0 else z_low,
            +1.0: z_high if self.high_branch_side > 0 else z_low,
        }
        apple_branch_z = self.branch_z[self.apple_side]
        apple_drop = float(cfg.get("apple_drop_below_branch", self.APPLE_DROP_BELOW_BRANCH))
        trunk_h = float(cfg.get("trunk_height", self.TRUNK_HEIGHT))
        # Canonical hang: black thin stem into visual +Z top; branch centered on apple in Y.
        # Shared hang_x for BOTH stem and apple: tip is the outer bound; sample up to
        # APPLE_X_JITTER toward the trunk.
        hang_x_tip = self.tree_x + self.apple_side * hang_dx - self.STEM_INSET_X
        x_jit = float(cfg.get("apple_x_jitter", self.APPLE_X_JITTER))
        self.hang_x = float(hang_x_tip - self.apple_side * np.random.uniform(0.0, x_jit))
        # Identity mesh extends along +Y from the pose origin — pull hang_y forward so
        # the bbox center lands on the branch/stem axis (tree_y).
        apple_scale = 0.8748 * apple_sm
        hang_y = self.tree_y - (
            self.APPLE_Y_CENTER_FRAC * self.APPLE_MESH_CENTER_Y * apple_scale)

        # ---- tree (concept sketch: tall trunk, top canopy, staggered long branches) ----
        # Stem X = hang_x (same sample as the apple).
        self.tree = self._build_tree(
            sapien.Pose([self.tree_x, self.tree_y, z0], [1, 0, 0, 0]),
            trunk_h=trunk_h,
            branch_z_left=self.branch_z[-1.0],
            branch_z_right=self.branch_z[+1.0],
            hang_dx=hang_dx,
            apple_side=self.apple_side,
            apple_drop=apple_drop,
            hang_x_local=self.hang_x - self.tree_x,
        )
        # Apple at the same hang_x; stem seats in the visual top (+Z).
        self._hang_pose = sapien.Pose(
            [self.hang_x, hang_y, z0 + apple_branch_z - apple_drop],
            list(self.APPLE_HANG_Q),
        )

        # ---- hanging apple (kinematic while attached to the tree) ----
        self.apple = create_actor(
            self, pose=self._hang_pose,
            modelname="220_apple_plain", model_id=0, convex=True,
            is_static=False, scale_mult=apple_sm,
        )
        self.apple.set_mass(0.05)
        self._apple_rigid = next(
            (c for c in self.apple.actor.get_components()
             if isinstance(c, sapien.physx.PhysxRigidDynamicComponent)), None)
        if self._apple_rigid is not None:
            try:
                self._apple_rigid.set_disable_gravity(True)
                self._apple_rigid.set_kinematic(True)
                self._apple_rigid.set_linear_damping(5.0)
                self._apple_rigid.set_angular_damping(20.0)
                for s in self._apple_rigid.get_collision_shapes():
                    m = s.get_physical_material()
                    m.set_static_friction(4.0)
                    m.set_dynamic_friction(4.0)
                    m.set_restitution(0.0)
            except Exception:
                pass
        self._apple_shapes = []
        for c in self.apple.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._apple_shapes = list(c.render_shapes)

        # ---- basket (same asset as packing) in front of the tree ----
        self.basket_id = int(np.random.choice([0, 1, 2, 3, 4]))
        self.basket = create_actor(
            self,
            pose=sapien.Pose(
                [self.basket_x, self.basket_y, z0],
                list(self.BASKET_Q),
            ),
            modelname="076_breadbasket",
            model_id=self.basket_id,
            convex=True,
            is_static=True,
            scale_mult=self.basket_scale,
        )
        self.basket_base_z = float(self.basket.get_pose().p[2])
        bcfg = getattr(self.basket, "config", None) or {}
        extents = bcfg.get("extents", [0.0, 0.7, 0.0])
        scale = bcfg.get("scale", [self.basket_scale] * 3)
        basket_height = float(extents[1]) * float(scale[1])
        if basket_height <= 0.0:
            basket_height = 0.07 * self.basket_scale
        self.basket_top_z = self.basket_base_z + basket_height
        self.basket_center = np.array(
            [self.basket_x, self.basket_y], dtype=np.float64)

        self.add_prohibit_area(self.apple, padding=0.03)
        self.add_prohibit_area(self.basket, padding=0.05)

        # ripeness / attach state
        self.ripeness = 0.0
        self._ripen_started = False
        self._apple_attached = True          # still hanging on the tree
        self.r_grasp = None                  # latched at detach
        self._welded = False
        self._weld_offset = None
        self._weld_arm = None
        self._set_apple_color(self.ripeness)

    def _build_tree(self, pose, trunk_h, branch_z_left, branch_z_right,
                    hang_dx, apple_side, apple_drop, hang_x_local=None):
        """Simple concept-sketch tree: tall trunk, top-only canopy, two staggered branches.

        Foliage / branches / stem are visual-only so they don't block curobo approach.
        A thin trunk collision keeps the tree physically grounded.
        """
        builder = self.scene.create_actor_builder()
        phys = self.scene.default_physical_material
        trunk_mat = sapien.render.RenderMaterial(base_color=[*self.TRUNK_COLOR, 1.0])
        branch_mat = sapien.render.RenderMaterial(base_color=[*self.BRANCH_COLOR, 1.0])
        leaf_mat = sapien.render.RenderMaterial(base_color=[*self.LEAF_COLOR, 1.0])
        stem_mat = sapien.render.RenderMaterial(base_color=[*self.STEM_COLOR, 1.0])

        # ---- tall thin trunk ----
        trunk_r = 0.014
        trunk_pose = sapien.Pose([0, 0, trunk_h / 2], self.VERTICAL_CYL_Q)
        builder.add_cylinder_collision(
            pose=trunk_pose, radius=trunk_r * 0.7, half_length=trunk_h / 2, material=phys)
        builder.add_cylinder_visual(
            pose=trunk_pose, radius=trunk_r, half_length=trunk_h / 2, material=trunk_mat)

        # ---- two long horizontal branches at staggered heights ----
        # Tip ends exactly above the apple stem (not past it), so the branch doesn't
        # visually skewer the fruit in top-down views.
        branch_len = float(hang_dx)
        branch_half = [branch_len / 2, 0.008, 0.008]
        for sign, bz in ((-1.0, branch_z_left), (+1.0, branch_z_right)):
            bp = sapien.Pose([sign * branch_len / 2, 0.0, float(bz)], [1, 0, 0, 0])
            builder.add_box_visual(pose=bp, half_size=branch_half, material=branch_mat)

        # ---- stem under the apple's branch (seats into the hole on top of the apple) ----
        branch_z_apple = float(branch_z_left if apple_side < 0 else branch_z_right)
        stem_top = branch_z_apple - 0.008
        # Reach well into the apple body past the pose origin.
        stem_bot = branch_z_apple - float(apple_drop) - 0.018
        stem_half_h = max(0.012, abs(stem_top - stem_bot) / 2)
        stem_z = (stem_top + stem_bot) / 2
        st = self.STEM_HALF_THICK
        # Must match the apple's sampled world X (passed as tree-local hang_x_local).
        if hang_x_local is None:
            raise ValueError("hang_x_local is required so stem X matches the apple")
        stem_pose = sapien.Pose([float(hang_x_local), 0.0, stem_z], [1, 0, 0, 0])
        builder.add_box_visual(
            pose=stem_pose, half_size=[st, st, stem_half_h], material=stem_mat)

        # ---- canopy ONLY at the top (cloud-like blob, concept sketch) ----
        canopy = [
            ([0.00, 0.00, trunk_h + 0.01], [0.055, 0.050, 0.035]),
            ([0.04, 0.01, trunk_h + 0.00], [0.040, 0.038, 0.030]),
            ([-0.04, -0.01, trunk_h + 0.00], [0.040, 0.038, 0.030]),
            ([0.00, 0.03, trunk_h - 0.01], [0.038, 0.035, 0.028]),
            ([0.00, -0.03, trunk_h - 0.01], [0.038, 0.035, 0.028]),
            ([0.00, 0.00, trunk_h + 0.045], [0.042, 0.040, 0.028]),
        ]
        for center, half in canopy:
            builder.add_box_visual(
                pose=sapien.Pose(center, [1, 0, 0, 0]), half_size=half, material=leaf_mat)

        builder.set_initial_pose(pose)
        entity = builder.build_static(name="apple_tree")
        data = {
            "center": [0, 0, 0],
            "extents": [2 * branch_len, 0.12, trunk_h + 0.10],
            "scale": [1, 1, 1],
            "transform_matrix": np.eye(4).tolist(),
            "contact_points_pose": [],
            "functional_matrix": [],
            "contact_points_description": [],
            "contact_points_group": [],
            "contact_points_mask": [],
            "target_point_description": [],
        }
        return Actor(entity, data)

    # ----------------------------------------------------------- ripeness
    def _color_for(self, r):
        d = float(np.clip(r, 0.0, 1.0))
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
        return list(rgb) + [1.0]

    def _set_apple_color(self, r):
        col = self._color_for(r)
        for s in self._apple_shapes:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    # --------------------------------------------------------- grasp weld
    def _ee_pos(self, arm):
        p = self.robot.get_left_ee_pose() if arm == "left" else self.robot.get_right_ee_pose()
        return np.array(p[:3], dtype=float)

    def _detach_apple(self, arm=None):
        """Break the tree attachment: freeze ripeness and (optionally) weld to the EE."""
        if self._apple_attached:
            self._apple_attached = False
            if self.r_grasp is None:
                self.r_grasp = float(self.ripeness)
            self._set_apple_color(self.ripeness)   # freeze visual at current ripeness
        if arm is not None and not self._welded:
            self._weld_arm = "left" if str(arm) == "left" else "right"
            self._weld_offset = (
                np.array(self.apple.get_pose().p, dtype=float) - self._ee_pos(self._weld_arm))
            if self._apple_rigid is not None:
                try:
                    self._apple_rigid.set_disable_gravity(True)
                    self._apple_rigid.set_kinematic(True)
                except Exception:
                    pass
            self._welded = True

    def _release_apple(self):
        self._welded = False
        if self._apple_rigid is not None:
            try:
                self._apple_rigid.set_kinematic(False)
                self._apple_rigid.set_disable_gravity(False)
                self._apple_rigid.set_linear_velocity(np.zeros(3))
                self._apple_rigid.set_angular_velocity(np.zeros(3))
            except Exception:
                pass

    def _update_welded_apple(self):
        if not getattr(self, "_welded", False):
            return
        target = self._ee_pos(self._weld_arm) + self._weld_offset
        q = self.apple.get_pose().q
        if self._apple_rigid is not None:
            self._apple_rigid.set_kinematic_target(sapien.Pose(target, q))
        else:
            self.apple.actor.set_pose(sapien.Pose(target, q))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not hasattr(self, "apple"):
            return
        # Keep the apple pinned to the hang pose until detached.
        if getattr(self, "_apple_attached", False) and self._apple_rigid is not None:
            try:
                self._apple_rigid.set_kinematic_target(self._hang_pose)
            except Exception:
                pass
        self._update_welded_apple()
        if not getattr(self, "_ripen_started", False):
            return
        if not getattr(self, "_apple_attached", False):
            return
        self.ripeness = min(1.0, self.ripeness + 1.0 / max(1, self.ripen_steps))
        self._set_apple_color(self.ripeness)

    def _ripen_until(self, target):
        max_steps = int(self.ripen_steps) + 600
        for j in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (j % self.save_freq == 0):
                self._take_picture()
            if (not self._apple_attached) or self.ripeness >= target:
                break

    # ------------------------------------------------------------- policy
    def play_once(self):
        arm = ArmTag("left" if self.apple_side < 0 else "right")
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] side={self.apple_side:+.0f} arm={arm} "
                  f"high_branch={self.high_branch_side:+.0f} "
                  f"branch_z={self.branch_z[self.apple_side]:.3f} "
                  f"ripen_steps={self.ripen_steps} red_window={self.red_window:.3f}",
                  flush=True)

        # OBSERVE: wait until almost red, leading by the grasp+lift duration.
        target = max(0.18, self.red_window - self.GRASP_LEAD_STEPS / max(1, self.ripen_steps))
        self._ripen_until(target)
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] trigger_ripeness={self.ripeness:.3f} target={target:.3f}", flush=True)

        # ACT: contact grasp (frames match stem-up hang quat), detach, lift, drop.
        self.move(self.grasp_actor(
            self.apple, arm_tag=arm, pre_grasp_dis=0.1, gripper_pos=0.0))
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] after grasp: plan={self.plan_success} "
                  f"fail={getattr(self, '_last_plan_fail', None)} "
                  f"apple={np.array(self.apple.get_pose().p)}", flush=True)
        if not self.plan_success:
            self.info["info"] = {
                "{A}": "220_apple_plain/base0",
                "{B}": f"076_breadbasket/base{self.basket_id}",
                "{a}": str(arm),
            }
            return self.info
        self._detach_apple(arm)
        self.move(self.move_by_displacement(arm_tag=arm, z=0.10, move_axis="world"))
        if os.environ.get("PRA_DEBUG"):
            print(f"[PRA] after lift: ripeness={self.ripeness:.3f} r_grasp={self.r_grasp} "
                  f"attached={self._apple_attached} plan={self.plan_success}", flush=True)

        bp = np.array(self.basket.get_pose().p)
        ap = np.array(self.apple.get_pose().p)
        # Hover above the basket opening (rim clearance), then open to drop.
        hover_z = max(self.basket_top_z + 0.04, float(ap[2]))
        self.move(self.move_by_displacement(
            arm_tag=arm,
            x=float(bp[0] - ap[0]),
            y=float(bp[1] - ap[1]),
            z=float(hover_z - ap[2]),
            move_axis="world",
        ))
        placed = self.plan_success
        self.move(self.open_gripper(arm))
        self._release_apple()
        self.move(self.move_by_displacement(arm_tag=arm, z=0.08, move_axis="arm"))
        self.plan_success = placed

        self.info["info"] = {
            "{A}": "220_apple_plain/base0",
            "{B}": f"076_breadbasket/base{self.basket_id}",
            "{a}": str(arm),
        }
        return self.info

    # --------------------------------------------------------- ripeness score
    def _ripeness_score(self):
        if self.r_grasp is None:
            return 0.0
        return float(np.clip(1.0 - abs(self.r_grasp - self.red_window) / 0.5, 0.0, 1.0))

    # ------------------------------------------------------------- success
    def check_success(self):
        ap = np.array(self.apple.get_pose().p)
        xy_close = float(np.linalg.norm(ap[:2] - self.basket_center)) < 0.12
        not_floor = ap[2] > (0.60 + self.table_z_bias)
        settled = ap[2] < (self.basket_top_z + 0.06)
        in_basket = bool(xy_close and settled and not_floor and (self.r_grasp is not None))
        success = bool(in_basket)

        ripe = self._ripeness_score()
        self.info["ripeness_score"] = ripe
        self.info["r_grasp"] = float(self.r_grasp) if self.r_grasp is not None else -1.0
        self.info["final_score"] = ripe if success else 0.0
        self.info["in_basket"] = success
        self.info["apple_side"] = float(self.apple_side)
        return success

    def get_obs(self):
        obs = super().get_obs()
        obs["ripening"] = {
            "ripeness": float(getattr(self, "ripeness", 0.0)),
            "r_grasp": float(self.r_grasp) if getattr(self, "r_grasp", None) is not None else -1.0,
            "red_window": float(getattr(self, "red_window", 0.5)),
            "attached": bool(getattr(self, "_apple_attached", False)),
            "apple_side": float(getattr(self, "apple_side", 0.0)),
        }
        return obs
