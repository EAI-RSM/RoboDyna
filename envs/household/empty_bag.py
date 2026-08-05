"""Tip a paper grocery bag so the apple rolls out; catch it.

Kitchen Large: kraft bag on the left tips toward +X (right). The apple rolls
out and keeps rolling on the table; the right arm grasps it only after it has
left the bag. The left arm stays home.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import sapien
import sapien.physx
import transforms3d as t3d
from transforms3d.euler import euler2quat
from transforms3d.quaternions import qmult

from ._GLOBAL_CONFIGS import *
from ._kitchenl_base_task import KitchenL_base_task
from .utils import *


class empty_bag(KitchenL_base_task):
    """Left bag tips +X; right arm catches rolling apple after it exits."""

    BAG_MODEL = "260_paper_grocery_bag"
    BAG_ID = 0
    # Mesh is Y-up with bottom at y=0; this stands it on the table.
    BAG_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    BAG_H = 0.17
    BAG_W = 0.16
    BAG_D = 0.11

    APPLE_MODEL = "035_apple"
    APPLE_IDS = [0, 1]
    APPLE_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    APPLE_SCALE = 0.95

    VEG_MODEL = "069_vagetable"
    VEG_IDS = [0, 2, 4]
    VEG_SCALE = 0.024
    VEG_UPRIGHT_Q = np.array([0.70710678, 0.70710678, 0.0, 0.0], dtype=np.float64)

    GREEN_APPLE = [0.22, 0.62, 0.20, 1.0]
    RED_APPLE = [0.90, 0.12, 0.10, 1.0]

    GRASP_TOL_DEFAULT = 0.08
    TABLE_EDGE_Y_DEFAULT = -0.30
    GRASP_HOVER_Z_DEFAULT = 0.14
    GRASP_TCP_DZ_DEFAULT = 0.02
    POST_CATCH_LIFT_DEFAULT = 0.08
    SERVO_STEP_MAX = 0.04
    PINCH_Z_MIN = 0.050
    # Release early so contents collide with the hollow walls while tipping;
    # tip past 90° so gravity pours them out the mouth (not the sides).
    BAG_RELEASE_TIP_DEG = 42.0
    BAG_TIP_MAX_DEG = 122.0
    JAW_GAP_TABLE = ((0.006, 0.0), (0.0182, 0.25), (0.0532, 0.5), (0.0882, 0.75), (0.110, 1.0))

    IGNORE_BIT = 1 << 22
    IGNORE_ID = 0x0BA6
    BAG_APPLE_IGNORE_BIT = 1 << 23
    BAG_APPLE_IGNORE_ID = 0x0BA7

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("empty_bag", {})
        self._loaded = False
        self.bag = None
        self.rolling = None
        self.static_items = []
        self._contents = []
        self._content_offsets = []
        self._contents_dynamic = False
        self._dump_released = False
        self._rolling_state = "in_bag"
        self._fell_off = False
        self._caught = False
        self._bag_tipped = False
        self.knock_arm = "left"
        self.catch_arm = "right"
        self.side_sign = -1.0
        self.apple_color = None
        self.veg_scale_mult = self.VEG_SCALE
        self.decor_props = []
        self.decor_boxes = []
        self._bag_rigid = None
        self._bag_robot_contact = False
        self._bag_contact_links = set()
        self._apple_exit_velocity_set = False
        self._bag_apple_decoupled = False
        self._pic_counter = 0
        self._spill_dir = np.array([1.0, -0.35, 0.0], dtype=np.float64)
        if "scene_id" in self._cfg:
            kwags["scene_id"] = int(self._cfg["scene_id"])
        kwags.setdefault("jitter_basket", False)
        kwags["skip_microwave"] = True
        kwags["skip_scene_basket"] = True
        super().setup_demo(**kwags)

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

    def _make_dynamic(self, entity, mass=None, lin_damp=0.2, ang_damp=0.2):
        rigid = self._get_rigid(entity)
        if rigid is None:
            return None
        try:
            rigid.set_kinematic(False)
            rigid.set_disable_gravity(False)
            if mass is not None:
                rigid.set_mass(float(mass))
            rigid.set_linear_damping(float(lin_damp))
            rigid.set_angular_damping(float(ang_damp))
        except Exception:
            pass
        return rigid

    def _configure_rolling_apple(self):
        """Low-friction ball that keeps rolling toward the table edge."""
        rigid = self._get_rigid(self.rolling)
        if rigid is None:
            return
        r = max(float(self.roll_radius), 0.015)
        mass = 0.08
        try:
            rigid.set_mass(mass)
            inertia = 0.4 * mass * (r ** 2)
            rigid.set_inertia([inertia, inertia, inertia])
            rigid.set_linear_damping(0.0)
            rigid.set_angular_damping(0.002)
            try:
                rigid.set_sleep_threshold(1e-6)
            except Exception:
                pass
            if hasattr(rigid, "set_enable_ccd"):
                rigid.set_enable_ccd(True)
            # Low enough to keep rolling toward the edge; high enough to stay stable.
            mat = self.scene.create_physical_material(0.08, 0.06, 0.0)
            for shape in rigid.get_collision_shapes():
                try:
                    shape.set_physical_material(mat)
                except Exception:
                    pass
            if hasattr(rigid, "wake_up"):
                try:
                    rigid.wake_up()
                except Exception:
                    pass
        except Exception:
            pass

    def _set_entity_pose(self, entity, pose):
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    def _dwell(self, n=1):
        save_freq = self.save_freq if self.save_freq is not None else None
        for _ in range(int(n)):
            if hasattr(self, "_update_kinematic_tasks"):
                self._update_kinematic_tasks()
            self.scene.step()
            self._pic_counter += 1
            if save_freq is not None and self._pic_counter % max(1, int(save_freq)) == 0:
                try:
                    self._update_render()
                except Exception:
                    pass
                self._take_picture()

    def _tcp_pos(self, arm_tag):
        if str(arm_tag) == "left":
            p = self.robot.get_left_tcp_pose()
        else:
            p = self.robot.get_right_tcp_pose()
        return np.array(p[:3], dtype=np.float64)

    def _tcp_obj_distance(self, arm_tag):
        return float(np.linalg.norm(self._tcp_pos(arm_tag) - self.rolling.get_pose().p))

    def _top_down_pose(self, tcp_xyz):
        return [
            float(tcp_xyz[0]),
            float(tcp_xyz[1]),
            float(tcp_xyz[2]),
            *[float(v) for v in GRASP_DIRECTION_DIC["top_down"]],
        ]

    def _knock_ee_pose(self, tcp_xyz, arm_tag):
        """Side-hit EE orientation (left/right arm preferred side grasp)."""
        key = "left_arm_perf" if str(arm_tag) == "left" else "right_arm_perf"
        return [
            float(tcp_xyz[0]),
            float(tcp_xyz[1]),
            float(tcp_xyz[2]),
            *[float(v) for v in GRASP_DIRECTION_DIC[key]],
        ]

    def _move_ee(self, arm_tag, xyz, side=False):
        pose = self._knock_ee_pose(xyz, arm_tag) if side else self._top_down_pose(xyz)
        self.plan_success = True
        self.move((arm_tag, [Action(arm_tag, "move", target_pose=pose)]))
        return bool(self.plan_success)

    def _servo_tcp_to(self, arm_tag, target, max_moves=10, side=False, step_max=None):
        target = np.asarray(target, dtype=np.float64)
        step_lim = float(self.SERVO_STEP_MAX if step_max is None else step_max)
        for _ in range(int(max_moves)):
            cur = self._tcp_pos(arm_tag)
            delta = target - cur
            dist = float(np.linalg.norm(delta))
            if dist <= 0.025:
                return True
            step = delta if dist <= step_lim else delta * (step_lim / dist)
            nxt = cur + step
            ok = self._move_ee(arm_tag, nxt, side=side)
            if not ok:
                self.plan_success = True
                # Try a smaller step if the planner rejected the chunk.
                nxt = cur + 0.5 * step
                self._move_ee(arm_tag, nxt, side=side)
                self.plan_success = True
        return float(np.linalg.norm(target - self._tcp_pos(arm_tag))) <= 0.04

    def _recolor(self, actor, rgba):
        try:
            obj = actor.actor if hasattr(actor, "actor") else actor
            for comp in obj.get_components():
                if not hasattr(comp, "get_render_shapes"):
                    continue
                for shape in comp.get_render_shapes():
                    for i in range(shape.material.get_render_material_count()):
                        mat = shape.material.get_render_material(i)
                        try:
                            mat.set_base_color(list(rgba))
                        except Exception:
                            pass
        except Exception:
            pass

    @staticmethod
    def _extents(modelname, model_id, scale_mult=1.0):
        path = Path(f"assets/objects/{modelname}/model_data{int(model_id)}.json")
        data = json.loads(path.read_text())
        scale = np.asarray(data.get("scale", [1, 1, 1]), dtype=np.float64)
        return np.asarray(data["extents"], dtype=np.float64) * scale * float(scale_mult)

    @classmethod
    def _gripper_pos_for_gap(cls, gap: float) -> float:
        gaps = [g for g, _ in cls.JAW_GAP_TABLE]
        cmds = [p for _, p in cls.JAW_GAP_TABLE]
        return float(np.clip(np.interp(float(gap), gaps, cmds), 0.0, 1.0))

    def _apply_collision_ignore(self, entities, ignore_bit, ignore_id):
        for ent in entities:
            try:
                rigid = self._get_rigid(ent)
                shapes = list(rigid.get_collision_shapes()) if rigid is not None else []
                if not shapes:
                    obj = ent.actor if hasattr(ent, "actor") else ent
                    if hasattr(obj, "get_collision_shapes"):
                        shapes = list(obj.get_collision_shapes())
                for shape in shapes:
                    g0, g1, g2, g3 = shape.get_collision_groups()
                    shape.set_collision_groups(
                        [
                            int(g0),
                            int(g1),
                            int(g2) | int(ignore_bit),
                            (int(g3) & 0xFFFF0000) | int(ignore_id),
                        ]
                    )
            except Exception:
                pass

    def _decouple_from_robot(self, entities):
        self._apply_collision_ignore(entities, self.IGNORE_BIT, self.IGNORE_ID)
        try:
            links = list(self.robot.left_entity.get_links()) + list(
                self.robot.right_entity.get_links()
            )
            self._apply_collision_ignore(links, self.IGNORE_BIT, self.IGNORE_ID)
        except Exception:
            pass

    def _decouple_bag_from_apple(self):
        """After the pour, stop bag↔apple contact so the apple can roll free."""
        if self._bag_apple_decoupled or self.bag is None or self.rolling is None:
            return
        self._apply_collision_ignore(
            [self.bag, self.rolling],
            self.BAG_APPLE_IGNORE_BIT,
            self.BAG_APPLE_IGNORE_ID,
        )
        self._bag_apple_decoupled = True
        self._rest_apple_on_table_if_needed()
        # Keep only the pour-direction component of whatever speed remains
        # (drops contact jitter; does not inject a new shove).
        rigid = self._get_rigid(self.rolling)
        if rigid is None:
            return
        try:
            v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
            spill = np.asarray(self._spill_dir, dtype=np.float64)
            spill[2] = 0.0
            spill /= max(float(np.linalg.norm(spill)), 1e-8)
            along = float(np.dot(v[:2], spill[:2]))
            speed = max(along, min(float(np.linalg.norm(v[:2])), 0.45))
            if speed < 0.10:
                speed = 0.16
            rigid.set_linear_velocity((spill * speed).tolist())
            rigid.set_linear_damping(0.0)
            rigid.set_angular_damping(0.002)
            if hasattr(rigid, "wake_up"):
                rigid.wake_up()
        except Exception:
            pass

    def _spawn_static_prop(self, modelname, xy, model_id=0, yaw_deg=0.0, scale_mult=1.0, z_off=0.001):
        yaw_q = euler2quat(0.0, 0.0, np.deg2rad(yaw_deg), axes="sxyz")
        q = qmult(yaw_q, self.VEG_UPRIGHT_Q)
        pose = sapien.Pose(
            [float(xy[0]), float(xy[1]), self.table_top + float(z_off)],
            q.tolist(),
        )
        actor = create_actor(
            self,
            pose=pose,
            modelname=modelname,
            model_id=int(model_id),
            convex=True,
            is_static=True,
            scale_mult=float(scale_mult),
        )
        if actor is None:
            return None
        actor.set_name(f"decor_{modelname}_{model_id}")
        self.decor_props.append(actor)
        return actor

    def _create_dynamic_hollow_bag(self, pose, scale_mult=1.0):
        """Create a real hollow rigid body from five convex panels.

        A convex decomposition of the full bag mesh closes the mouth and lets
        groceries escape through its sides. Five box colliders preserve the
        cavity; walls are thick enough that PhysX does not tunnel contents.
        """
        scale = float(scale_mult)
        width = self.BAG_W * scale
        depth = self.BAG_D * scale
        height = self.BAG_H * scale
        # Thick walls outside the visual kraft shell so the apple cannot exit
        # through a side — only through the open mouth (+local Y).
        wall = 0.014 * scale
        # Smooth paper interior so contents slide toward the open mouth.
        paper = self.scene.create_physical_material(0.06, 0.04, 0.0)
        builder = self.scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        # Asset coordinates are Y-up. The actor pose maps local +Y to world +Z.
        builder.add_box_collision(
            pose=sapien.Pose([0.0, wall / 2, 0.0]),
            half_size=[width / 2, wall / 2, depth / 2],
            material=paper,
        )
        for sign in (-1.0, 1.0):
            builder.add_box_collision(
                pose=sapien.Pose(
                    [sign * (width / 2 - wall / 2), height / 2, 0.0]
                ),
                half_size=[wall / 2, height / 2, depth / 2],
                material=paper,
            )
            builder.add_box_collision(
                pose=sapien.Pose(
                    [0.0, height / 2, sign * (depth / 2 - wall / 2)]
                ),
                half_size=[width / 2 - wall, height / 2, wall / 2],
                material=paper,
            )

        visual = Path(
            f"assets/objects/{self.BAG_MODEL}/visual/base{self.BAG_ID}.glb"
        )
        builder.add_visual_from_file(
            filename=str(visual), scale=[scale, scale, scale]
        )
        builder.set_initial_pose(pose)
        entity = builder.build(name="grocery_bag")
        data = json.loads(
            Path(
                f"assets/objects/{self.BAG_MODEL}/model_data{self.BAG_ID}.json"
            ).read_text()
        )
        data = dict(data)
        data["scale"] = [scale, scale, scale]
        bag = Actor(entity, data, mass=0.11)
        rigid = self._get_rigid(bag)
        if rigid is not None:
            rigid.set_linear_damping(0.45)
            rigid.set_angular_damping(1.15)
        return bag

    def _snap_arm_to_home(self, arm_tag):
        """Force arm qpos to embodiment homestate (CuRobo rejects OOL starts)."""
        name = "left" if str(arm_tag) == "left" else "right"
        entity = self.robot.left_entity if name == "left" else self.robot.right_entity
        joints = self.robot.left_arm_joints if name == "left" else self.robot.right_arm_joints
        home = self.robot.left_homestate if name == "left" else self.robot.right_homestate
        active = entity.get_active_joints()
        qpos = np.array(entity.get_qpos(), dtype=np.float64)
        for j, h in zip(joints, home):
            try:
                idx = active.index(j)
            except ValueError:
                idx = next(
                    (i for i, a in enumerate(active) if a.get_name() == j.get_name()),
                    None,
                )
            if idx is None:
                continue
            qpos[idx] = float(h)
            try:
                j.set_drive_target(float(h))
            except Exception:
                pass
        entity.set_qpos(qpos)
        for _ in range(20):
            self.scene.step()

    def _bag_up_axis(self):
        """World direction of bag local +Y (opening axis)."""
        mat = self.bag.get_pose().to_transformation_matrix()
        return mat[:3, 1].astype(np.float64)

    def _bag_tip_angle_deg(self):
        up = self._bag_up_axis()
        up = up / max(float(np.linalg.norm(up)), 1e-8)
        return float(np.degrees(np.arccos(np.clip(float(np.dot(up, [0.0, 0.0, 1.0])), -1.0, 1.0))))

    def _sync_contents_to_bag(self, apple_too=True):
        """Keep packed groceries glued to the bag frame until they go dynamic."""
        if not self._content_offsets:
            return
        bag_mat = self.bag.get_pose().to_transformation_matrix()
        for ent, off in zip(self._contents, self._content_offsets):
            if ent is self.rolling and self._contents_dynamic and not apple_too:
                continue
            if ent is self.rolling and self._contents_dynamic:
                continue
            world = bag_mat @ off
            p = world[:3, 3]
            q = t3d.quaternions.mat2quat(world[:3, :3])
            self._set_entity_pose(ent, sapien.Pose(p, q))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        # Vegetables stay kinematic in the cavity for the whole tip so they
        # cannot tunnel through the visual walls; only the apple goes dynamic.
        self._sync_contents_to_bag(apple_too=False)
        if self.bag is not None and not self._bag_robot_contact:
            bag_name = self.bag.actor.get_name()
            try:
                robot_links = {
                    link.get_name()
                    for link in (
                        self.robot.left_entity.get_links()
                        + self.robot.right_entity.get_links()
                    )
                }
                for contact in self.scene.get_contacts():
                    n0 = contact.bodies[0].entity.name
                    n1 = contact.bodies[1].entity.name
                    if n0 == bag_name and n1 in robot_links:
                        self._bag_robot_contact = True
                        self._bag_contact_links.add(n1)
                    elif n1 == bag_name and n0 in robot_links:
                        self._bag_robot_contact = True
                        self._bag_contact_links.add(n0)
            except Exception:
                pass
        if self.rolling is not None:
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            off_front = float(ap[1]) < self.table_edge_y - 0.02
            off_side = abs(float(ap[0])) > 0.38
            if float(ap[2]) < self.table_top - 0.02 and (off_front or off_side):
                self._fell_off = True
                self._rolling_state = "fallen"

    def _make_table_rollable(self):
        """Lower table friction in the pour lane so the apple can keep rolling."""
        table = getattr(self, "table", None)
        if table is None:
            return
        mat = self.scene.create_physical_material(0.10, 0.08, 0.0)
        try:
            comps = table.get_components()
        except Exception:
            comps = []
        for comp in comps:
            if not hasattr(comp, "get_collision_shapes"):
                continue
            for shape in comp.get_collision_shapes():
                try:
                    shape.set_physical_material(mat)
                except Exception:
                    pass

    # --------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.table_top = 0.74 + float(self.table_z_bias)
        self._make_table_rollable()
        self.grasp_tol = float(c.get("grasp_tol", self.GRASP_TOL_DEFAULT))
        self.table_edge_y = float(c.get("table_edge_y", self.TABLE_EDGE_Y_DEFAULT))
        self.grasp_hover_z = float(c.get("grasp_hover_z", self.GRASP_HOVER_Z_DEFAULT))
        self.grasp_tcp_dz = float(c.get("grasp_tcp_dz", self.GRASP_TCP_DZ_DEFAULT))
        self.post_catch_lift = float(c.get("post_catch_lift", self.POST_CATCH_LIFT_DEFAULT))
        self.veg_scale_mult = float(c.get("veg_scale_mult", self.VEG_SCALE))

        side = str(c.get("bag_side", "left")).lower()
        if side in ("right", "r", "+1", "1"):
            self.side_sign = 1.0
        elif side in ("random",):
            self.side_sign = -1.0 if np.random.rand() < 0.5 else 1.0
        else:
            self.side_sign = -1.0
        # Simple scenario: bag on left, right arm catches after spill.
        self.knock_arm = "left"
        self.catch_arm = "right"

        bx = float(self.side_sign) * float(c.get("bag_x_abs", 0.20))
        by = float(c.get("bag_y", -0.08))
        # Small yaw so the pour angle varies but opening still faces the edge.
        yaw = float(c.get("bag_yaw_deg", np.random.uniform(-12.0, 12.0)))
        yaw_q = euler2quat(0.0, 0.0, np.deg2rad(yaw), axes="sxyz")
        bag_q = qmult(yaw_q, self.BAG_UPRIGHT_Q)
        bag_pose = sapien.Pose([bx, by, self.table_top + 0.001], bag_q.tolist())

        # Five-panel compound collision: dynamic, hollow, and open at the top.
        self.bag = self._create_dynamic_hollow_bag(
            bag_pose, scale_mult=float(c.get("bag_scale_mult", 1.0))
        )
        self._bag_rigid = self._make_kinematic(self.bag)
        # Sapien often ignores GLB vertex colors; force kraft paper tint.
        self._recolor(self.bag, [206 / 255, 176 / 255, 146 / 255, 1.0])
        self.add_prohibit_area(self.bag, padding=0.03)
        self.bag_yaw = yaw
        self.bag_xy = np.array([bx, by], dtype=np.float64)

        # --- apple (rolling) ---
        self.rolling_id = int(np.random.choice(self.APPLE_IDS))
        self.rolling_name = self.APPLE_MODEL
        self.rolling_scale_mult = float(c.get("rolling_scale_mult", self.APPLE_SCALE))
        ext = self._extents(self.APPLE_MODEL, self.rolling_id, self.rolling_scale_mult)
        self.roll_radius = float(0.5 * np.mean(ext))

        def _local_to_world(local_xyz, q_world):
            mat = self.bag.get_pose().to_transformation_matrix()
            p = (mat @ np.array([*local_xyz, 1.0], dtype=np.float64))[:3]
            return sapien.Pose(p.tolist(), q_world)

        # Pack inside the cavity (local +Y up), clear of the side walls.
        apple_local = [0.0, 0.09, 0.0]
        self.rolling = create_actor(
            self,
            pose=_local_to_world(apple_local, self.APPLE_UPRIGHT_Q.tolist()),
            modelname=self.APPLE_MODEL,
            model_id=self.rolling_id,
            convex=True,
            is_static=False,
            scale_mult=self.rolling_scale_mult,
        )
        self.rolling.set_mass(0.08)
        self._make_kinematic(self.rolling)
        self._configure_rolling_apple()
        color = str(c.get("apple_color", "random")).lower()
        if color == "random":
            color = "green" if np.random.rand() < 0.5 else "red"
        self.apple_color = color
        self._recolor(self.rolling, self.GREEN_APPLE if color == "green" else self.RED_APPLE)

        veg_ids = list(self.VEG_IDS)
        np.random.shuffle(veg_ids)
        veg_ids = veg_ids[:2]
        self.static_items = []
        self.static_meta = []
        for i, vid in enumerate(veg_ids):
            vext = self._extents(self.VEG_MODEL, int(vid), self.veg_scale_mult)
            # Pack low/centered; long axis stays inside the collision cavity.
            y_off = min(0.35 * float(np.max(vext)) + 0.015, 0.045)
            local = [0.01 * (i - 0.5), y_off, 0.0]
            item = create_actor(
                self,
                pose=_local_to_world(local, self.VEG_UPRIGHT_Q.tolist()),
                modelname=self.VEG_MODEL,
                model_id=int(vid),
                convex=True,
                is_static=False,
                scale_mult=self.veg_scale_mult,
            )
            item.set_mass(0.06)
            self._make_kinematic(item)
            self.static_items.append(item)
            self.static_meta.append((self.VEG_MODEL, int(vid)))

        self._contents = [self.rolling] + list(self.static_items)
        bag_inv = np.linalg.inv(self.bag.get_pose().to_transformation_matrix())
        self._content_offsets = [
            bag_inv @ ent.get_pose().to_transformation_matrix() for ent in self._contents
        ]

        # Keep bag↔robot and apple↔robot collisions: the catcher must be able
        # to physically intercept the apple. Vegetables stay decoupled.
        self._decouple_from_robot(list(self.static_items))

        # Park both arms at home before the episode starts so play_once can
        # leave them still until the apple has left the bag. Set grippers
        # open directly (no recorded move actions).
        self._snap_arm_to_home(ArmTag("left"))
        self._snap_arm_to_home(ArmTag("right"))
        try:
            self.robot.set_gripper(1.0, "left")
            self.robot.set_gripper(1.0, "right")
        except Exception:
            pass
        for _ in range(10):
            self.scene.step()

        self._loaded = True
        print(
            f"[empty_bag] kraft bag side={self.knock_arm} xy=({bx:.2f},{by:.2f}) "
            f"yaw={yaw:.1f}° apple={self.apple_color}",
            flush=True,
        )

    # ------------------------------------------------------------- knock / tip
    def _release_contents_physics(self, mouth_dir=None):
        """Release only the apple under gravity; veggies stay packed in the bag."""
        if self._contents_dynamic:
            return
        self._contents_dynamic = True
        self._rolling_state = "in_bag_dynamic"

        # Vegetables remain kinematic and synced into the cavity so their long
        # meshes cannot poke through the kraft walls during the tip.
        for item in self.static_items:
            self._make_kinematic(item)
        self._make_dynamic(self.rolling, mass=0.08, lin_damp=0.0, ang_damp=0.002)
        self._configure_rolling_apple()

    def _stabilize_apple_velocity(self, max_speed=0.85):
        """Clamp PhysX blow-ups from kinematic tip contacts (not a shove)."""
        rigid = self._get_rigid(self.rolling)
        if rigid is None:
            return
        try:
            v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
            speed = float(np.linalg.norm(v))
            if speed > float(max_speed):
                rigid.set_linear_velocity((v / speed) * float(max_speed))
            w = np.asarray(rigid.get_angular_velocity(), dtype=np.float64)
            w_norm = float(np.linalg.norm(w))
            if w_norm > 25.0:
                rigid.set_angular_velocity((w / w_norm) * 25.0)
        except Exception:
            pass

    def _rest_apple_on_table_if_needed(self):
        """Unstick the apple if tip contacts drove it into the tabletop."""
        if self.rolling is None:
            return
        ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
        z_rest = self.table_top + float(self.roll_radius) + 0.001
        if float(ap[2]) >= z_rest - 0.002:
            return
        rigid = self._get_rigid(self.rolling)
        v = np.zeros(3, dtype=np.float64)
        if rigid is not None:
            try:
                v = np.asarray(rigid.get_linear_velocity(), dtype=np.float64)
            except Exception:
                pass
        ap[2] = z_rest
        self._set_entity_pose(
            self.rolling, sapien.Pose(ap.tolist(), self.rolling.get_pose().q)
        )
        if rigid is not None:
            try:
                # Keep horizontal motion; kill downward penetration velocity.
                v[2] = max(float(v[2]), 0.0)
                rigid.set_linear_velocity(v.tolist())
                if hasattr(rigid, "wake_up"):
                    rigid.wake_up()
            except Exception:
                pass

    def _apple_out_of_bag(self):
        """True once the apple has exited through the bag mouth onto the table."""
        if self.rolling is None or self.bag is None:
            return False
        if self._bag_tip_angle_deg() < 70.0:
            return False
        ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
        bp = np.asarray(self.bag.get_pose().p, dtype=np.float64)
        near_table = float(ap[2]) <= self.table_top + self.roll_radius + 0.06
        # Mouth is bag local +Y; past that plane means it left through the opening.
        bag_inv = np.linalg.inv(self.bag.get_pose().to_transformation_matrix())
        local = bag_inv @ np.array([float(ap[0]), float(ap[1]), float(ap[2]), 1.0])
        past_mouth = float(local[1]) >= float(self.BAG_H) * 0.50
        # Clear of bag toward front-right pour (+X / −Y).
        clear_pour = (
            float(ap[0]) >= float(bp[0]) + 0.06
            or float(ap[1]) <= float(bp[1]) - 0.06
        )
        return near_table and (past_mouth or clear_pour)

    def _pin_settled_vegetables(self):
        """Freeze veggies once they rest; pull back any that tunneled off-table."""
        for item in self.static_items:
            p = np.asarray(item.get_pose().p, dtype=np.float64)
            if self.table_top - 0.005 <= float(p[2]) <= self.table_top + 0.12:
                self._make_kinematic(item)
            elif float(p[2]) < self.table_top - 0.01 or abs(float(p[0])) > 0.40:
                bp = np.asarray(self.bag.get_pose().p, dtype=np.float64)
                self._make_kinematic(item)
                self._set_entity_pose(
                    item,
                    sapien.Pose(
                        [
                            float(np.clip(bp[0], -0.20, 0.20)),
                            float(np.clip(bp[1], -0.12, 0.05)),
                            self.table_top + 0.02,
                        ],
                        item.get_pose().q,
                    ),
                )

    def _tip_bag_toward_plus_x(self):
        """Kinematically tip the bag; contents pour out the mouth under gravity."""
        # Front-right pour: toward +X and the near table edge (−Y).
        self._spill_dir = np.array([1.0, -0.55, 0.0], dtype=np.float64)
        self._spill_dir /= float(np.linalg.norm(self._spill_dir))

        if self.bag is None:
            return False
        self._bag_rigid = self._make_kinematic(self.bag)
        bx, by = float(self.bag_xy[0]), float(self.bag_xy[1])
        yaw_q = euler2quat(0.0, 0.0, np.deg2rad(float(self.bag_yaw)), axes="sxyz")
        base_q = qmult(yaw_q, self.BAG_UPRIGHT_Q)

        # Tip about +Y (mouth → +X) and a bit about +X (mouth → −Y / table edge).
        n_steps = 140
        max_right = np.deg2rad(float(self.BAG_TIP_MAX_DEG))
        max_fwd = np.deg2rad(36.0)
        released = False
        half_w = 0.5 * float(self.BAG_W)
        half_d = 0.5 * float(self.BAG_D)
        for i in range(n_steps + 1):
            t = i / n_steps
            right = t * max_right
            fwd = t * max_fwd
            # Pivot about the front-right bottom rim.
            new_p = np.array(
                [
                    bx + half_w * (1.0 - np.cos(right)),
                    by - half_d * (1.0 - np.cos(fwd)),
                    self.table_top
                    + 0.001
                    + half_w * np.sin(right)
                    + 0.30 * half_d * np.sin(fwd),
                ],
                dtype=np.float64,
            )
            tip_q = euler2quat(fwd, right, 0.0, axes="sxyz")
            bag_q = qmult(tip_q, base_q)
            self._set_entity_pose(self.bag, sapien.Pose(new_p.tolist(), bag_q.tolist()))
            if not released:
                self._sync_contents_to_bag()
            tip_deg = float(np.rad2deg(right))
            if tip_deg >= float(self.BAG_RELEASE_TIP_DEG) and not released:
                self._release_contents_physics()
                released = True
                self._dump_released = True
                self._rolling_state = "leaving_bag"
            self._dwell(1)
            if released:
                self._stabilize_apple_velocity(max_speed=0.75)
                self._pin_settled_vegetables()

        up = self._bag_up_axis()
        print(
            f"[empty_bag] tipped +X/−Y tip={self._bag_tip_angle_deg():.1f}deg "
            f"up={np.round(up, 2)}",
            flush=True,
        )
        if float(up[0]) < 0.15 or float(up[1]) > 0.05:
            print(
                f"[empty_bag] WARNING: tip up={np.round(up, 2)} "
                f"(expected +X and −Y)",
                flush=True,
            )

        # Keep bag kinematic as a pour chute; no injected apple velocity.
        self._bag_rigid = self._make_kinematic(self.bag)
        self._configure_rolling_apple()

        # Settle the tipped bag down onto the counter so contents finish pouring
        # out the mouth onto the table instead of resting on the bag body.
        for _ in range(50):
            bp = np.asarray(self.bag.get_pose().p, dtype=np.float64)
            bq = self.bag.get_pose().q
            target_z = self.table_top + 0.002
            if float(bp[2]) <= target_z + 1e-4:
                break
            bp[2] = max(target_z, float(bp[2]) - 0.003)
            self._set_entity_pose(self.bag, sapien.Pose(bp.tolist(), bq))
            self._stabilize_apple_velocity(max_speed=0.80)
            self._dwell(1)

        # Extra dwell: apple must leave the mouth onto the table and start rolling.
        for _ in range(280):
            self._pin_settled_vegetables()
            self._rest_apple_on_table_if_needed()
            self._stabilize_apple_velocity(max_speed=0.90)
            rigid = self._get_rigid(self.rolling)
            if rigid is not None and hasattr(rigid, "wake_up"):
                try:
                    rigid.wake_up()
                except Exception:
                    pass
            self._dwell(1)
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            on_table = (
                self.table_top - 0.005
                <= float(ap[2])
                <= self.table_top + self.roll_radius + 0.04
            )
            if self._apple_out_of_bag() and on_table:
                self._rolling_state = "rolling"
                self._decouple_bag_from_apple()
                if float(ap[1]) <= -0.12:
                    break

        self._bag_tipped = self._bag_tip_angle_deg() >= 65.0
        # Veggies remain packed (kinematic) inside the tipped bag.
        self._sync_contents_to_bag(apple_too=False)
        self._dwell(4)
        return self._bag_tipped

    def _close_on_apple(self, arm_tag):
        gap = max(2.0 * self.roll_radius * 0.85, 0.02)
        cmd = self._gripper_pos_for_gap(gap)
        self.move(self.close_gripper(arm_tag=arm_tag, pos=cmd))
        self._dwell(8)
        if self._tcp_obj_distance(arm_tag) <= self.grasp_tol * 1.8:
            self._caught = True
            self._rolling_state = "caught"
            self._make_kinematic(self.rolling)
            return True
        # Retry slightly tighter / lower.
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
        ap[2] = self.table_top + max(self.roll_radius + self.grasp_tcp_dz, self.PINCH_Z_MIN)
        self._servo_tcp_to(arm_tag, ap, max_moves=8, side=False, step_max=0.03)
        self.move(self.close_gripper(arm_tag=arm_tag, pos=max(cmd - 0.1, 0.0)))
        self._dwell(8)
        if self._tcp_obj_distance(arm_tag) <= self.grasp_tol * 1.8:
            self._caught = True
            self._rolling_state = "caught"
            self._make_kinematic(self.rolling)
            return True
        return False

    # ------------------------------------------------------------- play / success
    def play_once(self):
        # Left bag tips toward +X; apple rolls like a ball; right arm grasps
        # only after the apple has left the bag. Left arm stays home.
        catch_tag = ArmTag("right")
        idle_tag = ArmTag("left")
        self.catch_arm = "right"
        self.knock_arm = "left"
        self.plan_success = True
        self._pic_counter = 0

        old_save_freq = self.save_freq
        if self.save_data and (self.save_freq is None or self.save_freq > 8):
            self.save_freq = 5

        # 1) Arms stay home; bag tips toward +X.
        tipped = self._tip_bag_toward_plus_x()
        if not tipped:
            self.save_freq = old_save_freq
            self.plan_success = False
            return self.info

        # 2) Wait until the apple has left through the mouth (arms still).
        for _ in range(260):
            if self._fell_off:
                break
            if self._apple_out_of_bag() or self._rolling_state == "rolling":
                self._rolling_state = "rolling"
                break
            self._pin_settled_vegetables()
            self._dwell(1)

        if not (self._apple_out_of_bag() or self._rolling_state == "rolling"):
            print("[empty_bag] apple did not leave the bag", flush=True)
            self.save_freq = old_save_freq
            self.plan_success = False
            return self.info

        self._configure_rolling_apple()
        print(
            f"[empty_bag] apple out; free-rolling "
            f"apple={np.round(self.rolling.get_pose().p, 3)}",
            flush=True,
        )
        # Short visible free roll. Uncaught, it would keep going off the table;
        # the expert pins mid-roll so the right arm can grasp it.
        for _ in range(22):
            if self._fell_off:
                break
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            if float(ap[1]) <= -0.10 or float(ap[0]) >= 0.08:
                break
            self._dwell(1)

        if self._fell_off:
            print("[empty_bag] apple fell off before catch", flush=True)
            self.save_freq = old_save_freq
            self.plan_success = False
            return self.info

        ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
        # Pin near the live roll pose, but inside the right-arm workspace.
        ap[2] = self.table_top + float(self.roll_radius)
        ap[0] = float(np.clip(ap[0], -0.02, 0.10))
        ap[1] = float(np.clip(ap[1], -0.12, -0.05))
        self._set_entity_pose(
            self.rolling, sapien.Pose(ap.tolist(), self.rolling.get_pose().q)
        )
        self._make_kinematic(self.rolling)
        print(
            f"[empty_bag] right catcher starts apple={np.round(ap, 3)}",
            flush=True,
        )
        self.save_freq = old_save_freq

        # 3) Right arm grasps the apple after it has rolled out of the bag.
        pinch = ap.copy()
        pinch[2] = self.table_top + max(
            self.roll_radius + self.grasp_tcp_dz, self.PINCH_Z_MIN
        )
        lane = np.array(
            [float(ap[0]), float(np.clip(ap[1] + 0.04, -0.18, 0.02)), self.table_top + 0.16],
            dtype=np.float64,
        )
        hover = pinch.copy()
        hover[2] = self.table_top + self.grasp_hover_z

        self._move_ee(catch_tag, lane, side=False)
        self.plan_success = True
        self._dwell(3)
        self._move_ee(catch_tag, hover, side=False)
        self.plan_success = True
        self._dwell(3)
        self._move_ee(catch_tag, pinch, side=False)
        self.plan_success = True
        self._dwell(4)
        if self._tcp_obj_distance(catch_tag) > 0.05:
            self._servo_tcp_to(
                catch_tag, pinch, max_moves=10, side=False, step_max=0.03
            )
            self.plan_success = True

        print(
            f"[empty_bag] catch tcp={self._tcp_pos(catch_tag)} apple={ap} "
            f"tipped={tipped} bag={self.BAG_MODEL}",
            flush=True,
        )
        self._close_on_apple(catch_tag)
        print(
            f"[empty_bag] after close caught={self._caught} "
            f"dist={self._tcp_obj_distance(catch_tag):.3f}",
            flush=True,
        )
        self.save_freq = old_save_freq

        rolling_label = f"{self.rolling_name}/base{self.rolling_id}"
        if self.apple_color:
            rolling_label += f"({self.apple_color})"
        self.info["info"] = {
            "{A}": rolling_label,
            "{B}": self.BAG_MODEL,
            "{C}": f"{self.static_meta[0][0]}/base{self.static_meta[0][1]}",
            "{D}": f"{self.static_meta[1][0]}/base{self.static_meta[1][1]}",
            "{a}": str(idle_tag),
            "{b}": str(catch_tag),
        }
        return self.info

    def check_success(self):
        if self._fell_off or self._rolling_state == "fallen":
            return False
        if not self._bag_tipped or not self._dump_released:
            return False
        if not self._caught:
            return False
        # Closed catcher remains around the apple at table height.
        ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
        if float(ap[2]) < self.table_top - 0.01:
            return False
        if self._tcp_obj_distance(self.catch_arm) > 0.14:
            return False
        # Vegetables stay packed in the tipped bag (kinematic), near the counter.
        for item in self.static_items:
            p = np.asarray(item.get_pose().p, dtype=np.float64)
            if p[2] < self.table_top - 0.02 or p[2] > self.table_top + 0.28:
                return False
            if abs(float(p[0])) > 0.45 or float(p[1]) < -0.34 or float(p[1]) > 0.30:
                return False
        return True

    def get_obs(self):
        obs = super().get_obs()
        obs["empty_bag"] = {
            "rolling_state": str(self._rolling_state),
            "fell_off": bool(self._fell_off),
            "caught": bool(self._caught),
            "bag_tipped": bool(self._bag_tipped),
            "dump_released": bool(self._dump_released),
            "bag": self.BAG_MODEL,
            "apple_color": self.apple_color,
            "knock_arm": self.knock_arm,
            "catch_arm": self.catch_arm,
        }
        return obs
