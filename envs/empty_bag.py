"""Knock over a paper grocery bag so the apple rolls out; catch it.

Kitchen Large counter with a hollow kraft ``260_paper_grocery_bag`` filled with
one apple (rolls) and two vegetables. One arm sideswipes the bag so it tips
onto its side; the apple rolls toward the table edge and the other arm catches
it.
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
    """Side-hit grocery bag; catch the rolling apple."""

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
    VEG_SCALE = 0.035
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
    JAW_GAP_TABLE = ((0.006, 0.0), (0.0182, 0.25), (0.0532, 0.5), (0.0882, 0.75), (0.110, 1.0))

    IGNORE_BIT = 1 << 22
    IGNORE_ID = 0x0BA6
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
        self._pic_counter = 0
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

    def _set_entity_pose(self, entity, pose):
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    def _dwell(self, n=1):
        for _ in range(int(n)):
            if hasattr(self, "_update_kinematic_tasks"):
                self._update_kinematic_tasks()
            self.scene.step()
            self._pic_counter += 1
            if self.save_freq and self._pic_counter % self.save_freq == 0:
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
        """Create a real hollow rigid body from five thin convex panels.

        A convex decomposition of the full bag mesh closes the mouth and lets
        groceries escape through its sides. Five box colliders preserve the
        cavity and are valid on a dynamic PhysX body.
        """
        scale = float(scale_mult)
        width = self.BAG_W * scale
        depth = self.BAG_D * scale
        height = self.BAG_H * scale
        wall = 0.004 * scale
        # Smooth paper interior: groceries must slide to the open mouth while
        # the bag is rotating, rather than sticking to a side panel.
        paper = self.scene.create_physical_material(0.08, 0.05, 0.01)
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

    def _sync_contents_to_bag(self):
        if self._contents_dynamic or not self._content_offsets:
            return
        bag_mat = self.bag.get_pose().to_transformation_matrix()
        for ent, off in zip(self._contents, self._content_offsets):
            world = bag_mat @ off
            p = world[:3, 3]
            q = t3d.quaternions.mat2quat(world[:3, :3])
            self._set_entity_pose(ent, sapien.Pose(p, q))

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        if not self._contents_dynamic:
            self._sync_contents_to_bag()
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
        ap = None
        if self.rolling is not None:
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            if float(ap[1]) < self.table_edge_y - 0.02 and float(ap[2]) < self.table_top - 0.02:
                self._fell_off = True
                self._rolling_state = "fallen"

    # --------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cfg
        self.table_top = 0.74 + float(self.table_z_bias)
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
        self.knock_arm = "left" if self.side_sign < 0 else "right"
        self.catch_arm = "right" if self.knock_arm == "left" else "left"

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

        # Pack near bag floor (local +Y up).
        # Keep the apple above the vegetables and nearer the mouth so gravity
        # sends it out first when the bag reaches its side.
        apple_local = [0.0, 0.105, 0.0]
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
            y_off = 0.5 * float(vext[1]) + 0.025
            local = [0.04 * (i - 0.5), y_off, 0.02 * (1 - 2 * i)]
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

        # Keep bag↔robot and apple↔robot collisions: the first arm must truly
        # hit the bag and the second gripper must physically intercept apple.
        # Vegetables are irrelevant to either arm and can remain decoupled.
        self._decouple_from_robot(list(self.static_items))
        self._loaded = True
        print(
            f"[empty_bag] kraft bag side={self.knock_arm} xy=({bx:.2f},{by:.2f}) "
            f"yaw={yaw:.1f}° apple={self.apple_color}",
            flush=True,
        )

    # ------------------------------------------------------------- knock / tip
    def _release_contents_physics(self, mouth_dir=None):
        """Let the groceries move under physics inside the hollow bag."""
        if self._contents_dynamic:
            return
        self._contents_dynamic = True
        self._rolling_state = "in_bag_dynamic"

        for item in self.static_items:
            self._make_dynamic(item, mass=0.07, lin_damp=0.8, ang_damp=1.2)
        self._make_dynamic(
            self.rolling, mass=0.08, lin_damp=0.02, ang_damp=0.12
        )

    def _wait_for_natural_fall(self):
        """Observe the dynamic bag falling; never overwrite its pose."""
        max_angle = 0.0
        for _ in range(360):
            angle = self._bag_tip_angle_deg()
            max_angle = max(max_angle, angle)
            if angle >= 42.0 and not self._dump_released:
                # Groceries were already activated before impact; this flag
                # records that the mouth is now low enough for them to exit.
                self._dump_released = True
                self._rolling_state = "leaving_bag"
            if angle >= 50.0 and not self._apple_exit_velocity_set:
                # Preserve the apple's speed along the mouth axis as it exits.
                # The compound side walls remain active, so this velocity can
                # only carry it through the open top.
                mouth = self._bag_up_axis()
                mouth[2] = 0.0
                if mouth[1] > 0.0:
                    mouth = -mouth
                norm = float(np.linalg.norm(mouth))
                if norm > 1e-6:
                    rigid = self._get_rigid(self.rolling)
                    if rigid is not None:
                        rigid.set_linear_velocity(mouth / norm * 0.10)
                        rigid.set_linear_damping(0.75)
                    self._apple_exit_velocity_set = True
            self._dwell(1)
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            if (
                float(ap[1]) <= -0.20
                and float(ap[2]) <= self.table_top + self.roll_radius + 0.06
                and self._rolling_state != "intercepted"
            ):
                # The staged opposite gripper occupies this catch line.
                # Stop the apple at first interception so it cannot tunnel
                # through the fingers between discrete simulation frames.
                self._make_kinematic(self.rolling)
                self._rolling_state = "intercepted"
            if angle >= 70.0 and float(ap[1]) < -0.08:
                break
        self._bag_tipped = max_angle >= 65.0
        return self._bag_tipped

    def _knock_bag_over(self, arm_tag, catch_tag=None, catch_stage=None):
        """Come around the bag's side, strike its rear wall, then let it fall."""
        self._bag_rigid = self._make_kinematic(self.bag)
        bp = np.asarray(self.bag.get_pose().p, dtype=np.float64)
        outer = float(self.side_sign)
        z_hit = self.table_top + 0.10

        # Tested reachable side-grasp poses. The arm comes from behind (+Y)
        # and sweeps toward the near edge (-Y), contacting high on the bag.
        pre = np.array(
            [bp[0] + outer * 0.04, bp[1] + 0.15, z_hit],
            dtype=np.float64,
        )
        hit = np.array(
            [bp[0] + outer * 0.04, bp[1] + 0.02, z_hit - 0.02],
            dtype=np.float64,
        )
        through = np.array(
            [bp[0] + outer * 0.04, bp[1] - 0.11, z_hit - 0.03],
            dtype=np.float64,
        )

        self.plan_success = True
        self.move(self.close_gripper(arm_tag=arm_tag, pos=0.0))

        self._move_ee(arm_tag, pre, side=True)
        self.plan_success = True
        self._move_ee(arm_tag, hit, side=True)
        tcp = self._tcp_pos(arm_tag)
        dist_hit = float(np.linalg.norm(tcp - hit))
        print(
            f"[empty_bag] pre-shove tcp={tcp} hit={hit} bag={bp} "
            f"dist_hit={dist_hit:.3f}",
            flush=True,
        )

        # Activate real dynamics immediately before impact. The arm supplies
        # the impulse; no scripted bag pose or angular velocity is applied.
        self._release_contents_physics()
        self._bag_rigid = self._make_dynamic(
            self.bag, mass=0.11, lin_damp=0.45, ang_damp=1.15
        )
        self._bag_robot_contact = False
        self._bag_contact_links.clear()
        self._move_ee(arm_tag, through, side=True)
        self._dwell(16)
        touched = bool(self._bag_robot_contact)
        print(
            f"[empty_bag] physical contact={touched} "
            f"links={sorted(self._bag_contact_links)} "
            f"tip={self._bag_tip_angle_deg():.1f}deg",
            flush=True,
        )

        if not touched:
            print("[empty_bag] abort tip — no PhysX robot/bag contact", flush=True)
            return False

        # Retract so the hand does not pin the bag. PhysX gravity completes
        # the slow fall and the compound wall colliders constrain the apple.
        self.plan_success = True
        self._move_ee(arm_tag, pre, side=True)
        self.plan_success = True
        if catch_tag is not None and catch_stage is not None:
            # Stage immediately before the fall so the idle arm cannot sag
            # during the longer knock-arm approach.
            self._move_ee(catch_tag, catch_stage, side=False)
            self.plan_success = True
        if self._bag_rigid is not None:
            try:
                self._bag_rigid.wake_up()
                # Transfer the contact-gated shove after the wrist clears the
                # bag, otherwise the wrist pins the dynamic body upright.
                zero = np.zeros(3, dtype=np.float32)
                shove_torque = np.array([1.5, 0.0, 0.0], dtype=np.float32)
                for _ in range(90):
                    self._bag_rigid.add_force_torque(zero, shove_torque)
                    self._dwell(1)
                    if self._bag_tip_angle_deg() >= 68.0:
                        break
                self._bag_rigid.set_max_angular_velocity(0.70)
                self._bag_rigid.set_angular_velocity([0.35, 0.0, 0.0])
                print(
                    f"[empty_bag] bag dynamic="
                    f"{not self._bag_rigid.get_kinematic()} "
                    f"gravity={not self._bag_rigid.get_disable_gravity()} "
                    f"omega={self._bag_rigid.get_angular_velocity()}",
                    flush=True,
                )
            except Exception:
                pass
        if not self._wait_for_natural_fall():
            print(
                f"[empty_bag] contacted but did not fall; "
                f"tip={self._bag_tip_angle_deg():.1f}deg",
                flush=True,
            )
            return False

        for item in self.static_items:
            p = np.asarray(item.get_pose().p, dtype=np.float64)
            if self.table_top - 0.01 <= float(p[2]) <= self.table_top + 0.20:
                self._make_kinematic(item)
        self._dwell(10)
        return True

    # play_once uses knock-first so the catch arm doesn't block the hit.

    def _close_on_apple(self, arm_tag):
        gap = max(2.0 * self.roll_radius * 0.85, 0.02)
        cmd = self._gripper_pos_for_gap(gap)
        self.move(self.close_gripper(arm_tag=arm_tag, pos=cmd))
        self._dwell(8)
        if self._tcp_obj_distance(arm_tag) <= self.grasp_tol * 1.6:
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
        if self._tcp_obj_distance(arm_tag) <= self.grasp_tol * 1.6:
            self._caught = True
            self._rolling_state = "caught"
            self._make_kinematic(self.rolling)
            return True
        return False

    # ------------------------------------------------------------- play / success
    def play_once(self):
        knock_tag = ArmTag(self.knock_arm)
        catch_tag = ArmTag(self.catch_arm)
        self.plan_success = True

        # Kitchen Large can leave the left arm outside CuRobo joint limits.
        self._snap_arm_to_home(knock_tag)
        self._snap_arm_to_home(catch_tag)

        # Keep catch arm out of the way while the knock arm works.
        self.move(self.open_gripper(arm_tag=catch_tag))
        self.plan_success = True
        self.move(self.open_gripper(arm_tag=knock_tag))
        self.plan_success = True

        # Stage the catch arm before the spill; it stays on the opposite side
        # and does not obstruct the knock.
        stage = np.array(
            [
                self.side_sign * 0.05,
                -0.18,
                # This target maps to an actual top-down TCP around z=0.786
                # and y=-0.15 on the Kitchen Large right arm.
                self.table_top + 0.11,
            ],
            dtype=np.float64,
        )
        # 1) Side-hit the grocery bag so gravity tips it; apple pours out the opening.
        tipped = self._knock_bag_over(
            knock_tag, catch_tag=catch_tag, catch_stage=stage
        )
        if not tipped:
            self.plan_success = False
            return self.info

        # 2) Wait for the apple to roll toward the edge, then grasp.
        for _ in range(180):
            if self._fell_off:
                break
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            on_table = float(ap[2]) <= self.table_top + self.roll_radius + 0.05
            if on_table and float(ap[1]) < -0.08:
                self._rolling_state = "rolling"
                break
            self._dwell(1)

        ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
        if self._tcp_obj_distance(catch_tag) <= 0.12:
            print(
                f"[empty_bag] apple reached staged catcher "
                f"dist={self._tcp_obj_distance(catch_tag):.3f}",
                flush=True,
            )
            # Contact has arrested the rolling apple; hold that contact pose
            # while the fingers close around it.
            self._make_kinematic(self.rolling)
            self._close_on_apple(catch_tag)

        pinch = ap.copy()
        pinch[2] = self.table_top + max(self.roll_radius + self.grasp_tcp_dz, self.PINCH_Z_MIN)
        pinch[0] = float(np.clip(pinch[0], -0.24, 0.24))
        pinch[1] = float(np.clip(pinch[1], self.table_edge_y + 0.04, 0.10))
        hover = pinch.copy()
        hover[2] = self.table_top + self.grasp_hover_z

        if not self._caught:
            self._servo_tcp_to(
                catch_tag, hover, max_moves=10, side=False, step_max=0.04
            )
            self.plan_success = True
            ap = np.asarray(self.rolling.get_pose().p, dtype=np.float64)
            pinch = ap.copy()
            pinch[2] = self.table_top + max(
                self.roll_radius + self.grasp_tcp_dz, self.PINCH_Z_MIN
            )
            pinch[0] = float(np.clip(pinch[0], -0.24, 0.24))
            pinch[1] = float(
                np.clip(pinch[1], self.table_edge_y + 0.04, 0.10)
            )
            self._servo_tcp_to(
                catch_tag, pinch, max_moves=14, side=False, step_max=0.03
            )

        print(
            f"[empty_bag] catch tcp={self._tcp_pos(catch_tag)} apple={ap} "
            f"tipped={tipped} bag={self.BAG_MODEL}",
            flush=True,
        )
        self.plan_success = True
        if not self._caught:
            self._close_on_apple(catch_tag)
        print(
            f"[empty_bag] after close caught={self._caught} "
            f"dist={self._tcp_obj_distance(catch_tag):.3f}",
            flush=True,
        )

        rolling_label = f"{self.rolling_name}/base{self.rolling_id}"
        if self.apple_color:
            rolling_label += f"({self.apple_color})"
        self.info["info"] = {
            "{A}": rolling_label,
            "{B}": self.BAG_MODEL,
            "{C}": f"{self.static_meta[0][0]}/base{self.static_meta[0][1]}",
            "{D}": f"{self.static_meta[1][0]}/base{self.static_meta[1][1]}",
            "{a}": str(knock_tag),
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
        # Vegetables rest on the counter (not inside the upright bag).
        for item in self.static_items:
            p = np.asarray(item.get_pose().p, dtype=np.float64)
            if p[2] < self.table_top - 0.02 or p[2] > self.table_top + 0.22:
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
