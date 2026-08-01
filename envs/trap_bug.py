"""Trap a bug that scurries out from under the office wall-shelf.

Scene mirrors the basic office bench layout (`121_wall-shelf` at the back of
the table, from robotwin_bench office). A cockroach, spider, or ant emerges
from under the bookshelf and runs left or right. The robot must place a hollow
~50%-transparent glass trap box over the still-moving bug (never paused for the
place). On success the bug stops under the trap; trap collision stays on.
The bug's heading always matches its travel direction.
Success: the trap footprint covers the bug while resting on the table.
"""
from __future__ import annotations

import numpy as np
import sapien
import sapien.physx
import sapien.render
from transforms3d.euler import euler2quat

from ._office_base_task import Office_base_task
from .utils import *
from .utils.actor_utils import Actor
from ._GLOBAL_CONFIGS import *


class trap_bug(Office_base_task):
    BUG_SPEED_MIN = 0.07
    BUG_SPEED_MAX = 0.14
    TRAP_HALF = [0.045, 0.045, 0.028]
    TRAP_WALL = 0.004
    TRAP_MASS = 0.35             # graspable while held; immovable once anchored
    # Trap is released this far above its seated height, so the fingers clear
    # the rim instead of scraping the table.
    RELEASE_CLEARANCE = 0.05
    HOVER_LIFT = 0.10            # extra height while carrying to the intercept
    INTERCEPT_AHEAD = 0.14       # how far down the run line the trap is parked
    WAIT_STEPS = 400             # per attempt, waiting for the bug to arrive
    # Fraction of the gripper-open action that elapses before the box lets go
    # (measured on the aloha gripper: release at ~170 of 300 steps).
    RELEASE_FRACTION = 0.57
    RELEASE_LAG_STEPS = 170      # fallback when the gripper plan is unavailable
    EMERGE_STEPS = 40
    # Max heading slew rate (rad/s) so the bug turns into its travel direction
    # instead of snapping when velocity wobbles or walls bounce it.
    TURN_RATE = 10.0
    # Species catalog. Flat-on-table uses local Y → world Z.
    # `forward_y` is which way the mesh head points at yaw=0 (+1 → +Y, -1 → -Y).
    # All current bug meshes face -Y at yaw=0 after flattening, so travel yaw
    # is arctan2(vx, forward_y * vy) with forward_y=-1 (head always leads).
    BUG_SPECIES = {
        "cockroach": {
            "model": "200_cockroach",
            "label": "cockroach",
            "half_h": 0.003,
            "forward_y": -1.0,
            "mass": 0.005,
        },
        "spider": {
            "model": "201_spider",
            "label": "spider",
            "half_h": 0.008,
            "forward_y": -1.0,
            "mass": 0.005,
        },
        "ant": {
            "model": "202_ant",
            "label": "ant",
            "half_h": 0.004,
            "forward_y": -1.0,
            "mass": 0.004,
        },
    }
    DEFAULT_BUG_TYPES = ("cockroach", "spider", "ant")

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("trap_bug", {})
        # Init before _init_task_env_ — load_camera may call _update_kinematic_tasks.
        self._loaded = False
        self.bug = None
        self.trap = None
        self._bug_rigid = None
        self._bug_moving = False
        self._bug_vel = np.zeros(3, dtype=np.float64)
        self._bug_phase = 0.0
        self._bug_mode = "emerge"
        self._bug_yaw = 0.0
        self._bug_half_h = 0.003
        self._bug_forward_y = -1.0
        self.bug_type = "cockroach"
        self.bug_model = "200_cockroach"
        self.run_sign = 1
        self.arm_side = "right"
        self._trap_anchored = False
        self._trap_rigid = None
        self._trap_anchor_pose = None
        self._bug_captured = False
        self._sim_steps = 0
        super().setup_demo(**kwags)
        self._configure_observer_camera()

    def _configure_observer_camera(self):
        cams = getattr(self, "cameras", None)
        if cams is None or getattr(cams, "observer_camera", None) is None:
            return
        camera = cams.observer_camera
        camera_pos = np.array([0.42, 0.55, 1.50], dtype=np.float64)
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

    def _set_entity_pose(self, entity, pose):
        rigid = self._get_rigid(entity)
        if rigid is not None:
            try:
                rigid.set_kinematic_target(pose)
                return
            except Exception:
                pass
        obj = entity.actor if hasattr(entity, "actor") else entity
        obj.set_pose(pose)

    # -------------------------------------------------------------- glass trap
    def _make_glass_material(self):
        """Nearly clear glass (high transmission, low alpha) so the trapped bug stays visible."""
        glass = sapien.render.RenderMaterial(base_color=[0.88, 0.95, 0.98, 0.12])
        glass.set_transmission(0.92)
        glass.set_transmission_roughness(0.01)
        glass.set_roughness(0.03)
        glass.set_metallic(0.0)
        try:
            glass.set_ior(1.45)
        except Exception:
            glass.ior = 1.45
        return glass

    def _make_plain_trap_material(self):
        """Simple alpha-transparent plastic — no glass transmission/IOR (viewer-friendly)."""
        mat = sapien.render.RenderMaterial(base_color=[0.75, 0.88, 0.95, 0.35])
        mat.set_transmission(0.0)
        try:
            mat.set_transmission_roughness(1.0)
        except Exception:
            pass
        mat.set_roughness(0.45)
        mat.set_metallic(0.0)
        try:
            mat.set_ior(1.0)
        except Exception:
            mat.ior = 1.0
        return mat

    def _create_glass_trap(self, pose: sapien.Pose) -> Actor:
        """Hollow open-bottom square box (reverse box) with glass or plain visuals."""
        hx, hy, hz = [float(v) for v in self.trap_half]
        wt = float(self.trap_wall)
        scene = self.scene
        builder = scene.create_actor_builder()
        builder.set_physx_body_type("dynamic")

        top_hz = wt / 2.0
        side_hz = hz - top_hz
        side_z = -hz + side_hz
        parts = [
            (sapien.Pose([0, 0, hz - top_hz]), [hx, hy, top_hz]),
            (sapien.Pose([hx - wt / 2, 0, side_z]), [wt / 2, hy, side_hz]),
            (sapien.Pose([-hx + wt / 2, 0, side_z]), [wt / 2, hy, side_hz]),
            (sapien.Pose([0, hy - wt / 2, side_z]), [hx - wt, wt / 2, side_hz]),
            (sapien.Pose([0, -hy + wt / 2, side_z]), [hx - wt, wt / 2, side_hz]),
        ]
        # Grippy, dead-bounce glass so the released box lands where it was
        # dropped instead of skating off the bug.
        material = sapien.physx.PhysxMaterial(
            static_friction=1.2, dynamic_friction=1.0, restitution=0.0)
        for local_pose, half in parts:
            builder.add_box_collision(
                pose=local_pose,
                half_size=half,
                material=material,
            )

        builder.set_initial_pose(pose)
        entity = builder.build(name="glass_trap")

        # Interactive / viewer: plain alpha transparency. Expert demos: glass.
        if bool(getattr(self, "_plain_trap", False)):
            visual = self._make_plain_trap_material()
        else:
            visual = self._make_glass_material()
        render_body = sapien.render.RenderBodyComponent()
        for local_pose, half in parts:
            shape = sapien.render.RenderShapeBox(half, visual)
            shape.set_local_pose(local_pose)
            render_body.attach(shape)
        entity.add_component(render_body)

        data = {
            "center": [0, 0, 0],
            "extents": [hx * 2, hy * 2, hz * 2],
            "scale": [1, 1, 1],
            "target_pose": [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]]],
            "contact_points_pose": [
                [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
                [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
                [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, hz], [0, 0, 0, 1]],
            ],
            "transform_matrix": np.eye(4).tolist(),
            "functional_matrix": [
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, -hz],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            "contact_points_description": ["Top lid center, grasp from above."],
            "contact_points_group": [[0, 1, 2, 3]],
            "contact_points_mask": [True],
            "target_point_description": ["Bottom rim of the open trap."],
        }
        return Actor(entity, data)

    def _yaw_quat_flat(self, yaw: float) -> np.ndarray:
        """Flat on table (mesh local Y → world Z) + heading about world Z."""
        return np.asarray(euler2quat(np.pi / 2, 0, float(yaw), axes="sxyz"), dtype=np.float64)

    def _heading_yaw(self, vx: float, vy: float) -> float:
        """Yaw that points the bug's head along velocity (world XY).

        At yaw=0 the mesh head faces ``(0, forward_y)``. Rotating by yaw
        about world Z, the matching travel heading is
        ``arctan2(vx, forward_y * vy)``.
        """
        if abs(vx) + abs(vy) < 1e-6:
            vx, vy = float(self.run_sign), 0.0
        return float(np.arctan2(vx, self._bug_forward_y * vy))

    def _select_bug_species(self):
        c = self._cfg
        forced = c.get("bug_type") or c.get("species")
        choices = list(c.get("bug_types", self.DEFAULT_BUG_TYPES))
        if forced is not None:
            key = str(forced).lower().strip()
            if key not in self.BUG_SPECIES:
                raise ValueError(f"unknown trap_bug species: {forced}")
        else:
            key = str(np.random.choice(choices)).lower().strip()
            if key not in self.BUG_SPECIES:
                raise ValueError(f"unknown trap_bug species in bug_types: {key}")
        spec = self.BUG_SPECIES[key]
        self.bug_type = key
        self.bug_model = spec["model"]
        self._bug_half_h = float(spec["half_h"])
        self._bug_forward_y = float(np.sign(spec.get("forward_y", -1.0)) or -1.0)
        self._bug_mass = float(spec["mass"])
        return spec

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        c = self._cfg
        self.trap_half = list(c.get("trap_half", self.TRAP_HALF))
        self.trap_wall = float(c.get("trap_wall", self.TRAP_WALL))
        # plain_trap: opaque-ish alpha box (no glass transmission) for interactive viewers.
        self._plain_trap = bool(c.get("plain_trap", False))
        self.release_clearance = float(
            c.get("release_clearance", self.RELEASE_CLEARANCE))
        self.emerge_steps = int(c.get("emerge_steps", self.EMERGE_STEPS))
        # Accept legacy roach_speed_* keys from earlier configs.
        speed_min = float(c.get("bug_speed_min", c.get("roach_speed_min", self.BUG_SPEED_MIN)))
        speed_max = float(c.get("bug_speed_max", c.get("roach_speed_max", self.BUG_SPEED_MAX)))
        if speed_min <= 0 or speed_max < speed_min:
            raise ValueError("trap_bug bug_speed_min/max invalid")

        self._select_bug_species()

        self.table_top = float(self.office_info["table_height"])
        shelf_x = float(self.office_info["furn_x_v"]["shelf"][self.arr_v])
        shelf_front_y = float(self.shelf_lims[1])

        # Run outward from an offset shelf; center layout randomizes left/right.
        self.run_sign = (
            int(np.sign(shelf_x))
            if abs(shelf_x) > 1e-6
            else int(np.random.choice([-1, 1]))
        )
        self.arm_side = "right" if self.run_sign > 0 else "left"

        # Spawn under the bookshelf front lip, head already aimed along the run.
        spawn_x = float(np.clip(
            shelf_x + self.run_sign * np.random.uniform(0.02, 0.10),
            -0.28, 0.28,
        ))
        spawn_y = float(shelf_front_y + 0.01)
        spawn_z = self.table_top + self._bug_half_h
        self._bug_start = np.array([spawn_x, spawn_y, spawn_z], dtype=np.float64)
        # Initially facing out from under the shelf (emerge direction = -Y).
        self._bug_yaw = self._heading_yaw(0.0, -1.0)
        bug_q = self._yaw_quat_flat(self._bug_yaw)

        self.bug = create_actor(
            self,
            pose=sapien.Pose(p=self._bug_start.tolist(), q=bug_q.tolist()),
            modelname=self.bug_model,
            model_id=0,
            convex=True,
            is_static=False,
        )
        self.bug.set_mass(self._bug_mass)
        self._bug_rigid = self._make_kinematic(self.bug)
        self._set_entity_pose(
            self.bug, sapien.Pose(self._bug_start.tolist(), bug_q.tolist()))

        self._base_speed = float(np.random.uniform(speed_min, speed_max))
        self._bug_phase = 0.0
        self._bug_moving = False
        self._bug_mode = "emerge"
        self._emerge_target_y = float(np.clip(
            spawn_y - np.random.uniform(0.10, 0.16),
            -0.22, 0.12,
        ))
        # Stay on the arm's half after emerging so place stays reachable.
        if self.run_sign > 0:
            xmin, xmax = 0.02, 0.32
        else:
            xmin, xmax = -0.32, -0.02
        self._table_bounds = (xmin, xmax, -0.25, min(shelf_front_y - 0.02, 0.12))

        # Trap on the same side the bug will run toward.
        trap_x = float(np.clip(
            self.run_sign * np.random.uniform(0.16, 0.28),
            -0.30, 0.30,
        ))
        trap_y = float(np.random.uniform(-0.18, -0.05))
        trap_z = self.table_top + self.trap_half[2]
        self.trap = self._create_glass_trap(
            sapien.Pose(p=[trap_x, trap_y, trap_z], q=[1, 0, 0, 0]))
        self.trap_mass = float(c.get("trap_mass", self.TRAP_MASS))
        self.trap.set_mass(self.trap_mass)
        self._trap_rigid = None
        for comp in self.trap.actor.get_components():
            if isinstance(comp, sapien.physx.PhysxRigidDynamicComponent):
                self._trap_rigid = comp
                # Heavy + high damping so a light bug cannot shove it.
                comp.set_linear_damping(8.0)
                comp.set_angular_damping(8.0)
        self._trap_anchored = False

        self.add_prohibit_area(self.trap, padding=0.04)
        try:
            self.prohibited_area.append([
                self._table_bounds[0] - 0.02, self._emerge_target_y - 0.08,
                self._table_bounds[1] + 0.02, spawn_y + 0.04,
            ])
        except Exception:
            pass

        self._loaded = True

    # ---------------------------------------------------------- bug motion
    def _trap_near_table(self) -> bool:
        """True when seated/anchored (or rim nearly on the table)."""
        if self.trap is None:
            return False
        # Only block after the trap has been released and anchored — otherwise the
        # descending box would shove the kinematic bug out before cover.
        if self._trap_anchored:
            return True
        return False

    def _resolve_trap_collision(self, cur, new_p, vx, vy):
        """Block the kinematic bug from tunneling through the hollow glass trap.

        Only active once the trap is anchored on the table. Capture is decided
        at anchor time; free bugs cannot enter the trap footprint.
        """
        if self.trap is None or not self._trap_near_table() or self._bug_captured:
            return new_p, vx, vy

        tp = np.asarray(self.trap.get_pose().p, dtype=np.float64)
        hx, hy = float(self.trap_half[0]), float(self.trap_half[1])

        cx, cy = float(cur[0] - tp[0]), float(cur[1] - tp[1])
        nx, ny = float(new_p[0] - tp[0]), float(new_p[1] - tp[1])

        def outside(x, y):
            return abs(x) >= hx or abs(y) >= hy

        # Outside / approaching: reject any step into the footprint.
        if not outside(nx, ny):
            if abs(cx) >= hx and abs(nx) < hx:
                vx = -vx
                nx = float(np.sign(cx) * hx)
                self.run_sign = int(np.sign(vx)) if abs(vx) > 1e-6 else self.run_sign
            if abs(cy) >= hy and abs(ny) < hy:
                vy = -vy
                ny = float(np.sign(cy) * hy)
            if abs(nx) < hx and abs(ny) < hy:
                if abs(nx) / hx > abs(ny) / hy:
                    nx = float(np.sign(nx if nx != 0 else cx if cx != 0 else 1.0) * hx)
                    vx = -abs(vx) * np.sign(nx)
                else:
                    ny = float(np.sign(ny if ny != 0 else cy if cy != 0 else 1.0) * hy)
                    vy = -abs(vy) * np.sign(ny)
            new_p = np.array([tp[0] + nx, tp[1] + ny, new_p[2]], dtype=np.float64)

        return new_p, vx, vy

    def _anchor_trap(self):
        """Freeze the trap where it landed so the bug cannot shove it."""
        if self.trap is None or self._trap_anchored:
            return
        # Keep the landed XY (no teleporting onto the bug); just square the
        # box up on the table so a bounced landing still seals.
        pose = self.trap.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        p[2] = self.table_top + self.trap_half[2]
        seated = sapien.Pose(p=p.tolist(), q=[1, 0, 0, 0])
        obj = self.trap.actor if hasattr(self.trap, "actor") else self.trap
        rigid = self._trap_rigid or self._get_rigid(self.trap)
        if rigid is not None:
            try:
                rigid.set_linear_velocity(np.zeros(3))
                rigid.set_angular_velocity(np.zeros(3))
                rigid.set_kinematic(True)
                rigid.set_kinematic_target(seated)
            except Exception:
                pass
        obj.set_pose(seated)
        self._trap_rigid = rigid
        self._trap_anchor_pose = seated
        self._trap_anchored = True
        # The landing box either swept the bug into the cavity or missed it;
        # that verdict is fixed here. On success, freeze the bug so it cannot
        # shove the (still-collidable) trap around.
        if self.bug is not None and self._bug_rigid is not None:
            rp = np.asarray(self.bug.get_pose().p, dtype=np.float64)
            if (abs(rp[0] - p[0]) < self.trap_half[0]
                    and abs(rp[1] - p[1]) < self.trap_half[1]):
                self._bug_captured = True
                ix = max(self.trap_half[0] - self.trap_wall - 0.002, 0.005)
                iy = max(self.trap_half[1] - self.trap_wall - 0.002, 0.005)
                rp[0] = float(np.clip(rp[0], p[0] - ix, p[0] + ix))
                rp[1] = float(np.clip(rp[1], p[1] - iy, p[1] + iy))
                rp[2] = float(self.table_top + self._bug_half_h)
                q = list(self.bug.get_pose().q)
                self._bug_rigid.set_kinematic_target(
                    sapien.Pose(p=rp.tolist(), q=q))
                self._stop_bug()

    def _stop_bug(self):
        """Halt scuttle once trapped — keeps PhysX trap collision intact."""
        self._bug_moving = False
        self._bug_vel = np.zeros(3, dtype=np.float64)
        if self.bug is None or self._bug_rigid is None:
            return
        pose = self.bug.get_pose()
        p = np.asarray(pose.p, dtype=np.float64)
        p[2] = float(self.table_top + self._bug_half_h)
        try:
            self._bug_rigid.set_kinematic_target(
                sapien.Pose(p.tolist(), list(pose.q)))
        except Exception:
            pass

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._sim_steps += 1
        # Keep an anchored trap pinned every step (collector two-pass safe).
        if self._trap_anchored and self._trap_anchor_pose is not None:
            if self._trap_rigid is not None:
                try:
                    self._trap_rigid.set_kinematic_target(self._trap_anchor_pose)
                except Exception:
                    pass
        # Interactive / late cover: stop as soon as success holds.
        if (
            self._bug_moving
            and self._trap_anchored
            and not self._bug_captured
            and self.check_success()
        ):
            self._bug_captured = True
            self._stop_bug()
        if not self._bug_moving or self._bug_rigid is None:
            return

        dt = float(self.scene.get_timestep())
        self._bug_phase += dt
        pose = self._bug_rigid.entity.get_pose()
        p = np.array(pose.p, dtype=np.float64)

        # Bursty scuttle + light lateral wobble (realistic, never paused for the place).
        pulse = 0.55 + 0.45 * abs(np.sin(self._bug_phase * 10.0))
        speed = self._base_speed * pulse
        wobble = 0.015 * np.sin(self._bug_phase * 14.0)

        if self._bug_mode == "emerge":
            vy = -speed
            vx = wobble * 0.5 * self.run_sign
            if p[1] <= self._emerge_target_y:
                self._bug_mode = "run"
        else:
            vx = self.run_sign * speed
            vy = -0.15 * speed + wobble

        xmin, xmax, ymin, ymax = self._table_bounds
        if p[0] + vx * dt < xmin or p[0] + vx * dt > xmax:
            self.run_sign *= -1
            vx = self.run_sign * abs(vx)
        if p[1] + vy * dt < ymin:
            vy = abs(speed) * 0.25
        if p[1] + vy * dt > ymax:
            vy = -abs(speed) * 0.3

        new_p = p + np.array([vx, vy, 0.0], dtype=np.float64) * dt
        new_p, vx, vy = self._resolve_trap_collision(p, new_p, vx, vy)
        new_p[0] = float(np.clip(new_p[0], xmin, xmax))
        new_p[1] = float(np.clip(new_p[1], ymin, ymax))
        new_p[2] = float(self.table_top + self._bug_half_h)

        self._bug_vel = np.array([vx, vy, 0.0], dtype=np.float64)
        # Slew the head toward the travel direction (never leave the body
        # sideways relative to velocity).
        target_yaw = self._heading_yaw(vx, vy)
        dyaw = (target_yaw - self._bug_yaw + np.pi) % (2.0 * np.pi) - np.pi
        max_turn = float(self.TURN_RATE) * dt
        self._bug_yaw = float(self._bug_yaw + float(np.clip(dyaw, -max_turn, max_turn)))
        q = self._yaw_quat_flat(self._bug_yaw)
        self._bug_rigid.set_kinematic_target(
            sapien.Pose(p=new_p.tolist(), q=q.tolist()))

    def check_stable(self):
        """Keep the kinematic bug parked under the shelf while the scene settles."""
        if self.bug is not None and not self._bug_moving:
            self._bug_yaw = self._heading_yaw(0.0, -1.0)
            q = self._yaw_quat_flat(self._bug_yaw)
            self._set_entity_pose(
                self.bug, sapien.Pose(self._bug_start.tolist(), q.tolist()))
        return super().check_stable()

    def _dwell(self, steps):
        # Counter is episode-global so single-step waits do not capture a frame
        # per physics tick.
        for _ in range(int(steps)):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (self._sim_steps % self.save_freq == 0):
                self._take_picture()

    def _start_bug(self):
        self._bug_moving = True

    # ------------------------------------------------------------- policy
    def _intercept_point(self, arm_tag) -> np.ndarray:
        """A spot on the bug's run line where the trap waits for it.

        The lane (y) is where the bug will have drifted to by the time it
        reaches the intercept, not where it is now — it sidles forward the
        whole way there.
        """
        rp = np.array(self.bug.get_pose().p, dtype=np.float64)
        xmin, xmax, ymin, ymax = self._table_bounds
        speed = max(self._base_speed * 0.836, 1e-3)
        ahead = rp[0] + self.run_sign * self.INTERCEPT_AHEAD
        if not (xmin + 0.04 <= ahead <= xmax - 0.04):
            # Too close to a wall to get there first; the bug bounces back.
            ahead = rp[0] - self.run_sign * self.INTERCEPT_AHEAD
        x = float(np.clip(ahead, xmin + 0.04, xmax - 0.04))
        t_cross = abs(x - rp[0]) / speed + self._release_delay(arm_tag)
        y = float(np.clip(rp[1] - 0.15 * speed * t_cross,
                          ymin + 0.03, min(ymax, 0.05)))
        return np.array([x, y], dtype=np.float64)

    def _fall_time(self) -> float:
        return float(np.sqrt(max(2.0 * self.release_clearance / 9.81, 0.0)))

    def _release_delay(self, arm_tag) -> float:
        """Seconds from commanding the gripper open until the trap lands.

        The gripper open is a dense action several hundred sim steps long and
        the box only drops partway through it, so the bug covers real ground
        between the command and the catch.
        """
        steps = float(self.RELEASE_LAG_STEPS)
        try:
            plan_fn = (self.robot.left_plan_grippers if arm_tag == "left"
                       else self.robot.right_plan_grippers)
            cur = (self.robot.get_left_gripper_val() if arm_tag == "left"
                   else self.robot.get_right_gripper_val())
            n = float(plan_fn(cur, 1.0)["num_step"]) * 1.5  # set_gripper pads by 50%
            steps = n * self.RELEASE_FRACTION
        except Exception:
            pass
        return steps * float(self.scene.get_timestep()) + self._fall_time()

    def _predict_arrival(self, t: float) -> np.ndarray:
        """Where the bug will be in ``t`` seconds, reflecting off the bounds.

        Uses the mean scuttle speed rather than the instantaneous burst so a
        long lead is not thrown off by the gait pulse.
        """
        rp = np.array(self.bug.get_pose().p, dtype=np.float64)
        xmin, xmax, ymin, ymax = self._table_bounds
        speed = self._base_speed * 0.836  # mean of the 0.55..1.0 pulse
        x = rp[0] + self.run_sign * speed * t
        span = xmax - xmin
        if span > 1e-6:
            u = (x - xmin) % (2.0 * span)
            x = xmin + (u if u <= span else 2.0 * span - u)
        y = float(np.clip(rp[1] - 0.15 * speed * t, ymin, ymax))
        return np.array([x, y], dtype=np.float64)

    def _wait_for_bug_under(self, arm_tag, x_tol=0.012, y_tol=0.024) -> bool:
        """Hold the trap still and let the bug scuttle under it.

        The bug is never paused; we step the sim until the spot it will
        occupy once the released trap lands is centred under the trap. Gives
        up early once the bug has sidled out of the lane, so the caller can
        re-aim instead of burning the whole window.
        """
        for _ in range(int(self.WAIT_STEPS)):
            tp = np.array(self.trap.get_pose().p, dtype=np.float64)
            arrival = self._predict_arrival(self._release_delay(arm_tag))
            if abs(arrival[1] - tp[1]) >= y_tol:
                return False
            if abs(arrival[0] - tp[0]) < x_tol:
                return True
            self._dwell(1)
        return False

    def _reaim_lane(self, arm_tag) -> None:
        """Nudge the held trap back onto the bug's lane (y drifts slowly)."""
        tp = np.array(self.trap.get_pose().p, dtype=np.float64)
        target = self._intercept_point(arm_tag)
        dy = float(np.clip(target[1] - tp[1], -0.08, 0.08))
        dx = float(np.clip(target[0] - tp[0], -0.08, 0.08))
        if abs(dy) < 0.004 and abs(dx) < 0.02:
            return
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, x=dx, y=dy, move_axis="world"))

    def play_once(self):
        # Start scurrying immediately — never pause for the place.
        self._start_bug()
        self._dwell(self.emerge_steps)

        arm_tag = ArmTag(self.arm_side)

        self.move(self.grasp_actor(
            self.trap, arm_tag=arm_tag, pre_grasp_dis=0.08, contact_point_id=[0, 1, 2, 3]))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))

        # Carry the trap high over an intercept spot on the bug's run line.
        # Displacements are measured on the trap itself, so the grasp offset
        # does not bias the alignment.
        target = self._intercept_point(arm_tag)
        tp = np.array(self.trap.get_pose().p, dtype=np.float64)
        hover_z = self.table_top + self.trap_half[2] + self.release_clearance + self.HOVER_LIFT
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=float(target[0] - tp[0]),
            y=float(target[1] - tp[1]),
            z=float(hover_z - tp[2]),
            move_axis="world",
        ))

        # Drop from a raised hover: high enough for the fingers to clear the rim.
        drop_z = self.table_top + self.trap_half[2] + self.release_clearance
        tp = np.array(self.trap.get_pose().p, dtype=np.float64)
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=float(drop_z - tp[2]), move_axis="world"))

        # Poised over the lane: wait for the bug, re-aiming if it drifts.
        for _ in range(8):
            if self._wait_for_bug_under(arm_tag):
                break
            self._reaim_lane(arm_tag)

        self.move(self.open_gripper(arm_tag=arm_tag))
        # Let the trap fall and settle over the bug before the arm clears out.
        self._dwell(30)
        self._anchor_trap()
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="world"))

        # Hold on success (bug stopped under the trap) or a miss.
        self._dwell(60)

        self.info["info"] = {
            "{A}": f"{self.bug_model}/base0",
            "{S}": self.bug_type,
            "{B}": "glass_trap",
            "{C}": "121_wall-shelf",
            "{a}": str(arm_tag),
            "{d}": "left" if self.run_sign < 0 else "right",
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        """Success = trap dropped onto the table with the bug under its footprint.

        Hovering the box over the bug (still held / in the air) must not count.
        Anchoring only happens after a release that seats on the table.
        """
        if self.trap is None or self.bug is None:
            return False
        if not bool(getattr(self, "_trap_anchored", False)):
            return False
        trap_p = np.array(self.trap.get_pose().p, dtype=np.float64)
        bug_p = np.array(self.bug.get_pose().p, dtype=np.float64)
        seated_z = float(self.table_top + self.trap_half[2])
        # Tight band around the seated rim height (not release-clearance hover).
        trap_on_table = abs(float(trap_p[2]) - seated_z) <= 0.012
        covered = (
            abs(bug_p[0] - trap_p[0]) < self.trap_half[0] * 0.95
            and abs(bug_p[1] - trap_p[1]) < self.trap_half[1] * 0.95
        )
        return bool(trap_on_table and covered)

    def get_obs(self):
        obs = super().get_obs()
        bug_p = self.bug.get_pose().p if self.bug is not None else [0, 0, 0]
        trap_p = self.trap.get_pose().p if self.trap is not None else [0, 0, 0]
        obs["trap_bug"] = {
            "bug_type": str(self.bug_type),
            "bug_model": str(self.bug_model),
            "bug_pos": [float(x) for x in bug_p],
            "trap_pos": [float(x) for x in trap_p],
            "bug_vel": [float(x) for x in self._bug_vel],
            "bug_yaw": float(self._bug_yaw),
            "run_dir": "left" if self.run_sign < 0 else "right",
            "bug_moving": bool(self._bug_moving),
            "bug_mode": str(self._bug_mode),
        }
        return obs
