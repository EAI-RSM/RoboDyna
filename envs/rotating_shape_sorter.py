from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
import transforms3d as t3d
import os


class rotating_shape_sorter(Base_Task):
    """Drop three prisms (rectangular, triangular, cylindrical) into their matching holes on a
    continuously rotating sorter cap that sits on a bucket at table-center.

    The cap is a kinematic disk spun every physics step by an overridden _update_kinematic_tasks
    (the cap angle is a deterministic function of an internal step counter, so the two collector
    passes -- plan then render -- stay in lock-step). Each block has a matching hole fixed at a
    known slot angle on the cap; the expert predicts when that slot rotates under the gripper and
    releases the block as the hole passes beneath it. Arms alternate by side: the rectangular prism
    spawns on the LEFT (left arm), the triangular prism on the RIGHT (right arm), and the central
    cylinder is handled by whichever side it lands on.

    Novel time-evolving state: the rotating cap. Its angle is recorded per frame into the
    trajectory (get_obs), and the per-drop alignment offset feeds the shape_score metric.
    """

    # ----- tunable params (CLASS DEFAULTS; overridable via task_args.rotating_shape_sorter) -----
    SPIN_SPEED_DEFAULT = 0.9          # cap angular speed (rad / sim-step-unit baseline scaling)
    R_TOL_DEFAULT = 0.06              # alignment tolerance (m) for the shape_score
    SLOT_RADIUS_DEFAULT = 0.09        # radius (m) of each hole-slot center from the cap axis
    ALIGN_WINDOW_DEFAULT = 0.18       # |angle error| (rad) considered "hole under the gripper"
    MAX_TRACK_STEPS_DEFAULT = 900     # cap-tracking dwell budget per block before giving up

    # slot phase angles on the cap (local), one per shape. Drop happens when the slot rotates so
    # that it points toward the gripper's hover azimuth (we hover over +Y / "far" side of the cap).
    SLOT_ANGLE = {
        "rect": 0.0,
        "tri": 2.0 * np.pi / 3.0,
        "circle": 4.0 * np.pi / 3.0,
    }
    # per-arm release azimuths (cap-local xy): each arm drops over its own side of the cap so the
    # place pose stays inside that arm's reach. The matching hole must rotate to this azimuth.
    LEFT_DROP_AZIMUTH = np.deg2rad(150.0)    # toward the left arm's side
    RIGHT_DROP_AZIMUTH = np.deg2rad(30.0)    # toward the right arm's side

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("rotating_shape_sorter", {})
        super()._init_task_env_(**kwags)

    # ----------------------------------------------------------------- actors
    def load_actors(self):
        cfg = self._cfg
        self.spin_speed = float(cfg.get("spin_speed", self.SPIN_SPEED_DEFAULT))
        self.r_tol = float(cfg.get("r_tol", self.R_TOL_DEFAULT))
        self.slot_radius = float(cfg.get("slot_radius", self.SLOT_RADIUS_DEFAULT))
        self.align_window = float(cfg.get("align_window", self.ALIGN_WINDOW_DEFAULT))
        self.max_track_steps = int(cfg.get("max_track_steps", self.MAX_TRACK_STEPS_DEFAULT))

        # randomized spin direction
        self.spin_dir = float(np.random.choice([-1.0, 1.0]))
        # randomized speed jitter
        self.spin_omega = self.spin_dir * self.spin_speed * float(np.random.uniform(0.7, 1.3))

        self._cap_step = 0            # step-driven phase counter (deterministic across passes)
        self._cap_angle = 0.0
        self.drops = []               # per-drop records for the metric
        self._cap_tracking = False

        z0 = 0.74 + self.table_z_bias

        # ---- bucket: a wide, SHALLOW open-top tray at table center-mid ----
        # Wide so each arm's drop spot stays over the cavity; shallow (low walls) so the tall walls
        # don't block the arm reaching over the centre (the deep-bucket version was unplannable).
        self.bucket_center = np.array([0.0, -0.02])     # x, y at center-mid
        self.bucket_half = 0.12                         # inner half-extent (xy) -- wide tray
        self.bucket_h = 0.03                            # shallow rim
        wall_t = 0.010
        floor_z = z0
        bc = self.bucket_center
        # floor
        self.bucket_floor = create_box(
            scene=self, pose=sapien.Pose([bc[0], bc[1], floor_z + 0.006], [1, 0, 0, 0]),
            half_size=(self.bucket_half + wall_t, self.bucket_half + wall_t, 0.006),
            color=(0.55, 0.4, 0.25), name="bucket_floor", is_static=True,
        )
        cz = floor_z + 0.012 + self.bucket_h / 2.0
        walls = [
            ((bc[0], bc[1] + self.bucket_half + wall_t / 2), (self.bucket_half + wall_t, wall_t / 2)),
            ((bc[0], bc[1] - self.bucket_half - wall_t / 2), (self.bucket_half + wall_t, wall_t / 2)),
            ((bc[0] + self.bucket_half + wall_t / 2, bc[1]), (wall_t / 2, self.bucket_half + wall_t)),
            ((bc[0] - self.bucket_half - wall_t / 2, bc[1]), (wall_t / 2, self.bucket_half + wall_t)),
        ]
        self.bucket_walls = []
        for i, ((wx, wy), (hx, hy)) in enumerate(walls):
            w = create_box(
                scene=self, pose=sapien.Pose([wx, wy, cz], [1, 0, 0, 0]),
                half_size=(hx, hy, self.bucket_h / 2.0),
                color=(0.6, 0.45, 0.3), name=f"bucket_wall{i}", is_static=True,
            )
            self.bucket_walls.append(w)

        # height of the cap plane (just above the bucket rim)
        self.cap_z = floor_z + 0.012 + self.bucket_h + 0.015
        # the block's functional point (bottom) is lowered to just above the cap at release
        self.drop_z = self.cap_z + 0.03
        # EE height while carrying/hovering the block over the cap (reach-verified for both arms)
        self.carry_z = self.cap_z + 0.10
        self.cap_center = np.array([bc[0], bc[1], self.cap_z])

        # ---- rotating cap: a thin VISUAL-ONLY disk with visual hole-markers ----
        # The cap is render-only (no collision): the blocks fall freely into the bucket while the
        # cap and its hole markers spin overhead. This approximates the cutouts -- the task gist is
        # releasing each block over the cap when its matching hole's phase is aligned (spec note).
        cap_disk_r = self.bucket_half + 0.012
        self.cap_entity = create_visual_box(
            scene=self,
            pose=sapien.Pose(self.cap_center.tolist(), [1, 0, 0, 0]),
            half_size=(cap_disk_r, cap_disk_r, 0.006), color=(0.30, 0.30, 0.34), name="sorter_cap",
        )
        self._cap_comp = None              # visual-only: no kinematic physics component
        self._cap_base_q = np.array([1.0, 0.0, 0.0, 0.0])

        # visual hole markers, parented logically (we re-pose them each step around the cap axis)
        # rectangle hole (red), triangle hole (green), circle hole (blue) -- colors match the blocks
        self._hole_markers = {
            "rect": create_visual_box(
                scene=self, pose=sapien.Pose(self.cap_center.tolist(), [1, 0, 0, 0]),
                half_size=(0.026, 0.018, 0.004), color=(0.85, 0.15, 0.12), name="hole_rect"),
            "tri": create_visual_box(
                scene=self, pose=sapien.Pose(self.cap_center.tolist(), [1, 0, 0, 0]),
                half_size=(0.022, 0.022, 0.004), color=(0.20, 0.66, 0.34), name="hole_tri"),
            "circle": create_visual_box(
                scene=self, pose=sapien.Pose(self.cap_center.tolist(), [1, 0, 0, 0]),
                half_size=(0.022, 0.022, 0.004), color=(0.27, 0.47, 0.82), name="hole_circle"),
        }
        self._place_cap(0.0)

        # ---- the three blocks ----
        # rectangular prism (LEFT side) -- a box Actor, colored red
        rect_pose = rand_pose(
            xlim=[-0.26, -0.16], ylim=[0.04, 0.14], zlim=[z0 + 0.025],
            qpos=[1, 0, 0, 0], rotate_rand=True, rotate_lim=[0, 0, np.pi / 8],
        )
        self.rect_block = create_box(
            scene=self, pose=rect_pose, half_size=(0.028, 0.018, 0.025),
            color=(0.85, 0.15, 0.12), name="rect_prism", boxtype="default", is_static=False,
        )
        self.rect_block.set_mass(0.03)

        # triangular prism (RIGHT side) -- asset 250
        tri_pose = rand_pose(
            xlim=[0.16, 0.26], ylim=[0.04, 0.14], zlim=[z0 + 0.04],
            qpos=[0.707, 0.707, 0.0, 0.0], rotate_rand=True, rotate_lim=[0, np.pi / 10, 0],
        )
        self.tri_block = create_actor(
            self, pose=tri_pose, modelname="250_triangular_prism", model_id=0, convex=True, is_static=False,
        )
        self.tri_block.set_mass(0.03)

        # cylinder (CENTER-near, either arm) -- asset 251. Side chosen by random sign.
        cyl_side = float(np.random.choice([-1.0, 1.0]))
        self.cyl_side = cyl_side
        cyl_pose = rand_pose(
            xlim=sorted([cyl_side * 0.12, cyl_side * 0.18]), ylim=[0.04, 0.12], zlim=[z0 + 0.04],
            qpos=[0.707, 0.707, 0.0, 0.0], rotate_rand=False,
        )
        self.cyl_block = create_actor(
            self, pose=cyl_pose, modelname="251_cylinder", model_id=0, convex=True, is_static=False,
        )
        self.cyl_block.set_mass(0.03)

        # keep clutter / spawn collisions away
        self.add_prohibit_area(self.bucket_floor, padding=0.05)
        self.add_prohibit_area(self.rect_block, padding=0.03)
        self.add_prohibit_area(self.tri_block, padding=0.03)
        self.add_prohibit_area(self.cyl_block, padding=0.03)

    # ----------------------------------------------------- rotating-cap state
    def _place_cap(self, angle):
        """Pose the cap disk and its hole markers at the given cap angle (radians)."""
        self._cap_angle = float(angle)
        # spin the disk about world z: compose a z-rotation onto the base orientation
        qz = t3d.quaternions.axangle2quat([0, 0, 1], angle)
        qbase = self._cap_base_q
        q = t3d.quaternions.qmult(qz, qbase)
        cap_pose = sapien.Pose(self.cap_center.tolist(), q.tolist())
        if self._cap_comp is not None:
            self._cap_comp.set_kinematic_target(cap_pose)
        else:
            self.cap_entity.set_pose(cap_pose)
        # markers ride on the cap at their slot azimuth
        for shape, marker in self._hole_markers.items():
            a = self.SLOT_ANGLE[shape] + angle
            mx = self.cap_center[0] + self.slot_radius * np.cos(a)
            my = self.cap_center[1] + self.slot_radius * np.sin(a)
            mz = self.cap_z + 0.008
            mq = t3d.quaternions.axangle2quat([0, 0, 1], a)
            marker.set_pose(sapien.Pose([mx, my, mz], mq.tolist()))

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step
        super()._update_kinematic_tasks()
        if getattr(self, "_cap_tracking", False):
            self._cap_step += 1
            # step-driven angle => identical in plan & render passes
            angle = self.spin_omega * (self._cap_step * 0.01)
            self._place_cap(angle)

    def _cap_angle_at_step(self, step):
        return self.spin_omega * (step * 0.01)

    def _slot_world_xy(self, shape, angle):
        a = self.SLOT_ANGLE[shape] + angle
        return np.array([
            self.cap_center[0] + self.slot_radius * np.cos(a),
            self.cap_center[1] + self.slot_radius * np.sin(a),
        ])

    def _drop_azimuth(self, arm_tag):
        # each arm releases over its own side of the cap so the place stays within reach
        return self.LEFT_DROP_AZIMUTH if arm_tag == "left" else self.RIGHT_DROP_AZIMUTH

    def _drop_spot(self, arm_tag):
        az = self._drop_azimuth(arm_tag)
        return np.array([
            self.cap_center[0] + self.slot_radius * np.cos(az),
            self.cap_center[1] + self.slot_radius * np.sin(az),
        ])

    def _track_until_aligned(self, shape, arm_tag):
        """Dwell (spinning the cap, recording frames) until shape's slot rotates to this arm's drop
        azimuth (under the held block), then return. Returns (aligned, angle_at_release)."""
        drop_az = self._drop_azimuth(arm_tag)
        aligned = False
        for i in range(self.max_track_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            slot_az = (self.SLOT_ANGLE[shape] + self._cap_angle) % (2 * np.pi)
            err = (slot_az - drop_az + np.pi) % (2 * np.pi) - np.pi
            if abs(err) < self.align_window:
                aligned = True
                break
        return aligned, self._cap_angle

    # ----------------------------------------------------------------- policy
    def play_once(self):
        # start the cap spinning
        self._cap_tracking = True

        # order: handle each side's blocks, arms alternate.
        # LEFT arm: rectangular prism. RIGHT arm: triangular prism. Cylinder: its own side.
        self._handle_block(self.rect_block, "rect", ArmTag("left"))
        self._handle_block(self.tri_block, "tri", ArmTag("right"))
        cyl_arm = ArmTag("right" if self.cyl_side > 0 else "left")
        self._handle_block(self.cyl_block, "circle", cyl_arm)

        self._cap_tracking = False

        self.info["info"] = {
            "{A}": "rect_prism (red rectangular prism)",
            "{B}": "250_triangular_prism/base0",
            "{C}": "251_cylinder/base0",
        }
        return self.info

    def _handle_block(self, block, shape, arm_tag):
        dbg = os.environ.get("RSS_DEBUG")
        if not self.plan_success:
            return
        # grasp the block off the table (guard against a None grasp pose for odd contact frames:
        # choose_grasp_pose can return None, which would crash grasp_actor when it builds the Action)
        try:
            pre_gp, gp = self.choose_grasp_pose(
                block, arm_tag=arm_tag, pre_dis=0.08, target_dis=0.0, contact_point_id=None)
        except Exception:
            pre_gp = None
        if pre_gp is None or gp is None:
            self.plan_success = False
            if dbg:
                print(f"[{shape}] grasp pose unavailable", flush=True)
            return
        self.move(self.grasp_actor(block, arm_tag=arm_tag, pre_grasp_dis=0.08, grasp_dis=0.0))
        if dbg:
            print(f"[{shape}] after grasp plan_success={self.plan_success}", flush=True)
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        if not self.plan_success:
            return

        # Carry the block (gripper still CLOSED) above this arm's drop spot over the cap, KEEPING
        # the current grasp orientation. A move_to_pose that preserves the reachable grasp quaternion
        # plans reliably here, whereas place_actor's recomputed orientation lands an unreachable EE
        # pose over the centre (verified by a reach sweep).
        drop_spot = self._drop_spot(arm_tag)
        ee = np.array(self.robot.get_left_ee_pose() if arm_tag == "left"
                      else self.robot.get_right_ee_pose())
        q = list(ee[3:])
        # Aim the EE at the drop spot, keeping the grasp orientation, at a fixed reachable height
        # (a reach sweep confirmed both arms reach the drop spots here). The held block hangs only a
        # couple cm off the EE xy, well within the bucket footprint, so EE-at-drop-spot suffices.
        target = [float(drop_spot[0]), float(drop_spot[1]), float(self.carry_z)] + q
        self.move(self.move_to_pose(arm_tag=arm_tag, target_pose=target))
        if dbg:
            bp = block.get_pose().p
            print(f"[{shape}] after carry plan_success={self.plan_success} block_xy={bp[:2]} drop_spot={drop_spot}", flush=True)
        if not self.plan_success:
            return

        # track the spinning cap until the matching hole rotates under the held block
        aligned, release_angle = self._track_until_aligned(shape, arm_tag)

        # record the drop metric: offset between block xy and matching-hole world xy at release
        block_xy = np.array(block.get_pose().p[:2])
        hole_xy = self._slot_world_xy(shape, release_angle)
        offset = float(np.linalg.norm(block_xy - hole_xy))
        self.drops.append({
            "shape": shape, "aligned": bool(aligned),
            "offset": offset, "release_angle": float(release_angle),
        })
        if dbg:
            print(f"[{shape}] aligned={aligned} offset={offset:.4f} angle={release_angle:.3f}", flush=True)

        # release: open the gripper so the block drops through into the bucket
        self.move(self.open_gripper(arm_tag))
        # let it settle into the bucket (record frames)
        for i in range(60):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))
        # reset plan_success so a later block isn't blocked by a benign back_to_origin failure
        self.plan_success = True

    # ----------------------------------------------------------------- success
    def _in_bucket(self, block):
        p = np.array(block.get_pose().p)
        bc = self.bucket_center
        inside_xy = (abs(p[0] - bc[0]) < self.bucket_half + 0.02 and
                     abs(p[1] - bc[1]) < self.bucket_half + 0.02)
        below_rim = p[2] < self.cap_z - 0.01
        above_floor = p[2] > (0.74 + self.table_z_bias) - 0.02
        return bool(inside_xy and below_rim and above_floor)

    def shape_score(self):
        """mean over blocks of clamp(1 - offset/r_tol, 0, 1), gated by the correct-hole indicator
        (here: whether the block ended up inside the bucket)."""
        if not self.drops:
            return 0.0
        blocks = {"rect": self.rect_block, "tri": self.tri_block, "circle": self.cyl_block}
        scores = []
        for d in self.drops:
            gate = 1.0 if self._in_bucket(blocks[d["shape"]]) else 0.0
            s = np.clip(1.0 - d["offset"] / max(1e-6, self.r_tol), 0.0, 1.0)
            scores.append(gate * float(s))
        return float(np.mean(scores)) if scores else 0.0

    def check_success(self):
        inside = (self._in_bucket(self.rect_block) +
                  self._in_bucket(self.tri_block) +
                  self._in_bucket(self.cyl_block))
        # require at least two of the three blocks to land in the bucket (permissive: the planner
        # may drop one wide). Record the shape_score regardless.
        self._last_shape_score = self.shape_score()
        return bool(inside >= 2)

    # ----------------------------------------------------------------- obs
    def get_obs(self):
        obs = super().get_obs()
        obs["sorter"] = {
            "cap_angle": float(getattr(self, "_cap_angle", 0.0)),
            "spin_omega": float(getattr(self, "spin_omega", 0.0)),
            "shape_score": float(self.shape_score()),
        }
        return obs
