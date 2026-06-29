from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.physx
import numpy as np


class assemble_markers_cylinder(Base_Task):
    """Dual-arm assembly: attach four markers to a short, static, vertical magnetic
    cylinder so they end up evenly spaced (90 deg apart) around it.

    Two markers spawn in the LEFT half of the table and two in the RIGHT half; each arm
    handles only the markers on its own side (hard reach limit). The cylinder sits at the
    table centre and is reachable by both arms. The four attach points are at 45/135/225/315
    degrees around the vertical axis -- the +x pair (45,315) is pressed in by the RIGHT arm,
    the -x pair (135,225) by the LEFT arm, so neither arm ever crosses the centreline.

    "Magnetic attach" = when a held marker is brought to within a small distance of its
    target attach point on the cylinder side, a fixed attachment is formed by switching the
    marker's rigid body to KINEMATIC and locking its pose to the cylinder each physics step
    (the cylinder is static, so the relative transform never changes). The marker's final
    yaw about the cylinder axis is recorded; success requires all four attached, and the
    evenness of the four yaw gaps is reported as a metric.
    """

    # ----- class-default params (overridable via task_args.assemble_markers_cylinder) -----
    CYL_RADIUS_DEFAULT = 0.035          # cylinder radius (m)
    CYL_HALF_LEN_DEFAULT = 0.075        # half height (m) -> 0.15 m tall (<= 0.4 m)
    ATTACH_DIST_DEFAULT = 0.06          # marker-tip -> surface threshold to trigger attach (m)
    ATTACH_Z_DEFAULT = 0.04             # height above cylinder base centre where markers ride
    DWELL_STEPS_DEFAULT = 40            # physics steps to settle/record after each attach press
    # target attach yaws around the vertical axis (deg). 90-deg spaced; +x pair vs -x pair.
    TARGET_YAWS_DEG = [45.0, 315.0, 135.0, 225.0]

    def setup_demo(self, **kwags):
        self._cfg = kwags.get("task_args", {}).get("assemble_markers_cylinder", {})
        super()._init_task_env_(**kwags)

    # ------------------------------------------------------------------ actors
    def load_actors(self):
        cfg = self._cfg
        self.cyl_radius = float(cfg.get("cyl_radius", self.CYL_RADIUS_DEFAULT))
        self.cyl_half_len = float(cfg.get("cyl_half_len", self.CYL_HALF_LEN_DEFAULT))
        self.attach_dist = float(cfg.get("attach_dist", self.ATTACH_DIST_DEFAULT))
        self.attach_z = float(cfg.get("attach_z", self.ATTACH_Z_DEFAULT))
        self.dwell_steps = int(cfg.get("dwell_steps", self.DWELL_STEPS_DEFAULT))

        table_z = 0.74 + self.table_z_bias

        # ---- the static magnetic cylinder, standing upright at the table centre ----
        # create_cylinder builds it with its length along local X; rotate so length -> world Z.
        self.cyl_yaw = float(np.random.uniform(-np.pi, np.pi))   # randomized cylinder yaw
        self.cyl_center_z = table_z + self.cyl_half_len
        # quat that maps local X -> world Z (rotate -90 deg about world Y), then any yaw is
        # irrelevant for a circular cylinder, so we just stand it up.
        upright_q = [0.7071068, 0.0, 0.7071068, 0.0]
        cyl_pose = sapien.Pose([0.0, 0.0, self.cyl_center_z], upright_q)
        self.cylinder = create_cylinder(
            scene=self, pose=cyl_pose,
            radius=self.cyl_radius, half_length=self.cyl_half_len,
            color=(0.55, 0.55, 0.6), name="magnet_cylinder",
        )
        # make it immovable (static) by switching its rigid body to kinematic
        for c in self.cylinder.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                c.set_kinematic(True)
        self.cyl_axis_xy = np.array([0.0, 0.0])   # cylinder axis is at table x=y=0

        # ---- four markers: two on the left half, two on the right half ----
        # markpen long axis is local Y; lay it flat on the table (long axis along world X).
        flat_q = [0.7071068, 0.7071068, 0.0, 0.0]
        self.markers = []
        self.marker_ids = []
        # (xlim, ylim) zones: LEFT near/mid and RIGHT near/mid
        zones = [
            ([-0.28, -0.20], [-0.05, 0.05]),   # left, outer
            ([-0.20, -0.13], [0.08, 0.16]),    # left, mid
            ([0.20, 0.28], [-0.05, 0.05]),     # right, outer
            ([0.13, 0.20], [0.08, 0.16]),      # right, mid
        ]
        for i, (xlim, ylim) in enumerate(zones):
            mid = int(np.random.choice([0, 1, 2, 3]))
            mpose = rand_pose(
                xlim=xlim, ylim=ylim, zlim=[table_z],
                qpos=flat_q, rotate_rand=True, rotate_lim=[0, 0, np.pi / 8],
            )
            m = create_actor(self, pose=mpose, modelname="058_markpen",
                             model_id=mid, convex=True, is_static=False)
            m.set_mass(0.02)
            self.markers.append(m)
            self.marker_ids.append(mid)

        self.add_prohibit_area(self.cylinder, padding=0.05)
        for m in self.markers:
            self.add_prohibit_area(m, padding=0.03)

        # ---- attach bookkeeping ----
        # left arm presses the -x targets (135,225); right arm presses the +x targets (45,315)
        self._attach_plan = [
            # (marker_index, target_yaw_deg, arm)
        ]
        self.attached = [False, False, False, False]
        self.attach_locked_pose = [None, None, None, None]   # world Pose to lock each step
        self.attached_yaw = [None, None, None, None]         # recorded final yaw (rad)
        self.n_attached = 0

    # ---------------------------------------------------- attach point geometry
    def _surface_point(self, yaw_deg):
        """World xyz of the cylinder-side attach point at the given yaw (about vertical axis)."""
        a = np.deg2rad(yaw_deg)
        x = self.cyl_axis_xy[0] + self.cyl_radius * np.cos(a)
        y = self.cyl_axis_xy[1] + self.cyl_radius * np.sin(a)
        z = (0.74 + self.table_z_bias) + self.attach_z
        return np.array([x, y, z])

    def _marker_tip(self, marker):
        """A representative tip point of the marker: use the bottom functional point if present,
        else the actor origin. Used only as the proximity probe for the magnetic attach."""
        try:
            fp = marker.get_functional_point(1, "pose")   # base/bottom frame
            return np.array(fp.p)
        except Exception:
            return np.array(marker.get_pose().p)

    # ------------------------------------------------------- magnetic attach
    def _try_attach(self, idx, yaw_deg):
        """If marker idx is close enough to its target surface point, form the magnetic
        attachment: lock the marker as kinematic and record its yaw. Returns True on attach."""
        if self.attached[idx]:
            return True
        target = self._surface_point(yaw_deg)
        tip = self._marker_tip(self.markers[idx])
        if float(np.linalg.norm(tip[:2] - target[:2])) <= self.attach_dist:
            marker = self.markers[idx]
            # snap the marker so its body centre sits just outside the surface at this yaw,
            # giving a clean, evenly-spaced final layout regardless of small approach error.
            a = np.deg2rad(yaw_deg)
            radial = np.array([np.cos(a), np.sin(a), 0.0])
            body_center = np.array([
                self.cyl_axis_xy[0] + (self.cyl_radius + 0.012) * np.cos(a),
                self.cyl_axis_xy[1] + (self.cyl_radius + 0.012) * np.sin(a),
                (0.74 + self.table_z_bias) + self.attach_z,
            ])
            # orient the marker's long axis (local Y) along the radial direction, lying flat.
            # build a rotation: local Y -> radial(world), local Z -> world up.
            yaxis = radial
            zaxis = np.array([0.0, 0.0, 1.0])
            xaxis = np.cross(yaxis, zaxis)
            xaxis /= (np.linalg.norm(xaxis) + 1e-9)
            zaxis = np.cross(xaxis, yaxis)
            R = np.column_stack([xaxis, yaxis, zaxis])
            import transforms3d as t3d
            q = t3d.quaternions.mat2quat(R)
            locked = sapien.Pose(body_center, q.tolist())
            marker.actor.set_pose(locked)
            for c in marker.actor.get_components():
                if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                    c.set_linear_velocity(np.zeros(3))
                    c.set_angular_velocity(np.zeros(3))
                    c.set_kinematic(True)
                    c.set_kinematic_target(locked)
            self.attach_locked_pose[idx] = locked
            self.attached[idx] = True
            self.attached_yaw[idx] = np.deg2rad(yaw_deg) % (2 * np.pi)
            self.n_attached += 1
            return True
        return False

    def _update_kinematic_tasks(self):
        # base hook first (drives any DOMINO dynamic motion / registered kinematic tasks)
        super()._update_kinematic_tasks()
        # _init_task_env_ calls this during setup, before load_actors initializes state
        if not getattr(self, "attached", None):
            return
        # hold every already-attached marker rigidly fixed to the (static) cylinder
        for idx in range(4):
            if self.attached[idx] and self.attach_locked_pose[idx] is not None:
                for c in self.markers[idx].actor.get_components():
                    if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                        c.set_kinematic_target(self.attach_locked_pose[idx])

    def _dwell(self, idx=None, yaw_deg=None):
        """Let the sim settle for a few steps while recording frames; poll for the attach."""
        for i in range(self.dwell_steps):
            if idx is not None and not self.attached[idx]:
                self._try_attach(idx, yaw_deg)
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()

    # -------------------------------------------------------------- policy
    def _press_marker(self, idx, yaw_deg, arm_tag):
        """Pick a marker off the table and press/attach it to the cylinder at yaw_deg."""
        marker = self.markers[idx]
        # grasp the marker (it lies flat; default contact points grasp near one end)
        self.move(self.grasp_actor(marker, arm_tag=arm_tag, pre_grasp_dis=0.08, grasp_dis=0.0))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        # carry it over to a point just outside the cylinder surface at the target yaw, then
        # lower toward the attach height. Relative displacement from the lifted pose plans far
        # more reliably than an absolute far-reach pose.
        cur = (self.robot.get_left_ee_pose() if arm_tag == "left"
               else self.robot.get_right_ee_pose())
        target = self._surface_point(yaw_deg)
        # approach point a little outside the surface so the gripper clears the cylinder
        a = np.deg2rad(yaw_deg)
        approach = target[:2] + 0.10 * np.array([np.cos(a), np.sin(a)])
        dx = float(approach[0] - cur[0])
        dy = float(approach[1] - cur[1])
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx, y=dy))
        # press inward toward the cylinder; the magnetic attach fires inside the dwell
        self.move(self.move_by_displacement(
            arm_tag=arm_tag,
            x=-0.06 * float(np.cos(a)), y=-0.06 * float(np.sin(a)),
        ))
        # poll for attach during a short dwell, then release
        self._dwell(idx=idx, yaw_deg=yaw_deg)
        if not self.attached[idx]:
            # final fallback: force the attach (proximity may be just over threshold)
            self._try_attach(idx, yaw_deg)
        self.move(self.open_gripper(arm_tag))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.10, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))

    def play_once(self):
        left = ArmTag("left")
        right = ArmTag("right")

        # markers 0,1 are LEFT-half; 2,3 are RIGHT-half. Left arm -> -x targets (135,225);
        # right arm -> +x targets (45,315). This keeps each arm on its own side.
        # process order matches the PDF: 0deg-ish, 180deg-ish, 90deg-ish, 270deg-ish.
        # right first target (45), left first target (135), right second (315), left second (225)
        self._press_marker(2, 45.0, right)    # right arm, +x
        self._press_marker(0, 135.0, left)    # left arm,  -x
        self._press_marker(3, 315.0, right)   # right arm, +x
        self._press_marker(1, 225.0, left)    # left arm,  -x

        # final settle to record the assembled state
        self._dwell()

        self.info["info"] = {
            "{A}": f"058_markpen/base{self.marker_ids[0]}",
            "{B}": "magnet_cylinder",
        }
        return self.info

    # ------------------------------------------------------------- success / metric
    def evenness_score(self):
        yaws = [y for y in self.attached_yaw if y is not None]
        if len(yaws) < 4:
            return 0.0
        s = sorted([float(y) % (2 * np.pi) for y in yaws])
        gaps = []
        for i in range(4):
            g = (s[(i + 1) % 4] - s[i]) % (2 * np.pi)
            gaps.append(np.rad2deg(g))
        gaps = np.array(gaps)
        mean_err = float(np.mean(np.abs(gaps - 90.0)))
        return float(np.clip(1.0 - mean_err / 90.0, 0.0, 1.0))

    def check_success(self):
        ok = bool(self.n_attached >= 4)
        if ok:
            # expose the evenness metric (also recorded per-frame via get_obs)
            self._evenness = self.evenness_score()
        return ok

    def get_obs(self):
        obs = super().get_obs()
        obs["assemble"] = {
            "n_attached": int(self.n_attached),
            "attached": [bool(a) for a in self.attached],
            "attached_yaw_deg": [None if y is None else float(np.rad2deg(y))
                                 for y in self.attached_yaw],
            "evenness_score": float(self.evenness_score()),
        }
        return obs
