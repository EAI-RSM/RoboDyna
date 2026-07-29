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
    """Stop a rolling cup, side-grasp it, upright it, and shelve it.

    Cup (021_cup ids 0/1/2/5/6/3) starts upright on the lower shelf, tips onto
    its side, falls onto the table, and rolls toward the near edge. The robot
    blocks/stops it (still on its side), grasps the cylindrical side, turns it
    upright in hand, and places it back on the shelf plate. Falling off the
    table is a failure.

    Shelf tip/fall/roll is a step-driven kinematic trajectory so both collector
    passes are deterministic.
    """

    CUP_IDS = [0, 1, 2, 5, 6, 3]
    ROLL_SPEED_DEFAULT = 0.14
    FALL_SPEED_XY_DEFAULT = 0.07
    TIP_DURATION_DEFAULT = 0.90
    GRAVITY = 9.81

    TABLE_EDGE_Y_DEFAULT = -0.30
    # Fraction along table-roll (shelf land → near edge). Higher = catch farther
    # from the shelf so the arm has room to grasp / upright / carry.
    INTERCEPT_FRAC_DEFAULT = 0.72
    # Keep the blocking hand at least this far in front of the shelf lip.
    CATCH_SHELF_CLEARANCE_DEFAULT = 0.28
    # Hand blocks when the cup is this close (XY) to the TCP.
    STOP_XY_TOL_DEFAULT = 0.055
    STOP_LEAD_Y_DEFAULT = 0.05
    # Block-hand height above the table (low enough to interrupt the roll).
    BLOCK_Z_ABOVE_DEFAULT = 0.045
    POST_GRASP_LIFT_DEFAULT = 0.14
    UPRIGHT_HOLD_DEFAULT = 0.50
    TIP_UP_DURATION_DEFAULT = 0.45
    CUP_X_ABS_MIN = 0.10
    CUP_X_ABS_MAX = 0.18
    GRASP_TCP_ERR_MAX = 0.035
    EE_TO_TCP = 0.12

    CUP_ROBOT_IGNORE_BIT = 1 << 21
    CUP_ROBOT_IGNORE_ID = 0x0C51

    CUP_UPRIGHT_Q = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    GRASP_SPAN_MAX = 0.100

    _office_arr_choices = [0, 2]

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
        self._welded = False
        self._ee_offset = None  # cup position in TCP frame
        self._held_quat_rel = None  # cup quat relative to TCP
        self._table_start_idx = 0
        self._roll_start_idx = 0
        self._intercept_idx = 0
        self._intercept_pos = np.zeros(3)
        self._stop_pos = np.zeros(3)
        self._stop_quat = None
        self.arm_side = "right"
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

    def _cup_upright_quat(self):
        return self.CUP_UPRIGHT_Q.copy()

    def _cup_tip_quat(self, tip_frac, roll_angle=0.0):
        tip_frac = float(np.clip(tip_frac, 0.0, 1.0))
        q_upright = self.CUP_UPRIGHT_Q
        q_tip = euler2quat(0.0, tip_frac * np.pi / 2.0, 0.0, axes="sxyz")
        q_side = qmult(q_tip, q_upright)
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

    # --------------------------------------------------------------- trajectory
    def _build_trajectory(self):
        """Upright on shelf → tip → fall → table roll → optional fall-off."""
        dt = float(self.scene.get_timestep())
        g = self.GRAVITY
        r = self.cup_radius
        h = self.cup_height
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
            traveled += abs(vy) * (t_fall / n_fall) + abs(
                (z_side_shelf - 0.5 * g * t * t)
                - (z_side_shelf - 0.5 * g * ((i - 1) / n_fall * t_fall) ** 2)
            )
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
            traveled += abs(vy) * dt + 0.5 * g * t * dt
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
        self.tip_up_duration = float(c.get("tip_up_duration", self.TIP_UP_DURATION_DEFAULT))
        cup_ids = c.get("cup_ids", self.CUP_IDS)
        try:
            cup_ids = [int(x) for x in cup_ids]
        except Exception:
            cup_ids = list(self.CUP_IDS)
        if not cup_ids:
            cup_ids = list(self.CUP_IDS)
        if "cup_id" in c:
            self.cup_id = int(c["cup_id"])
        else:
            self.cup_id = int(np.random.choice(cup_ids))

        axial, diameter = self._cup_dims(self.cup_id)
        self.cup_scale_mult = float(c.get(
            "cup_scale_mult",
            min(1.0, self.GRASP_SPAN_MAX / max(axial, 1e-6)),
        ))
        self.cup_axial_span, diameter = self._cup_dims(self.cup_id, self.cup_scale_mult)
        self.cup_height = float(self.cup_axial_span)
        self.cup_radius = float(c.get("cup_radius", max(0.025, 0.5 * diameter)))

        self.table_top = 0.74 + float(self.table_z_bias)

        self.shelf_lims = list(self.office_info["shelf_lims"])
        shelf_x = float(self.office_info["furn_x_v"]["shelf"][self.arr_v])
        # Prefer the real lower-shelf plate from the wall-shelf mesh.
        self._measure_shelf_plate()
        self.shelf_front_y = float(self.shelf_plate_ylim[0])
        self.shelf_z_surf = float(self.shelf_plate_z)
        self.shelf_depth = float(0.5 * (self.shelf_plate_ylim[0] + self.shelf_plate_ylim[1]))

        sign = 1.0 if shelf_x >= 0 else -1.0
        self.arm_side = "right" if sign > 0 else "left"
        x0, x1 = self.shelf_plate_xlim
        y0, y1 = self.shelf_plate_ylim
        cup_x = float(np.clip(
            sign * np.random.uniform(self.CUP_X_ABS_MIN, self.CUP_X_ABS_MAX),
            x0 + 0.04,
            x1 - 0.04,
        ))
        cup_y = float(np.random.uniform(y0 + 0.04, y1 - 0.03))
        cup_z = self.shelf_z_surf + 0.5 * self.cup_height
        self.cup_start = np.array([cup_x, cup_y, cup_z], dtype=np.float64)
        # Place-back: upright on the lower shelf plate (mid-depth, same lane).
        self.shelf_place = np.array(
            [
                cup_x,
                float(y0 + 0.45 * (y1 - y0)),
                self.shelf_z_surf + 0.5 * self.cup_height,
            ],
            dtype=np.float64,
        )

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
        self._decouple_cup_from_robot()
        self._set_entity_pose(self.cup, cup_pose)

        self.add_prohibit_area(self.cup, padding=0.08)

        self._build_trajectory()
        self._traj_step = 0
        self._cup_state = "parked"
        self._fell_off = False
        self._stopped = False
        self._grasped = False
        self._placed = False
        self._welded = False
        self._ee_offset = None
        self._held_quat_rel = None
        self._stop_quat = None
        self._loaded = True

    def _decouple_cup_from_robot(self):
        """Keep the scripted cup out of the robot contact solver."""
        if self._cup_rigid is None:
            return
        bit = int(self.CUP_ROBOT_IGNORE_BIT)
        ident = int(self.CUP_ROBOT_IGNORE_ID) & 0xFFFF
        try:
            for shape in self._cup_rigid.get_collision_shapes():
                g0, g1, _, _ = shape.get_collision_groups()
                shape.set_collision_groups([int(g0), int(g1), bit, ident])
            for articulation in (self.robot.left_entity, self.robot.right_entity):
                if articulation is None:
                    continue
                for link in articulation.get_links():
                    for shape in link.get_collision_shapes():
                        g0, g1, g2, g3 = shape.get_collision_groups()
                        shape.set_collision_groups([
                            int(g0), int(g1), int(g2) | bit,
                            (int(g3) & ~0xFFFF) | ident,
                        ])
        except Exception:
            pass

    def _measure_shelf_plate(self):
        """Lower shelf-plate top Z and XY bounds from the wall-shelf mesh."""
        # Fallback from office_info if mesh query fails.
        depth = 0.28
        half_y = self.office_info["shelf_area"][1] / 2.0
        shelf_x = float(self.office_info["furn_x_v"]["shelf"][self.arr_v])
        sx = self.office_info["shelf_area"][0]
        self.shelf_plate_z = float(self.office_info["shelf_heights"][0]) + float(self.table_z_bias)
        self.shelf_plate_xlim = (shelf_x - sx / 2.0, shelf_x + sx / 2.0)
        self.shelf_plate_ylim = (depth - half_y + 0.02, depth + half_y - 0.02)
        try:
            ent = self.shelf.actor if hasattr(self.shelf, "actor") else self.shelf
            T = ent.get_pose().to_transformation_matrix()
            pts = []
            for comp in ent.get_components():
                if not hasattr(comp, "get_collision_shapes"):
                    continue
                for shape in comp.get_collision_shapes():
                    verts = np.asarray(shape.get_vertices(), dtype=np.float64)
                    lp = shape.get_local_pose().to_transformation_matrix()
                    hom = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
                    pts.append((T @ lp @ hom.T).T[:, :3])
            if not pts:
                return
            P = np.concatenate(pts, axis=0)
            mid_z = 0.5 * (float(P[:, 2].min()) + float(P[:, 2].max()))
            low = P[P[:, 2] < mid_z]
            hist, edges = np.histogram(low[:, 2], bins=50)
            keep = np.where(hist >= max(5, int(hist.max() * 0.25)))[0]
            if keep.size == 0:
                return
            i = int(keep[-1])  # topmost dense band in the lower half = plate top
            z = float(0.5 * (edges[i] + edges[i + 1]))
            band = low[np.abs(low[:, 2] - z) < 0.02]
            if len(band) < 8:
                return
            self.shelf_plate_z = z
            self.shelf_plate_xlim = (float(band[:, 0].min()), float(band[:, 0].max()))
            self.shelf_plate_ylim = (float(band[:, 1].min()), float(band[:, 1].max()))
            self.shelf_lims = [
                self.shelf_plate_xlim[0],
                self.shelf_plate_ylim[0],
                self.shelf_plate_xlim[1],
                self.shelf_plate_ylim[1],
            ]
        except Exception:
            pass

    # ----------------------------------------------------------- kinematics
    def _release_cup(self):
        if self._cup_state != "parked":
            return
        self._cup_state = "rolling"
        self._traj_step = 0

    def _stop_cup(self):
        """Freeze the rolling cup where the hand blocked it (keep on-side pose)."""
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
        if self._cup_state == "grasped" and self._welded:
            self._update_welded_cup()
            return
        if self._cup_state in ("parked", "stopped", "placed", "grasped"):
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
        tcp = self._tcp_pos(arm_tag)
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        return cp - tcp

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

    def _update_welded_cup(self):
        """Keep the cup fixed in the TCP frame so arm rotation turns the cup."""
        if not self._welded or self._ee_offset is None or self._held_quat_rel is None:
            return
        arm = ArmTag(self.arm_side)
        try:
            tcp = self._tcp_pose(arm)
        except Exception:
            return
        R = quat2mat(tcp[3:])
        target_p = tcp[:3] + R @ np.asarray(self._ee_offset, dtype=np.float64)
        target_q = qmult(tcp[3:], self._held_quat_rel)
        self._set_entity_pose(
            self.cup,
            sapien.Pose(target_p.tolist(), np.asarray(target_q, dtype=np.float64).tolist()),
        )

    def _weld_cup_to_ee(self):
        arm = ArmTag(self.arm_side)
        tcp = self._tcp_pose(arm)
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        cq = np.array(self.cup.get_pose().q, dtype=np.float64)
        R = quat2mat(tcp[3:])
        self._ee_offset = R.T @ (cp - tcp[:3])
        self._held_quat_rel = np.asarray(qmult(qinverse(tcp[3:]), cq), dtype=np.float64)
        self._welded = True
        self._grasped = True
        self._cup_state = "grasped"
        if self._cup_rigid is not None:
            try:
                self._cup_rigid.set_disable_gravity(True)
                self._cup_rigid.set_kinematic(True)
            except Exception:
                pass

    def _unweld_cup(self):
        self._welded = False
        self._ee_offset = None
        self._held_quat_rel = None

    def _update_kinematic_tasks(self):
        super()._update_kinematic_tasks()
        if not getattr(self, "_loaded", False):
            return
        self._advance_cup()

    def check_stable(self):
        if self.cup is not None and self._cup_state == "parked":
            pose = sapien.Pose(
                self.cup_start.tolist(),
                self._cup_upright_quat().tolist(),
            )
            self._set_entity_pose(self.cup, pose)
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
        """Where the open hand waits to stop the rolling cup.

        Pull the catch toward the robot (smaller y) so it is not cramped against
        the shelf — needed for side-grasp / upright / re-shelve room.
        """
        tcp = self._intercept_pos.copy()
        y_max = float(self.shelf_front_y) - float(self.catch_shelf_clearance)
        # Also prefer the configured table-roll fraction target.
        y_max = min(y_max, float(self._intercept_pos[1]))
        tcp[1] = min(float(tcp[1]), y_max)
        # Stay short of the fall-off edge so the catch is still on the table.
        tcp[1] = max(float(tcp[1]), float(self.table_edge_y) + 0.06)
        tcp[2] = self.table_top + max(self.cup_radius + 0.01, self.block_z_above)
        return tcp

    def _approach_block(self, arm_tag):
        """Park an open hand on the roll lane (TCP target, not EE)."""
        target = self._block_tcp()
        quat = self._block_quat(arm_tag)
        pose = [float(target[0]), float(target[1]), float(target[2])] + quat
        return self.move_to_pose(arm_tag, pose)

    def _retarget_traj_to_tcp(self, arm_tag):
        if not self._traj:
            return
        tcp = self._tcp_pos(arm_tag)
        x_lane = float(tcp[0])
        y_catch = float(tcp[1])
        new_traj = []
        for pos, quat in self._traj:
            p = pos.copy()
            p[0] = x_lane
            new_traj.append((p, quat.copy()))
        self._traj = new_traj

        self.cup_start[0] = x_lane
        self.shelf_place[0] = x_lane
        if self._cup_state == "parked" and self.cup is not None:
            self._set_entity_pose(
                self.cup,
                sapien.Pose(self.cup_start.tolist(), self._cup_upright_quat().tolist()),
            )

        land = int(self._land_idx)
        best_i = land
        best_dy = 1e9
        for i in range(land, int(self._fall_off_idx) + 1):
            dy = abs(float(self._traj[i][0][1]) - y_catch)
            if dy < best_dy:
                best_dy = dy
                best_i = i
        self._intercept_idx = int(best_i)
        self._intercept_pos = self._traj[self._intercept_idx][0].copy()
        self._intercept_pos[0] = x_lane
        self._intercept_pos[1] = y_catch
        self._intercept_pos[2] = float(self._traj[self._intercept_idx][0][2])

    def _block_quat(self, arm_tag):
        key = ("top_down_little_left" if str(arm_tag) == "right"
               else "top_down_little_right")
        return list(GRASP_DIRECTION_DIC[key])

    def _side_grasp_quat_keys(self, arm_tag):
        """Prefer side-wall pinches of the lying cup (fingers along ±Y)."""
        if str(arm_tag) == "right":
            return ("down_left", "top_down_little_left", "down_right")
        return ("down_right", "top_down_little_right", "down_left")

    def _place_quat(self, arm_tag):
        # Front horizontal carry reaches the shelf plate via displacement steps.
        return list(GRASP_DIRECTION_DIC["front"])

    def _wait_until_at_hand(self, arm_tag, max_steps=2500):
        """Wait until the cup reaches the blocking hand (or the intercept index)."""
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
        self._set_entity_pose(
            self.cup,
            sapien.Pose(self._stop_pos.tolist(), self._stop_quat.tolist()),
        )

    def _try_contact_side_grasp(self, arm_tag):
        """Use annotated cup contact points when they yield a reachable side grasp."""
        try:
            pref = self.robot.get_grasp_perfect_direction(arm_tag)
        except Exception:
            pref = "front_left" if str(arm_tag) == "right" else "front_right"
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        best = None
        best_score = 1e9
        for i in range(4):
            try:
                pre = self.get_grasp_pose(self.cup, arm_tag, contact_point_id=i, pre_dis=0.10)
                grasp = self.get_grasp_pose(self.cup, arm_tag, contact_point_id=i, pre_dis=0.0)
            except Exception:
                continue
            if pre is None or grasp is None:
                continue
            # Lying-cup rim contacts often project far along ±X; skip those.
            if float(np.linalg.norm(np.asarray(grasp[:3], dtype=np.float64) - cp)) > 0.12:
                continue
            d_side = cal_quat_dis(grasp[-4:], GRASP_DIRECTION_DIC[pref])
            d_td = cal_quat_dis(
                grasp[-4:],
                GRASP_DIRECTION_DIC[
                    "top_down_little_left" if str(arm_tag) == "right"
                    else "top_down_little_right"
                ],
            )
            # Keep poses that look like side grasps, not top-down.
            if d_side > 0.55 and d_td < 0.25:
                continue
            score = 0.7 * d_side + 0.3 * d_td
            if score < best_score:
                best_score = score
                best = (pre, grasp)

        if best is None:
            return False
        pre, grasp = best
        self.plan_success = True
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self._restore_stop_pose()
        self.move(self.move_to_pose(arm_tag, pre))
        if not self.plan_success:
            return False
        self.move(self.move_to_pose(arm_tag, grasp))
        if not self.plan_success:
            return False
        tcp = self._tcp_pos(arm_tag)
        if float(np.linalg.norm(tcp - cp)) > self.GRASP_TCP_ERR_MAX + 0.04:
            return False
        self.move(self.close_gripper(arm_tag=arm_tag, pos=0.0))
        self._weld_cup_to_ee()
        self._dwell(8)
        return True

    def _try_estimated_side_grasp(self, arm_tag):
        """Estimate a side-wall grasp of the lying cup (axle along ±X).

        Absolute IK to table-height side poses often false-succeeds, so stage
        above the cup then walk the TCP down with short displacements.
        """
        self._restore_stop_pose()
        cp = np.array(self.cup.get_pose().p, dtype=np.float64)
        cp[2] = self.table_top + max(0.035, self.cup_radius + 0.005)
        above = cp + np.array([0.0, 0.0, 0.14], dtype=np.float64)
        block_q = self._block_quat(arm_tag)

        # Hover above the stopped cup first (reachable from the block pose).
        self.plan_success = True
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self.move(self.move_to_pose(arm_tag, above.tolist() + block_q))
        self._walk_tcp_toward(arm_tag, above, block_q, step=0.03, tol=0.06, max_steps=16)

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
            self.move(self.close_gripper(arm_tag=arm_tag, pos=0.0))
            self._weld_cup_to_ee()
            self._dwell(8)
            return True
        return False

    def _grasp_stopped_cup(self, arm_tag):
        """Side-grasp the stopped on-side cup (contact pose, else estimate)."""
        # Clear the blocking top-down pose first.
        quat_clear = self._block_quat(arm_tag)
        tcp = self._tcp_pos(arm_tag)
        clear = [float(tcp[0]), float(tcp[1] + 0.03), float(tcp[2] + 0.12)] + quat_clear
        self.plan_success = True
        self.move(self.move_to_pose(arm_tag, clear))
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self._restore_stop_pose()

        if self._try_contact_side_grasp(arm_tag):
            return True
        self.plan_success = True
        return self._try_estimated_side_grasp(arm_tag)

    def _cup_opening_axis(self):
        """World-frame cup opening axis (mesh height is local +Y)."""
        R = quat2mat(np.array(self.cup.get_pose().q, dtype=np.float64))
        return R[:, 1].copy()

    def _rotate_held_cup_upright(self, arm_tag):
        """Rotate the wrist so the welded cup turns upright (no free quat snap).

        Commands EE orientations via ``move_by_displacement`` (house style): the
        same world rotation that stands the cup up is applied to the current EE
        quat in small slerp steps so the relative weld makes the cup follow.
        """
        if not self._welded or self._held_quat_rel is None:
            return False
        ee0 = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)
        q_ee0 = ee0[3:].copy()
        tcp = self._tcp_pose(arm_tag)
        q_cup0 = np.asarray(qmult(tcp[3:], self._held_quat_rel), dtype=np.float64)
        q_up = self._cup_upright_quat()
        # Same world rotation that stands the cup up, applied to the EE.
        q_delta = qmult(q_up, qinverse(q_cup0))
        q_ee1 = np.asarray(qmult(q_delta, q_ee0), dtype=np.float64)

        # Lift clear of the table before / while pitching the wrist.
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=0.04, quat=q_ee0.tolist(), move_axis="world",
        ))

        n = 12
        ok = True
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3.0 - 2.0 * s)
            q = self._quat_slerp(q_ee0, q_ee1, s)
            self.plan_success = True
            # Tiny upward bias keeps the cup from scuffing while the wrist turns.
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                z=0.004,
                quat=q.tolist(),
                move_axis="world",
            ))
            if not self.plan_success:
                ok = False
                self.plan_success = True
            self._dwell(3)

        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=0.0, quat=q_ee1.tolist(), move_axis="world",
        ))
        self.plan_success = True
        self._dwell(6)
        # Opening axis should point mostly +Z after a successful wrist turn.
        return ok and float(self._cup_opening_axis()[2]) > 0.75

    def _walk_tcp_toward(self, arm_tag, target, quat=None, step=0.02, tol=0.03, max_steps=40):
        """Nudge the TCP toward a world target with short EE displacements."""
        target = np.asarray(target, dtype=np.float64)
        last = 1e9
        stall = 0
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
            # move_by_displacement expects an EE quat (same R as TCP on this robot).
            ee_q = (list(quat) if quat is not None
                    else np.array(self.get_arm_pose(arm_tag), dtype=np.float64)[3:].tolist())
            self.plan_success = True
            self.move(self.move_by_displacement(
                arm_tag=arm_tag,
                x=float(step_vec[0]),
                y=float(step_vec[1]),
                z=float(step_vec[2]),
                quat=ee_q,
                move_axis="world",
            ))
            if not self.plan_success:
                break
        return float(np.linalg.norm(target - self._tcp_pos(arm_tag))) <= tol + 0.025

    def _place_cup_on_shelf(self, arm_tag):
        """Lift, wrist-turn the cup upright, carry onto the shelf plate, set down."""
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, z=self.post_grasp_lift, move_axis="world",
        ))
        if not self.plan_success:
            return False

        # Arm rotates; relative weld makes the cup stand up with the wrist.
        if not self._rotate_held_cup_upright(arm_tag):
            # Best-effort: still try to shelve with whatever pose we have.
            self.plan_success = True
        self._dwell(4)

        # Carry with the current upright-holding EE pose (no free cup reorient).
        quat = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)[3:].tolist()

        # Refresh place target from the measured plate (lane may have retargeted).
        x0, x1 = self.shelf_plate_xlim
        y0, y1 = self.shelf_plate_ylim
        sp = self.shelf_place.copy()
        sp[0] = float(np.clip(sp[0], x0 + 0.05, x1 - 0.05))
        sp[1] = float(y0 + 0.45 * (y1 - y0))
        sp[2] = float(self.shelf_plate_z + 0.5 * self.cup_height)
        self.shelf_place = sp

        # Approach above the plate, then lower onto it.
        above = np.array([sp[0], sp[1], sp[2] + 0.10], dtype=np.float64)
        self._walk_tcp_toward(arm_tag, above, quat, step=0.025, tol=0.04, max_steps=50)
        quat = np.array(self.get_arm_pose(arm_tag), dtype=np.float64)[3:].tolist()
        self._walk_tcp_toward(arm_tag, sp + np.array([0.0, 0.0, 0.02]), quat,
                              step=0.02, tol=0.035, max_steps=30)

        self.plan_success = True
        self.move(self.open_gripper(arm_tag=arm_tag, pos=1.0))
        self._unweld_cup()
        # Seat by translating only — keep the orientation from the wrist turn.
        hand_p = np.array(self.cup.get_pose().p, dtype=np.float64)
        hand_q = np.array(self.cup.get_pose().q, dtype=np.float64)
        for i in range(1, 9):
            s = i / 8.0
            s = s * s * (3.0 - 2.0 * s)
            p = hand_p * (1.0 - s) + sp * s
            self._set_entity_pose(self.cup, sapien.Pose(p.tolist(), hand_q.tolist()))
            self._dwell(1)
        self._set_entity_pose(
            self.cup,
            sapien.Pose(sp.tolist(), hand_q.tolist()),
        )
        self._cup_state = "placed"
        self._placed = True
        self._dwell(16)
        self.plan_success = True
        self.move(self.move_by_displacement(
            arm_tag=arm_tag, y=-0.06, z=0.06, move_axis="world",
        ))
        return True

    def play_once(self):
        arm_tag = ArmTag(self.arm_side)
        self._pic_counter = 0
        dt = float(self.scene.get_timestep())

        # 1) Open hand and park on the roll lane as a blocker.
        self.move(self.open_gripper(arm_tag=arm_tag))
        self.move(self._approach_block(arm_tag))
        if self.plan_success:
            self._retarget_traj_to_tcp(arm_tag)

        # 2) Cup stands, tips, falls, rolls toward the hand.
        self._dwell(max(1, int(round(self.upright_hold / max(dt, 1e-4)))))
        self._release_cup()

        # 3) Stop the cup with the hand — leave it on its side.
        if self._wait_until_at_hand(arm_tag) and not self._fell_off:
            self._stop_cup()
            self._dwell(12)

            # 4) Side-grasp the lying cup, upright in hand, place on shelf plate.
            if self.plan_success and self._grasp_stopped_cup(arm_tag):
                self._place_cup_on_shelf(arm_tag)

        if not self._stopped and self._cup_state == "rolling":
            while self._traj_step < len(self._traj):
                self._dwell(1)
                if self._fell_off or self._cup_state == "fallen":
                    break

        self.info["info"] = {
            "{A}": f"021_cup/base{self.cup_id}",
            "{B}": "121_wall-shelf",
            "{C}": "036_cabinet",
            "{D}": "122_file-holder",
            "{a}": str(arm_tag),
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        if self._fell_off or self._cup_state == "fallen":
            return False
        if not self._placed:
            return False
        p = np.array(self.cup.get_pose().p, dtype=np.float64)
        # Cup resting upright on the measured lower shelf plate.
        bottom_z = p[2] - 0.5 * self.cup_height
        on_plate_z = abs(bottom_z - self.shelf_plate_z) < 0.04
        x0, x1 = self.shelf_plate_xlim
        y0, y1 = self.shelf_plate_ylim
        in_x = (x0 - 0.03) <= p[0] <= (x1 + 0.03)
        in_y = (y0 - 0.03) <= p[1] <= (y1 + 0.03)
        return bool(on_plate_z and in_x and in_y and not self._welded)

    def get_obs(self):
        obs = super().get_obs()
        obs["catch_rolling_cup"] = {
            "cup_state": str(self._cup_state),
            "fell_off": bool(self._fell_off),
            "stopped": bool(self._stopped),
            "grasped": bool(self._grasped),
            "placed": bool(self._placed),
            "traj_step": int(self._traj_step),
            "intercept_idx": int(self._intercept_idx),
        }
        return obs
