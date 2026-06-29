from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np
import json


class toast_bread(Base_Task):
    """Pick a slice of bread off the table, place it on a steamer, let it toast (a per-step
    timer browns its surface pale -> golden -> brown -> burnt), and once it reaches the target
    doneness (golden) grasp it off and set it on a serving plate. Single-arm, time-evolving state.

    Mirrors cook_meat: a contact-gated, step-count-driven 'toasting' timer that recolors the
    bread's render shapes over a multi-stop palette, with a frame-recording idle loop (NOT delay())
    so the browning lands in the rendered video / HDF5."""

    TOAST_STEPS_DEFAULT = 1000        # on-steamer sim steps for doneness 0 -> 1 (longer = slower, clearer change)
    TARGET_DONENESS_DEFAULT = 0.45    # golden: toast to just-golden, then lift off
    # 4-stop toasting gradient: pale dough -> golden -> brown -> burnt (near-black)
    COLOR_STOPS = [
        (0.00, [0.92, 0.80, 0.58]),   # raw: pale dough
        (0.45, [0.82, 0.55, 0.18]),   # golden (target): warm toasted gold
        (0.75, [0.45, 0.26, 0.10]),   # brown: dark toast
        (1.00, [0.06, 0.05, 0.04]),   # burnt: near-black
    ]

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's `task_args` block BEFORE init,
        # and initialize per-step state so an early _update_kinematic_tasks() can't crash.
        self._toast_cfg = kwags.get("task_args", {}).get("toast_bread", {})
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._toasting_active = False
        self._grasp_doneness = None
        self._bread_shapes = []
        super()._init_task_env_(**kwags)

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        self.toast_steps = int(self._toast_cfg.get("toast_steps", self.TOAST_STEPS_DEFAULT))
        self.target_doneness = float(self._toast_cfg.get("target_doneness", self.TARGET_DONENESS_DEFAULT))
        self.plate_scale_mult = float(self._toast_cfg.get("plate_scale_mult", 0.65))
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._toasting_active = False
        self._grasp_doneness = None

        self.steamer_scale_mult = float(self._toast_cfg.get("steamer_scale_mult", 0.10))

        # bread, steamer and serving plate all live on the SAME side so one arm can grasp the bread
        # off the table, toast it on the steamer, and set it on the plate (a cross-body reach to the
        # far side is unplannable). The steamer sits central on the side, the plate outboard, and the
        # bread is staged outboard-front.
        side = float(np.random.choice([-1.0, 1.0]))
        self._side = side

        # steamer: a round basket-style toasting vessel, static, in the reliable placement zone (the
        # cook_meat-proven central-back zone on the chosen side). 067_steamer has NO authored scale
        # (renders ~2.4 m raw) and NO functional/contact points, so we size it DOWN with scale_mult
        # (per-spawn, in-memory; we do NOT edit the asset) and compute its top surface ourselves.
        self.steamer_id = int(np.random.choice([0, 1]))
        steamer_pose = rand_pose(
            xlim=sorted([side * 0.03, side * 0.10]), ylim=[-0.16, -0.08],
            qpos=[0.707, 0.707, 0, 0], rotate_rand=False,
        )
        self.steamer = create_actor(
            self, pose=steamer_pose, modelname="067_steamer",
            model_id=self.steamer_id, convex=True, is_static=True,
            scale_mult=self.steamer_scale_mult,
        )
        # 067_steamer's model_data has no "scale" key, so create_actor drops the whole config
        # (Actor.config = None) -> add_prohibit_area / point getters crash on None. Give the actor a
        # minimal in-memory config (the on-disk asset is untouched) with the effective scale and the
        # asset's extents/center so add_prohibit_area can size its bounding box.
        _sd = json.load(open(f"assets/objects/067_steamer/model_data{self.steamer_id}.json"))
        sm = self.steamer_scale_mult
        self.steamer.config = {
            "scale": [sm, sm, sm],
            "extents": _sd["extents"],
            "center": _sd["center"],
        }
        # steamer top surface: the asset's local +Y is its axis; spawned with qpos [0.707,0.707,0,0]
        # (90 deg about X) that axis maps to world +Z, so the basket opens upward. Top world-z =
        # spawn z + scaled (center_y + extents_y/2). Used as the bread's toasting/drop target.
        _ext = np.asarray(_sd["extents"]) * sm
        _cen = np.asarray(_sd["center"]) * sm
        self._steamer_top_z = float(steamer_pose.p[2] + (_cen[1] + _ext[1] / 2.0))
        self._steamer_xy = np.array([float(steamer_pose.p[0]), float(steamer_pose.p[1])])

        # serving plate: OUTBOARD of the steamer at the same (back) y band. The bread is carried off
        # the steamer to the plate mostly laterally (+x, outboard) -- the cook_meat-proven set-down
        # direction, within reach -- and the plate is clearly separated from the steamer center so
        # check_success can distinguish "on the plate" from "on the steamer". Shrunk via scale_mult
        # (003_plate is a shared/stock asset, so we do NOT edit its model_data).
        plate_pose = rand_pose(
            xlim=sorted([side * 0.14, side * 0.20]), ylim=[-0.13, -0.05],
            qpos=[0.5, 0.5, 0.5, 0.5], rotate_rand=False,
        )
        self.plate = create_actor(
            self, pose=plate_pose, modelname="003_plate",
            model_id=0, convex=False, is_static=True, scale_mult=self.plate_scale_mult,
        )

        # bread slice: laid flat (thickness axis vertical) on the table, same side, outer-front.
        bread_pose = rand_pose(
            xlim=sorted([side * 0.23, side * 0.29]), ylim=[0.0, 0.09],
            zlim=[0.74 + self.table_z_bias], qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True, rotate_lim=[0, np.pi / 6, 0],
        )
        self.bread = create_actor(
            self, pose=bread_pose, modelname="202_bread_toast",
            model_id=0, convex=True, is_static=False,
        )
        self.bread.set_mass(0.02)

        self.add_prohibit_area(self.steamer, padding=0.05)
        self.add_prohibit_area(self.plate, padding=0.04)
        self.add_prohibit_area(self.bread, padding=0.03)

        # cache the bread's render shapes so the toasting timer can recolor them
        self._bread_shapes = []
        for c in self.bread.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._bread_shapes = list(c.render_shapes)
        self._set_bread_color(0.0)

    # -------------------------------------------------------- toasting state
    def _set_bread_color(self, doneness):
        d = float(np.clip(doneness, 0.0, 1.0))
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
        col = list(rgb) + [1.0]
        for s in self._bread_shapes:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step.
        super()._update_kinematic_tasks()
        # guard: per-step state may not exist yet on the setup-time call (before load_actors)
        if not getattr(self, "_toasting_active", False):
            return
        try:
            on_steamer = self.check_actors_contact("202_bread_toast", "067_steamer")
        except Exception:
            on_steamer = False
        if on_steamer:
            self.doneness = min(1.0, self.doneness + 1.0 / max(1, self.toast_steps))
            self.max_doneness = max(self.max_doneness, self.doneness)
            self._set_bread_color(self.doneness)

    def _toast_idle(self):
        # dwell on the steamer, recording frames, until the target doneness is reached
        max_steps = int(round(self.target_doneness * self.toast_steps)) + 30
        for i in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self.doneness >= self.target_doneness:
                break

    # ------------------------------------------------------------- policy
    def _dbg(self, tag):
        import os
        if os.environ.get("TOAST_DEBUG"):
            print(f"[toast_bread] {tag}: plan_success={self.plan_success}", flush=True)

    def play_once(self):
        arm_tag = ArmTag("right" if self.bread.get_pose().p[0] > 0 else "left")

        # grasp the raw bread off the table
        self.move(self.grasp_actor(self.bread, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._dbg("grasp_bread")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._dbg("lift_bread")

        # seat the bread on the steamer TOP surface. 067_steamer has no functional point, so we
        # target a pose at the steamer's world top-center (computed in load_actors from the scaled
        # AABB) with an upward-facing frame, and release slightly above so the bread settles onto the
        # solid top/rim (it is wider than the basket's holes, so it bridges them rather than dropping
        # through) instead of the gripper diving into the basket.
        steamer_target = [float(self._steamer_xy[0]), float(self._steamer_xy[1]),
                          self._steamer_top_z, 1.0, 0.0, 0.0, 0.0]
        self.move(
            self.place_actor(
                self.bread, target_pose=steamer_target, arm_tag=arm_tag,
                constrain="free", pre_dis=0.08, dis=0.04, is_open=True,
            ))
        self._dbg("place_on_steamer")
        # lift the gripper clear of the bread so the toasting dwell isn't perturbed; do NOT
        # back_to_origin (the outboard plate can block that path and it isn't needed for toasting).
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._dbg("lift_clear")

        # toast it on the steamer until golden
        self._toasting_active = True
        self._toast_idle()
        self._grasp_doneness = self.doneness
        self._toasting_active = False   # stop the timer before the grasp-off motion

        # grasp the toasted bread off the steamer and lift it clear
        self.move(self.grasp_actor(self.bread, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self._dbg("grasp_off_steamer")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))
        self._dbg("lift_off_steamer")

        # Set the toasted bread down on the serving plate. The plate is flat (not concave), so a
        # relative set-down -- shift the held bread horizontally over the plate center, lower, open
        # -- plans far more reliably than an absolute place onto the plate's functional frame (whose
        # authored orientation yields an unreachable gripper pose; this was the cook_meat lesson).
        bread_xy = np.array(self.bread.get_pose().p[:2])
        plate_xy = np.array(self.plate.get_functional_point(0)[:2])
        dx, dy = float(plate_xy[0] - bread_xy[0]), float(plate_xy[1] - bread_xy[1])
        # split into single-axis displacements (a diagonal one-shot move is much harder to plan):
        # the dominant outboard +x carry first (cook_meat-proven direction), then the small y trim.
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=dx))
        self._dbg("over_plate_x")
        self.move(self.move_by_displacement(arm_tag=arm_tag, y=dy))
        self._dbg("over_plate_y")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.06))
        self._dbg("lower_to_plate")
        self.move(self.open_gripper(arm_tag))
        self._dbg("release_on_plate")
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.08))

        self.info["info"] = {
            "{A}": "202_bread_toast/base0",
            "{B}": f"067_steamer/base{self.steamer_id}",
            "{C}": "003_plate/base0",
            "{a}": str(arm_tag),
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        if self._grasp_doneness is None:
            return False
        # toasted to at least (near) the golden target while on the steamer
        toasted_ok = self.max_doneness >= self.target_doneness - 0.05

        bread_p = self.bread.get_pose().p
        bread_z = float(bread_p[2])
        bread_xy = np.array(bread_p[:2])

        # bread ended up resting on/over the serving plate (the task's intent), and not back on the
        # steamer. The plate sits outboard of the steamer, so we judge "on the plate" by being
        # clearly closer to the plate center than to the steamer center, within a plate-sized radius.
        plate_xy = np.array(self.plate.get_functional_point(0)[:2])
        steamer_xy = self._steamer_xy
        d_plate = float(np.linalg.norm(bread_xy - plate_xy))
        d_steamer = float(np.linalg.norm(bread_xy - steamer_xy))
        on_plate = d_plate < 0.12
        off_steamer = d_plate < d_steamer            # nearer the plate than the steamer
        # not dropped on the floor / still resting on the table-level surfaces
        above_table = bread_z > (0.73 + self.table_z_bias)

        import os
        if os.environ.get("TOAST_DEBUG"):
            print(f"[toast_bread] check: toasted_ok={toasted_ok} (max_done={self.max_doneness:.3f}) "
                  f"d_plate={d_plate:.3f} d_steamer={d_steamer:.3f} on_plate={on_plate} "
                  f"off_steamer={off_steamer} bread_z={bread_z:.3f} above_table={above_table}",
                  flush=True)

        return bool(toasted_ok and on_plate and off_steamer and above_table)

    # record the toasting state into the trajectory (per-frame)
    def get_obs(self):
        obs = super().get_obs()
        obs["toasting"] = {
            "doneness": float(getattr(self, "doneness", 0.0)),
            "target_doneness": float(getattr(self, "target_doneness", self.TARGET_DONENESS_DEFAULT)),
        }
        return obs
