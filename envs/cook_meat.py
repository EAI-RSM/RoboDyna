from ._base_task import Base_Task
from .utils import *
import sapien
import sapien.render
import numpy as np


class cook_meat(Base_Task):
    """Place a raw steak on a pan, let it cook (a per-step timer browns it gradually),
    then grasp it off once it reaches a target doneness. Introduces a time-evolving,
    rendered object state on top of the standard pick-place primitives."""

    COOK_STEPS_DEFAULT = 1000         # on-pan sim steps for doneness 0 -> 1 (longer = slower, clearer change)
    TARGET_DONENESS_DEFAULT = 0.5     # medium: cook to the brown midpoint, then grasp off
    # cooking gradient: raw red -> MEDIUM warm red-brown (at the 0.5 target) -> well-done dark brown.
    # 0.5 is deliberately a clear MEDIUM (still reddish), not full brown, so picking at the 0.5
    # target reads visually as "medium", not "fully cooked".
    COLOR_STOPS = [
        (0.0, [1.00, 0.12, 0.09]),    # raw: vivid saturated red
        (0.5, [0.66, 0.30, 0.14]),    # medium: warm red-brown (clearly transitional)
        (1.0, [0.16, 0.08, 0.04]),    # well done: dark brown
    ]

    def setup_demo(self, **kwags):
        # capture task-scoped params from the (general) config's `task_args` block
        self._cook_cfg = kwags.get("task_args", {}).get("cook_meat", {})
        self._ep_seed = int(kwags.get("seed", 0))   # used to balance left/right hand per episode
        super()._init_task_env_(**kwags)

    # ---------------------------------------------------------------- actors
    def load_actors(self):
        c = self._cook_cfg
        # KEY per-episode randomization: the COOK SPEED = how many sim steps the steak takes to go
        # from raw(0) to fully done(1.0). Smaller = browns faster. Also a tight doneness band ~medium.
        self.cook_steps = int(np.random.uniform(c.get("cook_steps_min", 600), c.get("cook_steps_max", 1600)))
        self.target_doneness = float(np.random.uniform(c.get("target_doneness_min", 0.45),
                                                       c.get("target_doneness_max", 0.55)))
        self.doneness = 0.0
        self.max_doneness = 0.0
        self._cooking_active = False
        self._grasp_doneness = None

        # Pan and steak live on the SAME side so one arm can grasp the steak off the table, cook it
        # on the pan, and set it back down (a cross-body reach to the far side is unplannable).
        # Balance left/right by seed parity (np.random.choice was streaky -> all-left on seeds 0-3),
        # so both hands are reliably exercised across the dataset.
        side = -1.0 if (self._ep_seed % 2 == 0) else 1.0
        self._side = side

        # pan: a heavy static frying pan, kept in the arm's reliable placement zone on the side.
        # scale_mult enlarges the asset (mesh + functional point together) -> bigger bowl = easier,
        # more reliable steak placement (helps the UR5 reach/place yield).
        self.pan_scale = float(self._cook_cfg.get("pan_scale", 1.0))   # original size (config-tunable)
        # placement offset (world x,y) added to the pan functional point so the steak lands centered
        # in the bowl (the grasp offset otherwise lands it off-center) -- tune from the measured offset
        self.place_dx = float(self._cook_cfg.get("place_dx", 0.0))
        self.place_dy = float(self._cook_cfg.get("place_dy", 0.0))
        self.skillet_id = int(np.random.choice([0, 1, 2, 3]))
        skillet_pose = rand_pose(
            xlim=sorted([side * 0.02, side * 0.13]), ylim=[-0.17, -0.03],
            qpos=[0, 0, 0.707, 0.707], rotate_rand=False,
        )
        self.skillet = create_actor(
            self, pose=skillet_pose, modelname="106_skillet",
            model_id=self.skillet_id, convex=True, is_static=True, scale_mult=self.pan_scale,
        )

        # steak: laid flat (qpos lays the thickness axis vertical, like 075_bread) on the table,
        # on the same side, further out than the pan
        steak_pose = rand_pose(
            xlim=sorted([side * 0.15, side * 0.30]), ylim=[-0.02, 0.15],
            zlim=[0.74 + self.table_z_bias], qpos=[0.707, 0.707, 0.0, 0.0],
            rotate_rand=True, rotate_lim=[0, np.pi / 6, 0],
        )
        # thicken the steak along its thin axis (model-y, which qpos maps to world-z) so it stands
        # taller in the pan -> the gripper gets a clean bite to lift it back OUT of the bowl
        self.steak_thick = float(self._cook_cfg.get("steak_thick", 1.6))
        self.steak = create_actor(
            self, pose=steak_pose, modelname="200_steak",
            model_id=0, convex=True, is_static=False, scale_mult=(1.0, self.steak_thick, 1.0),
        )
        self.steak.set_mass(0.05)

        self.add_prohibit_area(self.skillet, padding=0.05)
        self.add_prohibit_area(self.steak, padding=0.03)

        # cache the steak's render shapes so the cooking timer can recolor it
        self._steak_shapes = []
        for c in self.steak.actor.get_components():
            if isinstance(c, sapien.render.RenderBodyComponent):
                self._steak_shapes = list(c.render_shapes)
        self._set_meat_color(0.0)

    # -------------------------------------------------------- cooking state
    def _set_meat_color(self, doneness):
        d = float(np.clip(doneness, 0.0, 1.0))
        # piecewise-linear interpolation across the multi-stop gradient
        stops = self.COLOR_STOPS
        for i in range(len(stops) - 1):
            d0, c0 = stops[i]
            d1, c1 = stops[i + 1]
            if d <= d1 or i == len(stops) - 2:
                t = 0.0 if d1 == d0 else (d - d0) / (d1 - d0)
                t = float(np.clip(t, 0.0, 1.0))
                rgb = [c0[k] + (c1[k] - c0[k]) * t for k in range(3)]
                break
        col = rgb + [1.0]
        for s in self._steak_shapes:
            try:
                s.material.set_base_color(col)
            except Exception:
                pass

    def _update_kinematic_tasks(self):
        # base hook drives DOMINO's dynamic object motion; runs every physics step
        super()._update_kinematic_tasks()
        if getattr(self, "_cooking_active", False):
            try:
                on_pan = self.check_actors_contact("200_steak", "106_skillet")
            except Exception:
                on_pan = False
            if on_pan:
                self.doneness = min(1.0, self.doneness + 1.0 / max(1, self.cook_steps))
                self.max_doneness = max(self.max_doneness, self.doneness)
                self._set_meat_color(self.doneness)

    def _cook_idle(self):
        # dwell on the pan, recording frames, until the target doneness is reached
        max_steps = int(round(self.target_doneness * self.cook_steps)) + 30
        for i in range(max_steps):
            self._update_kinematic_tasks()
            self.scene.step()
            if self.save_freq and (i % self.save_freq == 0):
                self._take_picture()
            if self.doneness >= self.target_doneness:
                break

    # ------------------------------------------------------------- policy
    def play_once(self):
        if self.use_dynamic:
            return self._play_once_dynamic()
        return self._play_once_static()

    def _pan_cook_table(self, arm_tag):
        # with the steak held: seat it in the pan, cook it, lift it off, then set it back on the table.
        # place_actor aligns the steak into the pan bowl reliably (a bare drop misses the small bowl).
        pan_target = list(self.skillet.get_functional_point(0))
        pan_target[0] += self.place_dx   # offset so the steak lands centered in the bowl
        pan_target[1] += self.place_dy
        self.move(
            self.place_actor(
                self.steak, target_pose=pan_target, arm_tag=arm_tag,
                constrain="free", pre_dis=0.08, dis=0.03, is_open=True,
            ))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self.move(self.back_to_origin(arm_tag))

        self._cooking_active = True
        self._cook_idle()
        self._grasp_doneness = self.doneness
        self._cooking_active = False  # stop the timer before the grasp-off motion

        # grasp the cooked steak off the pan
        self.move(self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.12, move_axis="arm"))

        # set it back down on the table, clear of the pan. Relative displacement moves (shift outboard
        # toward the arm's side, then lower) plan far more reliably than an absolute move_to_pose.
        self.move(self.move_by_displacement(arm_tag=arm_tag, x=self._side * 0.18))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=-0.06))
        self.move(self.open_gripper(arm_tag))

    def _play_once_static(self):
        arm_tag = ArmTag("right" if self.steak.get_pose().p[0] > 0 else "left")
        # grasp the raw steak off the table
        self.move(self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1))
        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table(arm_tag)
        self.info["info"] = {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    def get_dynamic_motion_config(self) -> dict:
        if not self.use_dynamic:
            return None
        p = self.steak.get_pose().p
        return {
            "target_actor": self.steak,
            "end_position": np.array([p[0], p[1], p[2]]),
            "table_bounds": (-0.35, 0.35, -0.25, 0.15),
            "check_z_threshold": 0.03,
            "check_z_actor": self.steak,
        }

    def _play_once_dynamic(self):
        arm_tag = ArmTag("right" if self.steak.get_pose().p[0] > 0 else "left")
        p = self.steak.get_pose().p
        self.end_position = np.array([p[0], p[1], p[2]])

        def robot_action_sequence(need_plan_mode):
            gr = self.grasp_actor(self.steak, arm_tag=arm_tag, pre_grasp_dis=0.1)
            if not gr or gr[1] is None or len(gr[1]) == 0:
                return
            self.move(gr)

        table_bounds = (-0.35, 0.35, -0.25, 0.15)
        success, _ = self.execute_dynamic_workflow(
            target_actor=self.steak,
            end_position=self.end_position,
            robot_action_sequence=robot_action_sequence,
            table_bounds=table_bounds,
        )
        if not success:
            print("Dynamic trajectory failed, fallback to static")
            return self._play_once_static()

        for c in self.steak.actor.get_components():
            if isinstance(c, sapien.physx.PhysxRigidDynamicComponent):
                try:
                    c.set_linear_velocity(np.zeros(3))
                    c.set_angular_velocity(np.zeros(3))
                    c.set_linear_damping(15.0)
                    c.set_angular_damping(40.0)
                except Exception:
                    pass

        self.move(self.move_by_displacement(arm_tag=arm_tag, z=0.1, move_axis="arm"))
        self._pan_cook_table(arm_tag)
        self.info["info"] = {
            "{A}": "200_steak/base0",
            "{B}": f"106_skillet/base{self.skillet_id}",
            "{a}": str(arm_tag),
        }
        return self.info

    # ------------------------------------------------------------- success
    def check_success(self):
        if self._grasp_doneness is None:
            return False
        cooked_ok = self.max_doneness >= self.target_doneness - 0.05
        steak_z = float(self.steak.get_pose().p[2])
        # success just needs the cooked steak set back down ON THE TABLE (not on the floor, not still
        # held aloft) and off the pan -- no requirement to return it to its exact start spot.
        on_table = (0.70 + self.table_z_bias) < steak_z < (0.80 + self.table_z_bias)
        # "off the pan" = horizontally clear of the pan bowl (a steak resting on the table that merely
        # grazes the pan still counts as plated back; only being IN the pan should fail)
        steak_xy = np.array(self.steak.get_pose().p[:2])
        pan_xy = np.array(self.skillet.get_functional_point(0)[:2])
        off_pan = float(np.linalg.norm(steak_xy - pan_xy)) > 0.16
        return bool(cooked_ok and on_table and off_pan)

    # record the cooking state into the trajectory (per-frame)
    def get_obs(self):
        obs = super().get_obs()
        obs["cooking"] = {
            "doneness": float(getattr(self, "doneness", 0.0)),
            "target_doneness": float(getattr(self, "target_doneness", self.TARGET_DONENESS_DEFAULT)),
            "cook_steps": float(getattr(self, "cook_steps", self.COOK_STEPS_DEFAULT)),
        }
        # record steak-vs-pan offset to measure/center the placement
        try:
            sxy = np.array(self.steak.get_pose().p)[:2]
            pxy = np.array(self.skillet.get_functional_point(0))[:2]
            obs["cooking"]["steak_xy"] = [float(sxy[0]), float(sxy[1])]
            obs["cooking"]["pan_xy"] = [float(pxy[0]), float(pxy[1])]
            obs["cooking"]["place_offset"] = [float(sxy[0] - pxy[0]), float(sxy[1] - pxy[1])]
        except Exception:
            pass
        return obs
