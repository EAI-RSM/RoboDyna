#!/home/xuan/miniconda3/envs/robodyna/bin/python
"""Headless smoke: advance cooking and dump frames of the pie timer."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sapien

REPO = Path(__file__).resolve().parents[1]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "script" / "bench_script"))
sys.path.insert(0, str(REPO / "script_exp"))

os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)


def main() -> None:
    from envs import CONFIGS_PATH
    from envs.cook_meat_timer import cook_meat_timer
    import interactive_cook_meat_timer as icm

    globals_dict = icm.__dict__
    globals_dict["CONFIGS_PATH"] = CONFIGS_PATH
    cfg = icm._configure_task("demo_dynamic", seed=0, use_robot=False)
    cfg["render_freq"] = 0  # headless (no sapien Viewer window)
    targs = cfg.setdefault("task_args", {}).setdefault("cook_meat_timer", {})
    targs.update(
        dual_setup_enabled=False,
        cook_steps=200,
        cook_speed_jitter=0.0,
        target_doneness_range=[0.45, 0.55],
        target_doneness_range_jitter=0.0,
        timer_n_sections=24,
    )

    env = cook_meat_timer()
    env.setup_demo(**cfg)
    st = env.stations[0]
    pan = list(env._pan_place_target(st))
    pan[2] += 0.012
    st["steak"].actor.set_pose(sapien.Pose(pan[:3], st["steak"].get_pose().q))
    st["cook_on"] = True

    out = Path("/tmp/robodyna_demos/cook_meat_timer_smoke")
    out.mkdir(parents=True, exist_ok=True)
    max_steps = int(0.75 * env.cook_steps) + 20
    save_every = max(1, max_steps // 24)
    for i in range(max_steps):
        env._update_kinematic_tasks()
        env.scene.step()
        if i % save_every == 0 or i == max_steps - 1:
            env.cameras.update_picture()
            rgb = env.cameras.get_rgb()
            frame = None
            for key in ("observer_camera", "head_camera"):
                if key in rgb:
                    frame = rgb[key]
                    break
            if frame is None and rgb:
                frame = next(iter(rgb.values()))
            if frame is not None:
                from PIL import Image

                phase = st.get("pie_timer", {}).get("phase")
                fill = st.get("pie_timer", {}).get("fill")
                path = out / f"f_{i:04d}_d{st['doneness']:.2f}_{phase}.png"
                Image.fromarray(np.asarray(frame)).save(path)
                print(
                    f"saved {path.name} doneness={st['doneness']:.3f} "
                    f"phase={phase} fill={fill}",
                    flush=True,
                )
    print("done", flush=True)
    env.close_env()


if __name__ == "__main__":
    main()
