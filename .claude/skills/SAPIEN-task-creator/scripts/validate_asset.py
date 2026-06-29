#!/usr/bin/env python3
"""Offscreen-render a benchmark object at its authored scale, before writing task code.

Catches the two cheap-to-miss problems early: a wrong scale (object renders tiny/huge) and a
non-recolorable material (textured GLB ignores base_color). Uses flat ambient lighting so the
albedo/base_color is visible rather than blown out by directional light.

Run inside the domino env with the Vulkan vars set:
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json; unset DISPLAY
  python validate_asset.py --name 200_steak --recolor

Writes PNGs to --out and prints object pixel coverage + (with --recolor) whether raw vs cooked differ.
"""
import argparse, json, os
os.environ.setdefault("VK_ICD_FILENAMES", "/usr/share/vulkan/icd.d/nvidia_icd.json")
os.environ.pop("DISPLAY", None)
import numpy as np
import sapien
import sapien.render
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="object dir under assets/objects, e.g. 200_steak")
    ap.add_argument("--model-id", type=int, default=0)
    ap.add_argument("--root", default="/shared_work/markhsp/DOMINO/assets/objects")
    ap.add_argument("--recolor", action="store_true", help="test a base_color raw->cooked change")
    ap.add_argument("--out", default="/tmp/asset_validate")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = os.path.join(args.root, args.name)
    md = json.load(open(f"{d}/model_data{args.model_id}.json"))
    scale = md.get("scale", [1, 1, 1])
    ext = np.asarray(md.get("extents", [0.2, 0.2, 0.2])) * np.asarray(scale)

    scene = sapien.Scene(); scene.set_timestep(1 / 250)
    scene.set_ambient_light([0.85, 0.85, 0.85])     # flat -> shows albedo, not lighting
    b = scene.create_actor_builder()
    b.add_visual_from_file(filename=f"{d}/visual/base{args.model_id}.glb", scale=scale)
    a = b.build_kinematic(name=args.name)
    a.set_pose(sapien.Pose([0, 0, float(ext[1]) / 2]))

    cam = scene.add_camera("cam", 480, 480, 0.7, 0.001, 10.0)
    r = float(np.linalg.norm(ext)); eye = np.array([r, r, r * 0.9]) * 1.4 + 1e-3
    tgt = np.array([0, 0, float(ext[1]) / 2])
    fwd = (tgt - eye); fwd /= np.linalg.norm(fwd)
    left = np.cross([0, 0, 1], fwd); left /= np.linalg.norm(left); up = np.cross(fwd, left)
    m = np.eye(4); m[:3, 0] = fwd; m[:3, 1] = left; m[:3, 2] = up; m[:3, 3] = eye
    cam.set_pose(sapien.Pose(m))

    shapes = [s for c in a.get_components()
              if isinstance(c, sapien.render.RenderBodyComponent) for s in c.render_shapes]

    def shoot(tag, color=None):
        if color is not None:
            for s in shapes:
                try: s.material.set_base_color(color)
                except Exception as e: print("recolor error:", repr(e))
        scene.step(); scene.update_render(); cam.take_picture()
        rgb = (cam.get_picture("Color")[..., :3] * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(rgb).save(f"{args.out}/{args.name}_{tag}.png")
        return rgb.astype(float)

    print(f"world dims (m): {np.round(ext, 4).tolist()}  (thickness/footprint sanity-check this)")
    base = shoot("base", [0.85, 0.40, 0.38, 1.0] if args.recolor else None)
    bg = base[0, 0]
    obj_px = int((np.abs(base - bg).sum(2) > 20).sum())
    print(f"object pixels in frame: {obj_px} / {480*480}  ({100*obj_px/(480*480):.1f}%)  "
          f"-> if ~0%, the scale is wrong")

    if args.recolor:
        cooked = shoot("cooked", [0.10, 0.06, 0.04, 1.0])
        changed = int((np.abs(base - cooked).sum(2) > 10).sum())
        print(f"recolor changed pixels: {changed}  -> if 0, the GLB is textured; re-integrate "
              f"with --strip-texture")
    print("wrote PNGs to", args.out)


if __name__ == "__main__":
    main()
