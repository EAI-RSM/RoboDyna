#!/usr/bin/env python3
"""Turn a GLB mesh into a RoboTwin/DOMINO benchmark object.

Handles the non-obvious parts learned while adding 200_steak:
  - bakes the GLB scene-graph transform into the vertices (else SAPIEN loads a tiny mesh)
  - optionally strips the texture so base_color is recolorable at runtime (cooking-style states)
  - scales to a real-world footprint
  - authors model_data0.json with a top-center grasp frame (4 yaw variants) + a bottom-center
    functional point, plus points_info.json and a NOTICE for license provenance

Run inside the domino env. Example:
  python integrate_object.py --src raw.glb --name 200_steak --width 0.10 --strip-texture \
      --source-url https://poly.pizza/m/xxx --author "Name" --license CC0

Grasp/functional frames default to a flat-object convention (thickness along local Y, like a steak
or bread). Validate with validate_asset.py and a smoke collection; tune the frames if grasp planning
fails.
"""
import argparse, json, os, shutil, subprocess, sys
import numpy as np
import trimesh

HAMBURG_GRASP_ROTS = [               # 4 yaw variants of a top-down grasp, proven on flat foods
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    [[0, 0, 1], [0, 1, 0], [-1, 0, 0]],
    [[-1, 0, 0], [0, 1, 0], [0, 0, -1]],
    [[0, 0, -1], [0, 1, 0], [1, 0, 0]],
]
FUNCTIONAL_ROT = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]


def mat(R, t):
    m = np.eye(4); m[:3, :3] = np.array(R, float); m[:3, 3] = t; return m.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source .glb path or http(s) URL")
    ap.add_argument("--name", required=True, help="object dir name, e.g. 200_steak")
    ap.add_argument("--width", type=float, default=0.10, help="target footprint (m) on the larger XZ axis")
    ap.add_argument("--strip-texture", action="store_true", help="bake a plain material (recolorable)")
    ap.add_argument("--base-color", default="220,220,220", help="plain material color when stripping")
    ap.add_argument("--dest-root", default="/shared_work/markhsp/DOMINO/assets/objects")
    ap.add_argument("--source-url", default="(unknown)")
    ap.add_argument("--author", default="(unknown)")
    ap.add_argument("--license", default="(unknown — verify before use)")
    args = ap.parse_args()

    src = args.src
    if src.startswith("http"):
        local = f"/tmp/{args.name}_src.glb"
        subprocess.check_call(["curl", "-fsSL", "-o", local, src]); src = local

    scene = trimesh.load(src)
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else (
        scene.dump(concatenate=True) if hasattr(scene, "dump") else scene)

    if args.strip_texture:
        col = [int(x) for x in args.base_color.split(",")] + [255]
        mesh.visual = trimesh.visual.TextureVisuals(
            material=trimesh.visual.material.PBRMaterial(baseColorFactor=col[:4]))

    dst = os.path.join(args.dest_root, args.name)
    os.makedirs(f"{dst}/visual", exist_ok=True)
    os.makedirs(f"{dst}/collision", exist_ok=True)
    mesh.export(f"{dst}/visual/base0.glb")
    mesh.export(f"{dst}/collision/base0.glb")

    ext = np.asarray(mesh.extents, float)            # raw, transform-baked
    center = (mesh.bounds[1] + mesh.bounds[0]) / 2
    scale = float(round(args.width / max(ext[0], ext[2]), 4))
    ymin, ymax = float(mesh.bounds[0][1]), float(mesh.bounds[1][1])
    cx, cz = float(center[0]), float(center[2])

    model_data = {
        "center": center.tolist(),
        "extents": ext.tolist(),
        "scale": [scale, scale, scale],
        "transform_matrix": np.eye(4).tolist(),
        "target_pose": [],
        "contact_points_pose": [mat(R, [cx, ymax, cz]) for R in HAMBURG_GRASP_ROTS],
        "functional_matrix": [mat(FUNCTIONAL_ROT, [cx, ymin, cz])],
        "orientation_point": [],
        "contact_points_group": [],
        "contact_points_mask": [],
        "target_point_discription": [],
        "contact_points_discription": ["Top-center, grasped from above."],
        "functional_point_discription": ["Bottom-center; placed onto / lifted off a target."],
        "orientation_point_discription": [],
        "stable": True,
    }
    json.dump(model_data, open(f"{dst}/model_data0.json", "w"), indent=2)
    json.dump({
        "contact_points": [{"id": [0, 1, 2, 3], "description": "Top-center",
                            "usage": "Grasp from above."}],
        "functional_points": [{"id": 0, "description": "Bottom-center",
                               "usage": "Place onto / lift off a target."}],
    }, open(f"{dst}/points_info.json", "w"), indent=2)
    open(f"{dst}/NOTICE", "w").write(
        f"Asset: {args.name}\nSource: {args.source_url}\nAuthor: {args.author}\n"
        f"License: {args.license}\n")

    print(f"wrote {dst}")
    print(f"  scale={scale}  world dims (m)={np.round(ext * scale, 4).tolist()}")
    print(f"  grasp(top) raw={[round(cx,4), round(ymax,4), round(cz,4)]}  "
          f"functional(bottom) raw={[round(cx,4), round(ymin,4), round(cz,4)]}")
    print("  next: validate_asset.py --name", args.name, "then a 2-episode smoke collection")


if __name__ == "__main__":
    main()
