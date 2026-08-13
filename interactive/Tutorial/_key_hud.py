"""Top-right SAPIEN viewer HUD that shows keycap figures via an off-stage camera."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import sapien
from PIL import Image
from sapien import internal_renderer as R
from sapien.utils.viewer.plugin import Plugin

from _keycaps import (
    cluster_size,
    draw_action_stage,
    draw_advanced_stage,
    draw_arm_keys,
    draw_control_stage,
    draw_part2_play_keys,
    draw_play_keys,
    draw_view_key,
)

_DISPLAY_WIDTH = 480
_HALF_H = 0.12
_STAGE_ORIGIN_STEP = 4.0
# ImGui title bar + window padding; keep this larger than the picture so
# the instruction line is visible without a vertical scrollbar.
_WINDOW_PAD_W = 32
_WINDOW_PAD_H = 88
_FLASH_SECONDS = 0.22


def _texture_from_image(img: Image.Image):
    arr = np.ascontiguousarray(np.array(img.convert("RGBA"), dtype=np.uint8))
    try:
        return sapien.render.RenderTexture2D(arr, "r8g8b8a8unorm", srgb=True)
    except Exception:
        path = Path("/tmp/robodyna_tutorial_hud.png")
        img.convert("RGBA").save(path)
        return sapien.render.RenderTexture2D(str(path), srgb=True)


def _textured_quad(scene, tex, half_w: float, half_h: float, name: str):
    mat = sapien.render.RenderMaterial()
    mat.set_base_color_texture(tex)
    mat.set_emission_texture(tex)
    mat.base_color = [1, 1, 1, 1]
    mat.emission = [1, 1, 1, 1]
    vertices = np.array(
        [[0, half_w, -half_h], [0, -half_w, -half_h], [0, -half_w, half_h], [0, half_w, half_h]],
        dtype=np.float32,
    )
    uvs = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
    normals = np.array([[-1, 0, 0]] * 4, dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    mesh = sapien.render.RenderShapeTriangleMesh(vertices, triangles, normals, uvs, mat)
    body = sapien.render.RenderBodyComponent()
    body.attach(mesh)
    ent = sapien.Entity()
    ent.add_component(body)
    ent.set_name(name)
    scene.add_entity(ent)
    return ent, mat


def _add_hud_camera(scene, img: Image.Image, origin, name: str):
    aspect = img.width / float(img.height)
    half_h = _HALF_H
    half_w = _HALF_H * aspect
    tex = _texture_from_image(img)
    ent, mat = _textured_quad(scene, tex, half_w, half_h, f"tutorial_hud_{name}")
    ent.set_pose(sapien.Pose(p=list(origin)))
    cam = scene.add_camera(name, img.width, img.height, 1.0, 0.05, 5.0)
    cam.set_orthographic_parameters(0.05, 5.0, -half_w, half_w, -half_h, half_h)
    cam.entity.set_pose(sapien.Pose(p=[origin[0] - 0.8, origin[1], origin[2]]))
    return cam, tex, mat


class TutorialKeyHud(Plugin):
    """ImGui window in the top-right showing the current tutorial key overlay."""

    def __init__(self, cameras, textures, materials, sizes=None, display_width=None, window_pad_h=None):
        self.cameras = cameras
        self.textures = textures
        self.materials = materials
        self.sizes = dict(sizes or {})
        self.stage = next(iter(cameras), "arms")
        self.pressed_arms: set[str] = set()
        self.v_pressed = False
        self.caption = ""
        self.on_frame = None
        self.ui_window = None
        self.ui_picture = None
        self.ui_caption = None
        self._laid_out_stage = None
        self._held: set[str] = set()
        self._flash_until: dict[str, float] = {}
        self._play_drawn: set[str] | None = None
        self.play_drawer = draw_play_keys
        self.flash_drawer = None
        self.flash_enabled = False
        self.flash_kwargs: dict = {}
        canvas_w, canvas_h = cluster_size()
        if self.sizes:
            canvas_w, canvas_h = next(iter(self.sizes.values()))
        self.pic_w = int(display_width or _DISPLAY_WIDTH)
        self.pic_h = max(1, int(round(self.pic_w * canvas_h / canvas_w)))
        self.window_pad_w = _WINDOW_PAD_W
        self.window_pad_h = int(window_pad_h if window_pad_h is not None else _WINDOW_PAD_H)

    def before_render(self):
        if self.on_frame is not None:
            viewer = getattr(self, "viewer", None)
            window = getattr(viewer, "window", None) if viewer is not None else None
            self.on_frame(window)
        if self.flash_enabled:
            self._tick_flash()

    def _display_wh(self) -> tuple[int, int]:
        size = self.sizes.get(self.stage)
        if size is None:
            for alt in ("view", "arms", "play", "arrows"):
                size = self.sizes.get(alt)
                if size is not None:
                    break
        if size is None:
            return self.pic_w, self.pic_h
        width, height = size
        pic_w = self.pic_w
        pic_h = max(1, int(round(self.pic_w * height / float(width))))
        return pic_w, pic_h

    def update_texture(self, name: str, img: Image.Image) -> None:
        tex = self.textures.get(name)
        if tex is None:
            return
        self.sizes[name] = img.size
        arr = np.ascontiguousarray(np.array(img.convert("RGBA"), dtype=np.uint8))
        try:
            tex.upload(arr)
        except Exception:
            new_tex = _texture_from_image(img)
            mat = self.materials.get(name)
            if mat is not None:
                mat.set_base_color_texture(new_tex)
                mat.set_emission_texture(new_tex)
            self.textures[name] = new_tex

    def mark_arm_pressed(self, key: str) -> None:
        if key in self.pressed_arms:
            return
        self.pressed_arms.add(key)
        self.update_texture("arms", draw_arm_keys(self.pressed_arms))

    def flash(self, label: str) -> None:
        self._flash_until[label] = time.perf_counter() + _FLASH_SECONDS

    def set_held(self, held: set[str]) -> None:
        self._held = set(held)

    def _flash_image_drawer(self):
        if self.stage == "play":
            return self.play_drawer
        return self.flash_drawer

    def _tick_flash(self) -> None:
        drawer = self._flash_image_drawer()
        if drawer is None or self.stage not in self.textures:
            return
        now = time.perf_counter()
        active = set(self._held)
        expired = [key for key, until in self._flash_until.items() if now >= until]
        for key in expired:
            self._flash_until.pop(key, None)
        for key, until in self._flash_until.items():
            if now < until:
                active.add(key)
        if active != self._play_drawn:
            self._play_drawn = set(active)
            self.update_texture(self.stage, drawer(active))

    def set_stage(self, stage: str) -> None:
        self.stage = stage
        self._held = set()
        self._flash_until = {}
        self._play_drawn = None
        drawer = self._flash_image_drawer() if self.flash_enabled else None
        if drawer is not None and stage in self.textures:
            self.update_texture(stage, drawer())
            self._play_drawn = set()
        elif stage == "done":
            self.v_pressed = True
            if "view" in self.textures:
                self.update_texture("view", draw_view_key(pressed=True))

    def get_ui_windows(self):
        cam = self.cameras.get(self.stage)
        if cam is None:
            cam = self.cameras.get("view") or next(iter(self.cameras.values()))
        try:
            cam.take_picture()
        except Exception:
            pass
        pw, ph = self._display_wh()
        if self.ui_window is None:
            self.ui_picture = R.UIPicture().Size(pw, ph)
            self.ui_window = (
                R.UIWindow()
                .Label("Keys")
                .append(self.ui_picture)
            )
        self.ui_picture.Size(pw, ph)
        self.ui_window.Size(pw + self.window_pad_w, ph + self.window_pad_h)
        if self._laid_out_stage != self.stage:
            self._laid_out_stage = self.stage
            ww = 1920
            try:
                ww = int(self.viewer.window.size[0])
            except Exception:
                pass
            self.ui_window.Pos(max(16, ww - pw - 40), 16)
        self.ui_picture.Picture(cam._internal_renderer, "Color")
        return [self.ui_window]


def build_staged_hud(
    scene,
    images: dict[str, Image.Image],
    start_stage: str,
    *,
    display_width: int | None = None,
    window_pad_h: int | None = None,
) -> TutorialKeyHud:
    """Off-stage UV quads + ortho cameras, one per named overlay stage."""
    cameras = {}
    textures = {}
    materials = {}
    sizes = {}
    for index, (name, img) in enumerate(images.items()):
        origin = (80.0, index * _STAGE_ORIGIN_STEP, 80.0)
        cam, tex, mat = _add_hud_camera(scene, img, origin, name)
        cameras[name] = cam
        textures[name] = tex
        materials[name] = mat
        sizes[name] = img.size
    hud = TutorialKeyHud(
        cameras,
        textures,
        materials,
        sizes=sizes,
        display_width=display_width,
        window_pad_h=window_pad_h,
    )
    hud.stage = start_stage
    return hud


def build_key_hud(scene) -> TutorialKeyHud:
    """Off-stage UV quads + ortho cameras for the 1/2/3 cluster and the V key."""
    hud = build_staged_hud(
        scene,
        {
            "arms": draw_arm_keys(),
            "view": draw_view_key(),
            "play": draw_play_keys(),
        },
        start_stage="arms",
    )
    hud.cameras["done"] = hud.cameras["play"]
    return hud


def build_part2_key_hud(scene) -> TutorialKeyHud:
    """Overlays for arrows → E/Q → R/T → F/G → Space, then a compact play strip."""
    stages = ("arrows", "height", "yaw", "tilt", "space")
    images = {name: draw_control_stage(name) for name in stages}
    images["play"] = draw_part2_play_keys()
    hud = build_staged_hud(scene, images, start_stage="arrows")
    hud.play_drawer = draw_part2_play_keys
    return hud


def build_part3_key_hud(scene) -> TutorialKeyHud:
    """Overlays for grasp → hold button → on/off switch → push box."""
    stages = ("grasp", "hold", "switch", "push")
    images = {name: draw_action_stage(name) for name in stages}
    hud = build_staged_hud(scene, images, start_stage="grasp")
    hud.flash_drawer = lambda pressed=None, **kwargs: draw_action_stage(hud.stage, pressed)
    hud.flash_enabled = True
    return hud


def build_part4_key_hud(scene) -> TutorialKeyHud:
    """Larger overlays for ball → stove → mallet → force key."""
    stages = ("ball", "stove", "mallet", "force_key")
    images = {name: draw_advanced_stage(name) for name in stages}
    hud = build_staged_hud(
        scene,
        images,
        start_stage="ball",
        display_width=640,
        window_pad_h=120,
    )
    hud.flash_drawer = lambda pressed=None, **kwargs: draw_advanced_stage(
        hud.stage, pressed, **hud.flash_kwargs
    )
    hud.flash_enabled = True
    return hud
