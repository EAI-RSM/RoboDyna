"""Keycap PNGs for tutorial HUD overlays (same look as the arrow-key demo)."""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

KEY_SIZE = 180
RADIUS = 28
PAD = 16
GAP = 14
CANVAS_PAD = 12
SUBLABEL_H = 38
INSTRUCTION_H = 78

ARM_SUBLABELS = ("left", "right", "both")
ARM_INSTRUCTION = "Select gripper"
VIEW_INSTRUCTION = "Switch view,\nhead camera and gripper"
ARROW_INSTRUCTION = "Move the arm left, right, forward, and back"
HEIGHT_SUBLABELS = ("up", "down")
HEIGHT_INSTRUCTION = "Raise and lower the arm.\nZ min and max is capped."
YAW_SUBLABELS = ("left", "right")
YAW_INSTRUCTION = "Rotate the gripper left or right"
TILT_SUBLABELS = ("left", "right")
TILT_INSTRUCTION = "Tilt the gripper left or right"
SPACE_INSTRUCTION = "Open and close the gripper"
SPACE_BAR_W = KEY_SIZE * 2 + GAP
GRASP_INSTRUCTION = "Arrows to the cube. Space closes; E lifts."
HOLD_INSTRUCTION = "Q to press and hold; E to release"
SWITCH_INSTRUCTION = "Press ON (red), press again OFF (green)"
PUSH_INSTRUCTION = "Close the gripper, then push the box to the green line"

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
)

_FILL = (36, 38, 44, 255)
_FILL_INNER = (48, 51, 58, 255)
_OUTLINE = (210, 214, 220, 255)
_INNER_OUTLINE = (88, 92, 102, 255)
_GLYPH = (245, 247, 250, 255)
_FILL_DONE = (28, 72, 46, 255)
_FILL_INNER_DONE = (36, 92, 58, 255)
_OUTLINE_DONE = (120, 220, 150, 255)
_INNER_OUTLINE_DONE = (70, 160, 100, 255)
_GLYPH_DONE = (230, 255, 236, 255)
_CANVAS = (18, 18, 22, 230)
_SUBLABEL = (200, 204, 212, 255)
_INSTRUCTION = (232, 234, 238, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _rounded_rect(draw, xy, radius, fill, outline=None, width=3):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_keycap(label: str, *, pressed: bool = False, size: int = KEY_SIZE) -> Image.Image:
    scale = size / float(KEY_SIZE)
    pad = max(6, int(round(PAD * scale)))
    radius = max(8, int(round(RADIUS * scale)))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = _FILL_DONE if pressed else _FILL
    inner = _FILL_INNER_DONE if pressed else _FILL_INNER
    outline = _OUTLINE_DONE if pressed else _OUTLINE
    inner_outline = _INNER_OUTLINE_DONE if pressed else _INNER_OUTLINE
    glyph = _GLYPH_DONE if pressed else _GLYPH
    _rounded_rect(
        d,
        [pad + 3, pad + 6, size - pad + 3, size - pad + 6],
        radius,
        fill=(0, 0, 0, 90),
    )
    _rounded_rect(
        d,
        [pad, pad, size - pad, size - pad],
        radius,
        fill=fill,
        outline=outline,
        width=max(2, int(round(4 * scale))),
    )
    inset = max(5, int(round(8 * scale)))
    _rounded_rect(
        d,
        [pad + inset, pad + inset, size - pad - inset, size - pad - inset],
        max(4, radius - inset),
        fill=inner,
        outline=inner_outline,
        width=max(1, int(round(2 * scale))),
    )
    text = str(label)
    if len(text) == 1:
        font_px = int(round(92 * scale))
    elif len(text) == 2:
        font_px = int(round(64 * scale))
    else:
        font_px = int(round(48 * scale))
    font = _font(max(16, font_px))
    bbox = d.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) / 2 - bbox[0]
    y = (size - th) / 2 - bbox[1] - 2
    d.text((x, y), text, font=font, fill=glyph)
    return img


def cluster_size() -> tuple[int, int]:
    return (
        KEY_SIZE * 3 + GAP * 2 + CANVAS_PAD * 2,
        KEY_SIZE + CANVAS_PAD * 2 + SUBLABEL_H + INSTRUCTION_H,
    )


def _wrap_text(text: str, font, draw, max_width: int) -> list[str]:
    lines: list[str] = []
    for para in text.split("\n"):
        current = ""
        for word in para.split():
            trial = word if not current else f"{current} {word}"
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines or [text]


def _draw_centered_lines(draw, lines: list[str], font, y: int, canvas_w: int, fill) -> None:
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (canvas_w - tw) / 2 - bbox[0]
        draw.text((x, y - bbox[1]), line, font=font, fill=fill)
        y += th + 4


def draw_key_row(
    labels: tuple[str, ...],
    pressed: set[str] | None = None,
    *,
    sublabels: tuple[str, ...] | None = None,
    instruction: str = "",
) -> Image.Image:
    pressed = pressed or set()
    width, height = cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    n = len(labels)
    row_w = n * KEY_SIZE + max(0, n - 1) * GAP
    x0 = (width - row_w) // 2
    y0 = CANVAS_PAD
    sub_font = _font(28)
    for i, label in enumerate(labels):
        cap = draw_keycap(label, pressed=label in pressed)
        x = x0 + i * (KEY_SIZE + GAP)
        canvas.alpha_composite(cap, (x, y0))
        if sublabels is not None and i < len(sublabels):
            hint = sublabels[i]
            bbox = d.textbbox((0, 0), hint, font=sub_font)
            tw = bbox[2] - bbox[0]
            tx = x + (KEY_SIZE - tw) / 2 - bbox[0]
            ty = y0 + KEY_SIZE + 4 - bbox[1]
            d.text((tx, ty), hint, font=sub_font, fill=_SUBLABEL)
    if instruction:
        inst_font = _font(26)
        max_w = width - CANVAS_PAD * 2
        lines = _wrap_text(instruction, inst_font, d, max_w)
        inst_y = y0 + KEY_SIZE + (SUBLABEL_H if sublabels else 10) + 4
        _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_arm_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("1", "2", "3"),
        pressed,
        sublabels=ARM_SUBLABELS,
        instruction=ARM_INSTRUCTION,
    )


def draw_view_key(pressed: bool = False) -> Image.Image:
    return draw_key_row(
        ("V",),
        {"V"} if pressed else set(),
        instruction=VIEW_INSTRUCTION,
    )


PLAY_KEY_SIZE = 112
PLAY_GAP = 10
PLAY_PAD = 10
PLAY_SUB_H = 34
PLAY_LABELS = ("1", "2", "3", "V", "Esc")
PLAY_SUBLABELS = ("left", "right", "both", "view", "quit")


def play_cluster_size() -> tuple[int, int]:
    n = len(PLAY_LABELS)
    return (
        n * PLAY_KEY_SIZE + (n - 1) * PLAY_GAP + PLAY_PAD * 2,
        PLAY_KEY_SIZE + PLAY_PAD * 2 + PLAY_SUB_H,
    )


def draw_play_keys(pressed: set[str] | None = None) -> Image.Image:
    """Compact 1 / 2 / 3 / V / Esc strip used after the Part 1 lesson."""
    pressed = pressed or set()
    width, height = play_cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    sub_font = _font(20)
    y0 = PLAY_PAD
    x0 = PLAY_PAD
    for i, label in enumerate(PLAY_LABELS):
        cap = draw_keycap(label, pressed=label in pressed, size=PLAY_KEY_SIZE)
        x = x0 + i * (PLAY_KEY_SIZE + PLAY_GAP)
        canvas.alpha_composite(cap, (x, y0))
        hint = PLAY_SUBLABELS[i]
        bbox = d.textbbox((0, 0), hint, font=sub_font)
        tw = bbox[2] - bbox[0]
        tx = x + (PLAY_KEY_SIZE - tw) / 2 - bbox[0]
        ty = y0 + PLAY_KEY_SIZE + 2 - bbox[1]
        d.text((tx, ty), hint, font=sub_font, fill=_SUBLABEL)
    return canvas


PLAY2_KEY = 72
PLAY2_GAP = 8
PLAY2_GROUP = 16
PLAY2_PAD = 10
PLAY2_CAPTION = "Practice — keys flash. Esc quits."


def _play2_arrow_block() -> tuple[int, int]:
    k, g = PLAY2_KEY, PLAY2_GAP
    return (3 * k + 2 * g, 2 * k + g)


def part2_play_cluster_size() -> tuple[int, int]:
    aw, ah = _play2_arrow_block()
    letters_w = 6 * PLAY2_KEY + 3 * PLAY2_GAP + 2 * PLAY2_GROUP
    width = PLAY2_PAD + aw + PLAY2_GROUP + letters_w + PLAY2_PAD
    height = PLAY2_PAD + ah + PLAY2_GAP + PLAY2_KEY + PLAY2_PAD + PLAY_SUB_H
    return width, height


def draw_part2_play_keys(pressed: set[str] | None = None) -> Image.Image:
    """Compact arrows + E/Q + R/T + F/G + Space + Esc for Part 2 practice."""
    pressed = pressed or set()
    width, height = part2_play_cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    k, g = PLAY2_KEY, PLAY2_GAP
    x0 = y0 = PLAY2_PAD
    aw, ah = _play2_arrow_block()
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed, size=k),
        (x0 + k + g, y0),
    )
    by = y0 + k + g
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed, size=k),
        (x0, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed, size=k),
        (x0 + k + g, by),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed, size=k),
        (x0 + 2 * (k + g), by),
    )
    lx = x0 + aw + PLAY2_GROUP
    groups = (("E", "Q"), ("R", "T"), ("F", "G"))
    for gi, pair in enumerate(groups):
        gx = lx + gi * (2 * k + PLAY2_GAP + PLAY2_GROUP)
        for j, label in enumerate(pair):
            cap = draw_keycap(label, pressed=label in pressed, size=k)
            canvas.alpha_composite(cap, (gx + j * (k + g), by))
    sy = y0 + ah + PLAY2_GAP
    canvas.alpha_composite(
        draw_wide_keycap("Space", pressed="Space" in pressed, width=aw, height=k),
        (x0, sy),
    )
    canvas.alpha_composite(
        draw_keycap("Esc", pressed="Esc" in pressed, size=k),
        (lx, sy),
    )
    inst_font = _font(22)
    lines = _wrap_text(PLAY2_CAPTION, inst_font, d, width - PLAY2_PAD * 2)
    _draw_centered_lines(d, lines, inst_font, sy + k + 6, width, _INSTRUCTION)
    return canvas


def _arrow_pts(cx, cy, direction, length=46, width=38):
    if direction == "up":
        return [
            (cx, cy - length),
            (cx - width, cy + length * 0.45),
            (cx - width * 0.28, cy + length * 0.45),
            (cx - width * 0.28, cy + length),
            (cx + width * 0.28, cy + length),
            (cx + width * 0.28, cy + length * 0.45),
            (cx + width, cy + length * 0.45),
        ]
    if direction == "down":
        return [
            (cx, cy + length),
            (cx - width, cy - length * 0.45),
            (cx - width * 0.28, cy - length * 0.45),
            (cx - width * 0.28, cy - length),
            (cx + width * 0.28, cy - length),
            (cx + width * 0.28, cy - length * 0.45),
            (cx + width, cy - length * 0.45),
        ]
    if direction == "left":
        return [
            (cx - length, cy),
            (cx + length * 0.45, cy - width),
            (cx + length * 0.45, cy - width * 0.28),
            (cx + length, cy - width * 0.28),
            (cx + length, cy + width * 0.28),
            (cx + length * 0.45, cy + width * 0.28),
            (cx + length * 0.45, cy + width),
        ]
    return [
        (cx + length, cy),
        (cx - length * 0.45, cy - width),
        (cx - length * 0.45, cy - width * 0.28),
        (cx - length, cy - width * 0.28),
        (cx - length, cy + width * 0.28),
        (cx - length * 0.45, cy + width * 0.28),
        (cx - length * 0.45, cy + width),
    ]


def draw_arrow_keycap(direction: str, *, pressed: bool = False, size: int = KEY_SIZE) -> Image.Image:
    scale = size / float(KEY_SIZE)
    pad = max(6, int(round(PAD * scale)))
    radius = max(8, int(round(RADIUS * scale)))
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = _FILL_DONE if pressed else _FILL
    inner = _FILL_INNER_DONE if pressed else _FILL_INNER
    outline = _OUTLINE_DONE if pressed else _OUTLINE
    inner_outline = _INNER_OUTLINE_DONE if pressed else _INNER_OUTLINE
    glyph = _GLYPH_DONE if pressed else _GLYPH
    _rounded_rect(
        d,
        [pad + 3, pad + 6, size - pad + 3, size - pad + 6],
        radius,
        fill=(0, 0, 0, 90),
    )
    _rounded_rect(
        d,
        [pad, pad, size - pad, size - pad],
        radius,
        fill=fill,
        outline=outline,
        width=max(2, int(round(4 * scale))),
    )
    inset = max(5, int(round(8 * scale)))
    _rounded_rect(
        d,
        [pad + inset, pad + inset, size - pad - inset, size - pad - inset],
        max(4, radius - inset),
        fill=inner,
        outline=inner_outline,
        width=max(1, int(round(2 * scale))),
    )
    cx = cy = size / 2
    d.polygon(
        _arrow_pts(cx, cy - 2 * scale, direction, length=46 * scale, width=38 * scale),
        fill=glyph,
    )
    return img


def arrow_cluster_size() -> tuple[int, int]:
    return (
        KEY_SIZE * 3 + GAP * 2 + CANVAS_PAD * 2,
        KEY_SIZE * 2 + GAP + CANVAS_PAD * 2 + INSTRUCTION_H,
    )


def draw_arrow_cluster(pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    width, height = arrow_cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    x0, y0 = CANVAS_PAD, CANVAS_PAD
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed),
        (x0 + KEY_SIZE + GAP, y0),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed),
        (x0, y0 + KEY_SIZE + GAP),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed),
        (x0 + KEY_SIZE + GAP, y0 + KEY_SIZE + GAP),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed),
        (x0 + 2 * (KEY_SIZE + GAP), y0 + KEY_SIZE + GAP),
    )
    inst_font = _font(26)
    max_w = width - CANVAS_PAD * 2
    lines = _wrap_text(ARROW_INSTRUCTION, inst_font, d, max_w)
    inst_y = y0 + KEY_SIZE * 2 + GAP + 8
    _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_height_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("E", "Q"),
        pressed,
        sublabels=HEIGHT_SUBLABELS,
        instruction=HEIGHT_INSTRUCTION,
    )


def draw_yaw_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("R", "T"),
        pressed,
        sublabels=YAW_SUBLABELS,
        instruction=YAW_INSTRUCTION,
    )


def draw_tilt_keys(pressed: set[str] | None = None) -> Image.Image:
    return draw_key_row(
        ("F", "G"),
        pressed,
        sublabels=TILT_SUBLABELS,
        instruction=TILT_INSTRUCTION,
    )


def draw_wide_keycap(
    label: str,
    *,
    pressed: bool = False,
    width: int = SPACE_BAR_W,
    height: int = KEY_SIZE,
) -> Image.Image:
    scale = height / float(KEY_SIZE)
    pad = max(6, int(round(PAD * scale)))
    radius = max(8, int(round(RADIUS * scale)))
    img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = _FILL_DONE if pressed else _FILL
    inner = _FILL_INNER_DONE if pressed else _FILL_INNER
    outline = _OUTLINE_DONE if pressed else _OUTLINE
    inner_outline = _INNER_OUTLINE_DONE if pressed else _INNER_OUTLINE
    glyph = _GLYPH_DONE if pressed else _GLYPH
    _rounded_rect(
        d,
        [pad + 3, pad + 6, width - pad + 3, height - pad + 6],
        radius,
        fill=(0, 0, 0, 90),
    )
    _rounded_rect(
        d,
        [pad, pad, width - pad, height - pad],
        radius,
        fill=fill,
        outline=outline,
        width=max(2, int(round(4 * scale))),
    )
    inset = max(5, int(round(8 * scale)))
    _rounded_rect(
        d,
        [pad + inset, pad + inset, width - pad - inset, height - pad - inset],
        max(4, radius - inset),
        fill=inner,
        outline=inner_outline,
        width=max(1, int(round(2 * scale))),
    )
    font = _font(max(16, int(round(56 * scale))))
    bbox = d.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (width - tw) / 2 - bbox[0]
    y = (height - th) / 2 - bbox[1] - 2
    d.text((x, y), label, font=font, fill=glyph)
    return img


def draw_space_key(pressed: bool = False) -> Image.Image:
    width, height = cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    cap = draw_wide_keycap("Space", pressed=pressed)
    x0 = (width - cap.width) // 2
    y0 = CANVAS_PAD
    canvas.alpha_composite(cap, (x0, y0))
    inst_font = _font(26)
    max_w = width - CANVAS_PAD * 2
    lines = _wrap_text(SPACE_INSTRUCTION, inst_font, d, max_w)
    inst_y = y0 + KEY_SIZE + 14
    _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_control_stage(stage: str, pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    if stage == "arrows":
        return draw_arrow_cluster(pressed)
    if stage == "height":
        return draw_height_keys(pressed)
    if stage == "yaw":
        return draw_yaw_keys(pressed)
    if stage == "tilt":
        return draw_tilt_keys(pressed)
    if stage == "space":
        return draw_space_key(pressed="Space" in pressed)
    raise ValueError(f"Unknown control stage: {stage}")


def draw_space_plus_key(
    letter: str,
    *,
    space_pressed: bool = False,
    letter_pressed: bool = False,
    space_sub: str = "",
    letter_sub: str = "",
    instruction: str = "",
) -> Image.Image:
    """Wide Space key beside a letter key, with optional sublabels."""
    width, height = cluster_size()
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    space = draw_wide_keycap("Space", pressed=space_pressed)
    key = draw_keycap(letter, pressed=letter_pressed)
    row_w = space.width + GAP + KEY_SIZE
    x0 = (width - row_w) // 2
    y0 = CANVAS_PAD
    canvas.alpha_composite(space, (x0, y0))
    canvas.alpha_composite(key, (x0 + space.width + GAP, y0))
    sub_font = _font(28)
    if space_sub:
        bbox = d.textbbox((0, 0), space_sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        tx = x0 + (space.width - tw) / 2 - bbox[0]
        ty = y0 + KEY_SIZE + 4 - bbox[1]
        d.text((tx, ty), space_sub, font=sub_font, fill=_SUBLABEL)
    if letter_sub:
        bbox = d.textbbox((0, 0), letter_sub, font=sub_font)
        tw = bbox[2] - bbox[0]
        kx = x0 + space.width + GAP
        tx = kx + (KEY_SIZE - tw) / 2 - bbox[0]
        ty = y0 + KEY_SIZE + 4 - bbox[1]
        d.text((tx, ty), letter_sub, font=sub_font, fill=_SUBLABEL)
    if instruction:
        inst_font = _font(26)
        max_w = width - CANVAS_PAD * 2
        lines = _wrap_text(instruction, inst_font, d, max_w)
        inst_y = y0 + KEY_SIZE + SUBLABEL_H + 4
        _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_grasp_keys(pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    return draw_space_plus_key(
        "E",
        space_pressed="Space" in pressed,
        letter_pressed="E" in pressed,
        space_sub="close",
        letter_sub="lift",
        instruction=GRASP_INSTRUCTION,
    )


def draw_hold_keys(pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    return draw_key_row(
        ("Q", "E"),
        pressed,
        sublabels=("press", "release"),
        instruction=HOLD_INSTRUCTION,
    )


def draw_switch_keys(pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    return draw_key_row(
        ("Q",),
        pressed,
        instruction=SWITCH_INSTRUCTION,
    )


def draw_push_keys(pressed: set[str] | None = None) -> Image.Image:
    """Space bar above the arrow cluster for the push-box stage."""
    pressed = pressed or set()
    arrow_w, _arrow_h = arrow_cluster_size()
    width = arrow_w
    height = (
        CANVAS_PAD
        + KEY_SIZE
        + GAP
        + KEY_SIZE * 2
        + GAP
        + INSTRUCTION_H
        + CANVAS_PAD
    )
    canvas = Image.new("RGBA", (width, height), _CANVAS)
    d = ImageDraw.Draw(canvas)
    space = draw_wide_keycap("Space", pressed="Space" in pressed)
    x0 = (width - space.width) // 2
    y0 = CANVAS_PAD
    canvas.alpha_composite(space, (x0, y0))
    ay = y0 + KEY_SIZE + GAP
    canvas.alpha_composite(
        draw_arrow_keycap("up", pressed="up" in pressed),
        (CANVAS_PAD + KEY_SIZE + GAP, ay),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("left", pressed="left" in pressed),
        (CANVAS_PAD, ay + KEY_SIZE + GAP),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("down", pressed="down" in pressed),
        (CANVAS_PAD + KEY_SIZE + GAP, ay + KEY_SIZE + GAP),
    )
    canvas.alpha_composite(
        draw_arrow_keycap("right", pressed="right" in pressed),
        (CANVAS_PAD + 2 * (KEY_SIZE + GAP), ay + KEY_SIZE + GAP),
    )
    inst_font = _font(26)
    max_w = width - CANVAS_PAD * 2
    lines = _wrap_text(PUSH_INSTRUCTION, inst_font, d, max_w)
    inst_y = ay + KEY_SIZE * 2 + GAP + 8
    _draw_centered_lines(d, lines, inst_font, inst_y, width, _INSTRUCTION)
    return canvas


def draw_action_stage(stage: str, pressed: set[str] | None = None) -> Image.Image:
    pressed = pressed or set()
    if stage == "grasp":
        return draw_grasp_keys(pressed)
    if stage == "hold":
        return draw_hold_keys(pressed)
    if stage == "switch":
        return draw_switch_keys(pressed)
    if stage == "push":
        return draw_push_keys(pressed)
    raise ValueError(f"Unknown action stage: {stage}")
