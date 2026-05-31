import math
import random
import zlib
import numpy as np
from pathlib import Path
from typing import List
from PIL import Image, ImageDraw, ImageFilter
from constants import TYPE_STYLE
from models import Cell

def _make_background(w: int, h: int) -> Image.Image:
    base = np.zeros((h, w, 3), dtype=np.uint8)
    base[..., 0] = 245
    base[..., 1] = 225
    base[..., 2] = 232
    noise = np.random.normal(0, 5, (h, w, 1))
    tint = np.array([8, -2, 2], dtype=np.float32)
    arr = np.clip(base.astype(np.float32) + noise + tint, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB").filter(ImageFilter.GaussianBlur(radius=1.2))

def _draw_cell(draw: ImageDraw.ImageDraw, x: int, y: int, cell_type: str, seed: int) -> None:
    style = TYPE_STYLE.get(cell_type, TYPE_STYLE["fibroblast"])
    rng = random.Random(seed)
    r = rng.randint(style["min_r"], style["max_r"])
    elong = rng.uniform(*style["elong"])
    rx, ry = max(2, int(r * elong)), r
    angle = rng.uniform(0, math.pi)
    pts = []
    for t in np.linspace(0, 2 * math.pi, 40, endpoint=False):
        ca, sa = math.cos(angle), math.sin(angle)
        ex, ey = rx * math.cos(t), ry * math.sin(t)
        pts.append((x + ex * ca - ey * sa, y + ex * sa + ey * ca))
    draw.polygon(pts, fill=style["fill"] + (style["fill_alpha"],))
    nr = max(1, r // 3)
    draw.ellipse((x - nr, y - nr, x + nr, y + nr), fill=style["nucleus"] + (style["nucleus_alpha"],))

def generate_image(cells: List[Cell], out_path: Path) -> Image.Image:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx, my = max(c.x for c in cells), max(c.y for c in cells)
    W, H = max(2048, mx + 100), max(2048, my + 100)
    img = _make_background(W, H)
    draw = ImageDraw.Draw(img, "RGBA")
    for c in cells:
        data_str = f"{c.id}-{c.x}-{c.y}-{c.cell_type}"
        seed = zlib.adler32(data_str.encode())
        _draw_cell(draw, c.x, c.y, c.cell_type, seed)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    img.save(out_path, dpi=(600, 600))
    return img
