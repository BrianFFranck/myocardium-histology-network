import math
import random
import zlib
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .constants import TYPE_STYLE, WINDOW_SIZE
from .models import Cell


# ── Geração da imagem sintética ────────────────────────────────────

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
    draw.ellipse((x - nr, y - nr, x + nr, y + nr),
                 fill=style["nucleus"] + (style["nucleus_alpha"],))


def generate_image(cells: List[Cell], out_path: Path) -> Image.Image:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mx = max(c.x for c in cells)
    my = max(c.y for c in cells)
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


# ── Extração de métricas morfológicas ─────────────────────────────

def extract_metrics(cells: List[Cell], img: Image.Image, window: int = WINDOW_SIZE) -> None:
    """
    Extrai por célula:
      - mean_intensity, median_intensity, std_intensity,
        min_intensity, max_intensity, p25_intensity, p75_intensity
      - approx_area   : píxeis abaixo do threshold de Otsu simplificado
      - circularity   : 4π·área/perímetro²  (erosão 4-vizinhança, sem underflow)
    """
    gray = np.array(img.convert("L"), dtype=np.float32)
    h_img, w_img = gray.shape
    half = window // 2

    for cell in cells:
        x0 = max(0, cell.x - half)
        y0 = max(0, cell.y - half)
        x1 = min(w_img, cell.x + half)
        y1 = min(h_img, cell.y + half)
        patch = gray[y0:y1, x0:x1]

        if patch.size == 0:
            cell.mean_intensity = cell.median_intensity = 0.0
            cell.std_intensity  = cell.min_intensity    = 0.0
            cell.max_intensity  = cell.p25_intensity    = 0.0
            cell.p75_intensity  = 0.0
            cell.approx_area    = 0
            cell.circularity    = 0.0
            continue

        flat = patch.ravel()
        cell.mean_intensity   = float(np.mean(flat))
        cell.median_intensity = float(np.median(flat))
        cell.std_intensity    = float(np.std(flat))
        cell.min_intensity    = float(np.min(flat))
        cell.max_intensity    = float(np.max(flat))
        cell.p25_intensity    = float(np.percentile(flat, 25))
        cell.p75_intensity    = float(np.percentile(flat, 75))

        thresh = cell.mean_intensity
        binary = (patch < thresh).astype(np.uint8)
        cell.approx_area = int(np.sum(binary))

        area = cell.approx_area
        if area > 0 and binary.shape[0] > 2 and binary.shape[1] > 2:
            eroded = (
                binary[1:-1, 1:-1] &
                binary[:-2,  1:-1] &
                binary[2:,   1:-1] &
                binary[1:-1, :-2]  &
                binary[1:-1, 2:]
            )
            diff = np.clip(
                binary[1:-1, 1:-1].astype(np.int16) - eroded.astype(np.int16),
                0, 1
            )
            perimeter = int(np.sum(diff))
            cell.circularity = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        else:
            perimeter = int(np.sum(binary))
            cell.circularity = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
