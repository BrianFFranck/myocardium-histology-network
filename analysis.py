import math
import numpy as np
from typing import List
from PIL import Image
from models import Cell
from constants import WINDOW_SIZE

def extract_metrics(cells: List[Cell], img: Image.Image, window: int = WINDOW_SIZE) -> None:
    gray = np.array(img.convert("L"), dtype=np.float32)
    h_img, w_img = gray.shape
    half = window // 2

    for cell in cells:
        x0, y0 = max(0, cell.x - half), max(0, cell.y - half)
        x1, y1 = min(w_img, cell.x + half), min(h_img, cell.y + half)
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
        
        n_flat = len(flat)
        idx25 = max(0, int(n_flat * 0.25) - 1)
        idx75 = max(0, int(n_flat * 0.75) - 1)
        partitioned = np.partition(flat, [idx25, idx75])
        cell.p25_intensity = float(partitioned[idx25])
        cell.p75_intensity = float(partitioned[idx75])

        thresh = cell.mean_intensity
        binary = (patch < thresh).astype(np.uint8)
        cell.approx_area = int(np.sum(binary))

        area = cell.approx_area
        if area > 0 and binary.shape[0] > 2 and binary.shape[1] > 2:
            eroded = (
                binary[1:-1, 1:-1] &
                binary[:-2, 1:-1] &
                binary[2:, 1:-1] &
                binary[1:-1, :-2] &
                binary[1:-1, 2:]
            )
            diff = np.clip(binary[1:-1, 1:-1].astype(np.int16) - eroded.astype(np.int16), 0, 1)
            perimeter = int(np.sum(diff))
            cell.circularity = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
        else:
            perimeter = int(np.sum(binary))
            cell.circularity = (4 * math.pi * area / perimeter ** 2) if perimeter > 0 else 0.0
