import argparse
import csv
import math
import random
import sys
import zlib
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════

WINDOW_SIZE = 64

TYPE_STYLE: Dict[str, Dict] = {
    "cardiomyocyte": {
        "fill": (222, 128, 54), "fill_alpha": 180,
        "nucleus": (95, 35, 60), "nucleus_alpha": 255,
        "min_r": 7, "max_r": 14, "elong": (1.0, 1.0),
    },
    "inflammatory": {
        "fill": (145, 70, 160), "nucleus": (65, 20, 75),
        "fill_alpha": 180, "nucleus_alpha": 255,
        "min_r": 4, "max_r": 8, "elong": (1.0, 1.0),
    },
    "fibroblast": {
        "fill": (205, 170, 180), "nucleus": (95, 75, 90),
        "fill_alpha": 180, "nucleus_alpha": 255,
        "min_r": 5, "max_r": 10, "elong": (1.0, 1.0),
    },
    "endothelial": {
        "fill": (170, 185, 205), "nucleus": (75, 85, 105),
        "fill_alpha": 180, "nucleus_alpha": 255,
        "min_r": 3, "max_r": 6, "elong": (1.0, 1.0),
    },
}

# ══════════════════════════════════════════════════════════════════
# PARTE 1 — MODELO DE DADOS
# ══════════════════════════════════════════════════════════════════

class Cell:
    #Melhora cerca de ~50% consumo de memória, impede a criação de um dicionário interno para cada células
    __slots__ = ['id', 'x', 'y', 'cell_type',
        'mean_intensity', 'median_intensity', 'std_intensity',
        'min_intensity', 'max_intensity', 'p25_intensity',
        'p75_intensity', 'approx_area', 'circularity'
    ]

    def __init__(self, cell_id: str, x: int, y: int, cell_type: str) -> None:
        self.id = cell_id
        self.x = x
        self.y = y
        self.cell_type = cell_type.lower()

        # Métricas morfológicas (extract_metrics)
        self.mean_intensity:   Optional[float] = None
        self.median_intensity: Optional[float] = None
        self.std_intensity:    Optional[float] = None
        self.min_intensity:    Optional[float] = None
        self.max_intensity:    Optional[float] = None
        self.p25_intensity:    Optional[float] = None
        self.p75_intensity:    Optional[float] = None
        self.approx_area:      Optional[int]   = None
        self.circularity:      Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"Cell(id={self.id}, type={self.cell_type}, x={self.x}, y={self.y})"
        )


# ══════════════════════════════════════════════════════════════════
# PARTE 2 — LEITURA DO CSV
# ══════════════════════════════════════════════════════════════════

def iter_csv(
    csv_path: Path,
    required_columns: List[str],
    encoding: str = "utf-8",
) -> Generator[Dict[str, Any], None, None]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV não encontrado: {csv_path.resolve()}")
    with csv_path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV vazio: {csv_path}")
        col_map = {name.strip(): i for i, name in enumerate(header)}
        missing = set(required_columns) - set(col_map.keys())
        if missing:
            raise ValueError(f"Colunas em falta no CSV: {sorted(missing)}")
        max_idx = max(col_map[c] for c in required_columns)
        for row in reader:
            if not row or len(row) <= max_idx:
                continue
            try:
                row_data = {}
                for col in required_columns:
                    val = row[col_map[col]].strip()
                    if not val:
                        raise ValueError("Campo vazio")
                    row_data[col] = val
                yield row_data
            except (ValueError, IndexError):
                continue


def load_cells(csv_path: Path) -> List[Cell]:
    cells: List[Cell] = []
    required = ["cell_id", "x", "y", "cell_type"]
    VALID_TYPES = set(TYPE_STYLE.keys())

    for data in iter_csv(csv_path, required):
        ctype = data["cell_type"].lower().strip()
        if ctype in VALID_TYPES:
            cells.append(Cell(
            cell_id=data["cell_id"],
            x=int(float(data["x"])),
            y=int(float(data["y"])),
            cell_type=data["cell_type"],
         ))
        else:print(f"Aviso: Célula ignorada {data['cell_id']} por ser um tipo inválido: {ctype}")
    return cells


# ══════════════════════════════════════════════════════════════════
# PARTE 3 — GERAÇÃO DA IMAGEM SINTÉTICA
# ══════════════════════════════════════════════════════════════════

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
       # seed = hash((c.id, c.x, c.y, c.cell_type)) & 0xFFFFFFFF
        _draw_cell(draw, c.x, c.y, c.cell_type, seed)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    img.save(out_path, dpi=(600, 600))
    return img


# ══════════════════════════════════════════════════════════════════
# PARTE 4 — EXTRAÇÃO DE MÉTRICAS MORFOLÓGICAS  (CORRIGIDA + EXPANDIDA)
# ══════════════════════════════════════════════════════════════════

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

    #Limite Janela
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
        cell.p25_intensity    = float(np.percentile(flat, 25))
        cell.p75_intensity    = float(np.percentile(flat, 75))

        thresh = cell.mean_intensity
        binary = (patch < thresh).astype(np.uint8)
        cell.approx_area = int(np.sum(binary))

        area = cell.approx_area
        if area > 0 and binary.shape[0] > 2 and binary.shape[1] > 2:
        # Erosão 4-conectada correcta
            eroded = (
                binary[1:-1, 1:-1] &
                binary[:-2, 1:-1] &
                binary[2:, 1:-1] &
                binary[1:-1, :-2] &
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


# ══════════════════════════════════════════════════════════════════
# PARTE 5 — CONSTRUÇÃO DA REDE ESPACIAL
# ══════════════════════════════════════════════════════════════════

def estimate_threshold(cells: List[Cell], sample_size: int = 500) -> float:
    if len(cells) < 2:
        return 20.0
    rng = random.Random(42)
    sample = rng.sample(cells, min(sample_size, len(cells)))
    nearest_distances = []
    for cell in sample:
        best = float("inf")
        for other in cells:
            if cell.id == other.id:
                continue
            d = math.hypot(cell.x - other.x, cell.y - other.y)
            if d < best:
                best = d
        nearest_distances.append(best)
    mean_nn = np.mean(nearest_distances)
    threshold = mean_nn * 1.8
    threshold = max(12, threshold)
    threshold = min(35, threshold)
    return float(threshold)


class SpatialNetwork:
    def __init__(self, cells: List[Cell]) -> None:
        self.cells = cells
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {c.id: [] for c in cells}

    def build_naive(self, threshold: float) -> float:
        for c in self.cells:
            self.adjacency[c.id] = []
        t0 = time.perf_counter()
        n = len(self.cells)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = self.cells[i], self.cells[j]
                dist = math.hypot(a.x - b.x, a.y - b.y)
                if dist <= threshold:
                    self.adjacency[a.id].append((b.id, dist))
                    self.adjacency[b.id].append((a.id, dist))
        return time.perf_counter() - t0

    def build_optimized(self, threshold: float) -> float:
        for c in self.cells:
            self.adjacency[c.id] = []
        t0 = time.perf_counter()
        #construção da grelha
        grid: Dict[Tuple[int, int], List[Cell]] = defaultdict(list)
        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)
            grid[(gx, gy)].append(cell)
         #Evita duplicados (A-B/B-A)
        seen_edges = set()

        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)

            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby = grid.get((gx + dx, gy + dy),[])
                    for other in nearby:
                        #Ignorar própria celula
                        if cell.id == other.id:
                            continue
                        #sorted evita A-B e B-A, cria-se chave única
                        edge_key = tuple(sorted((cell.id, other.id)))

                        if edge_key not in seen_edges:
                            dist = math.hypot(cell.x - other.x, cell.y - other.y)

                            local_threshold = threshold
                            if cell.cell_type != other.cell_type:
                                local_threshold *= 0.75

                            #Adiciona ligação aos nós do grafo
                            if dist < local_threshold:
                                self.adjacency[cell.id].append((other.id, dist))
                                self.adjacency[other.id].append((cell.id, dist))
                                seen_edges.add(edge_key)

         #Converte as listas de vizinhos em tuplas para -memoria +desempenho
        for cid in self.adjacency:
            self.adjacency[cid] = tuple(self.adjacency[cid])

        return time.perf_counter() - t0

        temp_neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nearby = grid.get((gx + dx, gy + dy), [])
                    for other in nearby:
                        if cell.id == other.id:
                            continue
                        dist = math.hypot(cell.x - other.x, cell.y - other.y)
                        local_threshold = threshold
                        if cell.cell_type != other.cell_type:
                            local_threshold *= 0.75
                        if dist <= local_threshold:
                            temp_neighbors[cell.id].append((other.id, dist))
        MAX_NEIGHBORS = 12
        for cid, neighbors in temp_neighbors.items():
            neighbors.sort(key=lambda x: x[1])
            for nid, dist in neighbors[:MAX_NEIGHBORS]:
                self.adjacency[cid].append((nid, dist))
        return time.perf_counter() - t0

    def degree(self, cell_id: str) -> int:
        return len(self.adjacency[cell_id])

    def find_clusters(self) -> List[List[str]]:
        visited: set = set()
        clusters: List[List[str]] = []
        for start_id in self.adjacency:
            if start_id in visited:
                continue
            cluster: List[str] = []
            queue = deque([start_id])
            visited.add(start_id)
            while queue:
                cid = queue.popleft()
                cluster.append(cid)
                for neighbor_id, _ in self.adjacency[cid]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            clusters.append(cluster)
        return sorted(clusters, key=len, reverse=True)

    def find_hubs(self, top_n: int = 5) -> List[Tuple[str, int]]:
        degrees = [(cid, len(neighbors)) for cid, neighbors in self.adjacency.items()]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    def edge_count(self) -> int:
        return sum(len(v) for v in self.adjacency.values()) // 2

    # ── NOVO: todas as distâncias de arestas ───────────────────────
    def all_edge_distances(self) -> List[float]:
        """Devolve lista de distâncias de todas as arestas (sem duplicados)."""
        seen: set = set()
        dists: List[float] = []
        for cid, neighbors in self.adjacency.items():
            for nid, dist in neighbors:
                key = (min(cid, nid), max(cid, nid))
                if key not in seen:
                    seen.add(key)
                    dists.append(dist)
        return dists


# ══════════════════════════════════════════════════════════════════
# PARTE 6 — ANÁLISES ADICIONAIS (NOVAS)
# ══════════════════════════════════════════════════════════════════

# ── 6a. Densidade celular ──────────────────────────────────────────
def compute_density(cells: List[Cell], image_size: Tuple[int, int]) -> Dict[str, float]:
    """
    Densidade = número de células / área total da imagem (em milhares de px²).
    image_size = (width, height) em píxeis.
    """
    W, H = image_size
    area_px2 = W * H
    density = len(cells) / (area_px2 / 1_000_000)   # células por Mpx²
    return {
        "n_cells":       len(cells),
        "image_width":   W,
        "image_height":  H,
        "area_Mpx2":     area_px2 / 1_000_000,
        "density_per_Mpx2": density,
    }


# ── 6b. Estatísticas completas do grau da rede ────────────────────
def compute_degree_stats(cells: List[Cell], network: "SpatialNetwork") -> Dict[str, float]:
    degrees = [network.degree(c.id) for c in cells]
    arr = np.array(degrees, dtype=np.float64)
    return {
        "mean":   float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std":    float(np.std(arr)),
        "min":    float(np.min(arr)),
        "max":    float(np.max(arr)),
        "p25":    float(np.percentile(arr, 25)),
        "p75":    float(np.percentile(arr, 75)),
    }


# ── 6c. Hubs pelo critério média + 2×std ─────────────────────────
def find_hubs_statistical(
    cells: List[Cell],
    network: "SpatialNetwork",
    deg_stats: Dict[str, float],
) -> List[Tuple[str, int]]:
    """Retorna células com grau > média + 2×std."""
    threshold = deg_stats["mean"] + 2 * deg_stats["std"]
    hubs = [
        (c.id, network.degree(c.id))
        for c in cells
        if network.degree(c.id) > threshold
    ]
    return sorted(hubs, key=lambda x: x[1], reverse=True)


# ── 6d. Estatísticas completas de clusters ────────────────────────
def compute_cluster_stats(clusters: List[List[str]]) -> Dict[str, float]:
    sizes = [len(cl) for cl in clusters]
    arr = np.array(sizes, dtype=np.float64)
    size_dist: Dict[int, int] = defaultdict(int)
    for s in sizes:
        size_dist[s] += 1
    return {
        "n_clusters":     len(clusters),
        "mean_size":      float(np.mean(arr)),
        "median_size":    float(np.median(arr)),
        "std_size":       float(np.std(arr)),
        "max_size":       int(np.max(arr)),
        "min_size":       int(np.min(arr)),
        "n_isolated":     int(size_dist.get(1, 0)),
        "size_dist":      dict(sorted(size_dist.items())),
    }


# ── 6e. Distância entre células ligadas ───────────────────────────
def compute_edge_distance_stats(network: "SpatialNetwork") -> Dict[str, float]:
    dists = network.all_edge_distances()
    if not dists:
        return {"n_edges": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0, "p25": 0.0, "p75": 0.0}
    arr = np.array(dists, dtype=np.float64)
    return {
        "n_edges": len(dists),
        "mean":    float(np.mean(arr)),
        "median":  float(np.median(arr)),
        "std":     float(np.std(arr)),
        "min":     float(np.min(arr)),
        "max":     float(np.max(arr)),
        "p25":     float(np.percentile(arr, 25)),
        "p75":     float(np.percentile(arr, 75)),
    }


# ── 6f. Comparação por tipo celular ──────────────────────────────
def compute_type_comparison(
    cells: List[Cell],
    network: "SpatialNetwork",
) -> Dict[str, Dict[str, float]]:
    """
    Para cada tipo celular calcula:
      mean/std de intensity, area, circularity, grau da rede.
    """
    buckets: Dict[str, List[Cell]] = defaultdict(list)
    for c in cells:
        buckets[c.cell_type].append(c)

    result: Dict[str, Dict[str, float]] = {}
    for ctype, group in sorted(buckets.items()):
        intensities  = [c.mean_intensity  for c in group if c.mean_intensity  is not None]
        areas        = [c.approx_area     for c in group if c.approx_area     is not None]
        circularities= [c.circularity     for c in group if c.circularity     is not None]
        degrees      = [network.degree(c.id) for c in group]

        def _stats(lst: list) -> Tuple[float, float]:
            a = np.array(lst, dtype=np.float64)
            return (float(np.mean(a)) if len(a) else 0.0,
                    float(np.std(a))  if len(a) else 0.0)

        i_m, i_s   = _stats(intensities)
        a_m, a_s   = _stats(areas)
        c_m, c_s   = _stats(circularities)
        d_m, d_s   = _stats(degrees)

        result[ctype] = {
            "n":                len(group),
            "intensity_mean":   i_m, "intensity_std":  i_s,
            "area_mean":        a_m, "area_std":        a_s,
            "circularity_mean": c_m, "circularity_std": c_s,
            "degree_mean":      d_m, "degree_std":      d_s,
        }
    return result


# ══════════════════════════════════════════════════════════════════
# PARTE 7 — RELATÓRIO NO TERMINAL  (EXPANDIDO)
# ══════════════════════════════════════════════════════════════════

def print_report(
    cells: List[Cell],
    network: SpatialNetwork,
    clusters: List[List[str]],
    hubs_topn: List[Tuple[str, int]],
    hubs_stat: List[Tuple[str, int]],
    threshold: float,
    time_naive: Optional[float],
    time_optimized: float,
    image_path: Path,
    image_size: Tuple[int, int],
    deg_stats: Dict[str, float],
    cluster_stats: Dict[str, float],
    edge_dist_stats: Dict[str, float],
    type_comparison: Dict[str, Dict[str, float]],
    density_stats: Dict[str, float],
) -> None:
    cell_map = {c.id: c for c in cells}
    type_counts: Dict[str, int] = defaultdict(int)
    for c in cells:
        type_counts[c.cell_type] += 1

    line = "═" * 66
    thin = "─" * 66

    print(f"\n{line}")
    print("  RELATÓRIO DE ANÁLISE HISTOLÓGICA DE MIOCÁRDIO")
    print(f"{line}")
    print(f"  Imagem analisada : {image_path}")
    print(f"  Threshold (px)   : {threshold:.2f}")
    print(f"  Total de células : {len(cells)}")

    # ── 1. Métricas morfológicas por célula ───────────────────────
    print(f"\n{thin}")
    print("  1. MÉTRICAS MORFOLÓGICAS POR CÉLULA")
    print(thin)
    valid = [c for c in cells if c.mean_intensity is not None]
    if valid:
        intensities = [c.mean_intensity for c in valid]
        medians     = [c.median_intensity for c in valid]
        stds        = [c.std_intensity for c in valid]
        mins        = [c.min_intensity for c in valid]
        maxs        = [c.max_intensity for c in valid]
        p25s        = [c.p25_intensity for c in valid]
        p75s        = [c.p75_intensity for c in valid]
        print(f"  Métrica          {'Média':>8} {'Mediana':>8} {'Std':>8} {'Mín':>8} {'Máx':>8} {'P25':>8} {'P75':>8}")
        print(f"  {thin[:64]}")
        print(f"  Intensidade    "
              f"  {np.mean(intensities):7.2f}"
              f"  {np.mean(medians):7.2f}"
              f"  {np.mean(stds):7.2f}"
              f"  {np.mean(mins):7.2f}"
              f"  {np.mean(maxs):7.2f}"
              f"  {np.mean(p25s):7.2f}"
              f"  {np.mean(p75s):7.2f}")

    # ── 2. Área aproximada ────────────────────────────────────────
    print(f"\n{thin}")
    print("  2. ÁREA APROXIMADA")
    print(thin)
    areas = [c.approx_area for c in valid if c.approx_area is not None]
    if areas:
        arr = np.array(areas, dtype=np.float64)
        print(f"  Média    : {np.mean(arr):.1f} px²")
        print(f"  Mediana  : {np.median(arr):.1f} px²")
        print(f"  Std      : {np.std(arr):.1f} px²")
        print(f"  Mín/Máx  : {np.min(arr):.0f} / {np.max(arr):.0f} px²")
        print(f"  P25/P75  : {np.percentile(arr,25):.1f} / {np.percentile(arr,75):.1f} px²")

    # ── 3. Circularidade ──────────────────────────────────────────
    print(f"\n{thin}")
    print("  3. CIRCULARIDADE")
    print(thin)
    circs = [c.circularity for c in valid if c.circularity is not None]
    if circs:
        arr = np.array(circs, dtype=np.float64)
        print(f"  Média    : {np.mean(arr):.4f}  (1.0 = círculo perfeito)")
        print(f"  Mediana  : {np.median(arr):.4f}")
        print(f"  Std      : {np.std(arr):.4f}")
        print(f"  Mín/Máx  : {np.min(arr):.4f} / {np.max(arr):.4f}")

    # ── 4. Contagem por tipo celular ──────────────────────────────
    print(f"\n{thin}")
    print("  4. CONTAGEM POR TIPO CELULAR")
    print(thin)
    for ctype, count in sorted(type_counts.items()):
        pct = count / len(cells) * 100
        #bar = "█" * int(pct / 2)
        print(f"  {ctype:<18} {count:>5}  ({pct:5.1f}%)  ")

    # ── 5. Densidade celular ──────────────────────────────────────
    print(f"\n{thin}")
    print("  5. DENSIDADE CELULAR")
    print(thin)
    print(f"  Imagem             : {density_stats['image_width']:.0f} × {density_stats['image_height']:.0f} px")
    print(f"  Área total         : {density_stats['area_Mpx2']:.2f} Mpx²")
    print(f"  Densidade          : {density_stats['density_per_Mpx2']:.1f} células / Mpx²")

    # ── 6. Grau da rede ────────────────────────────────────────────
    print(f"\n{thin}")
    print("  6. GRAU DA REDE")
    print(thin)
    print(f"  Nós              : {len(cells)}")
    print(f"  Arestas          : {network.edge_count()}")
    print(f"  Média            : {deg_stats['mean']:.2f}")
    print(f"  Mediana          : {deg_stats['median']:.2f}")
    print(f"  Std              : {deg_stats['std']:.2f}")
    print(f"  Mín / Máx        : {deg_stats['min']:.0f} / {deg_stats['max']:.0f}")
    print(f"  P25 / P75        : {deg_stats['p25']:.1f} / {deg_stats['p75']:.1f}")

    # ── 7. Hubs ────────────────────────────────────────────────────
    print(f"\n{thin}")
    print("  7. HUBS  (grau > média + 2×std)")
    print(thin)
    hub_thresh = deg_stats["mean"] + 2 * deg_stats["std"]
    print(f"  Threshold de hub : {hub_thresh:.2f}")
    print(f"  Hubs encontrados : {len(hubs_stat)}")
    for cid, deg in hubs_stat[:10]:
        c = cell_map[cid]
        print(f"  id={cid:<8} tipo={c.cell_type:<16} grau={deg}")
    if not hubs_stat:
        print("  (nenhuma célula excede o critério)")

    # ── 8. Clusters ────────────────────────────────────────────────
    print(f"\n{thin}")
    print("  8. CLUSTERS (componentes conexas)")
    print(thin)
    cs = cluster_stats
    print(f"  Total            : {cs['n_clusters']}")
    print(f"  Tamanho médio    : {cs['mean_size']:.1f}  (mediana: {cs['median_size']:.1f})")
    print(f"  Std tamanho      : {cs['std_size']:.1f}")
    print(f"  Maior cluster    : {cs['max_size']} células")
    print(f"  Menor cluster    : {cs['min_size']} células")
    print(f"  Isoladas (size=1): {cs['n_isolated']}")
    # distribuição condensada
    dist_items = sorted(cs["size_dist"].items())
    buckets_disp = [(s, n) for s, n in dist_items if s <= 10]
    large = sum(n for s, n in dist_items if s > 10)
    disp = "  Distribuição     : " + "  ".join(f"size={s}:{n}" for s, n in buckets_disp)
    if large:
        disp += f"  size>10:{large}"
    print(disp)
    # Top 3 clusters com composição por tipo
    for i, cl in enumerate(clusters[:3], 1):
        comp: Dict[str, int] = defaultdict(int)
        for cid in cl:
            comp[cell_map[cid].cell_type] += 1
        comp_str = ", ".join(f"{k}:{v}" for k, v in sorted(comp.items()))
        print(f"  Cluster #{i}: {len(cl)} células  [{comp_str}]")

    # ── 9. Distância entre células ligadas ────────────────────────
    print(f"\n{thin}")
    print("  9. DISTÂNCIA ENTRE CÉLULAS LIGADAS")
    print(thin)
    ed = edge_dist_stats
    print(f"  Arestas          : {ed['n_edges']}")
    print(f"  Média            : {ed['mean']:.2f} px")
    print(f"  Mediana          : {ed['median']:.2f} px")
    print(f"  Std              : {ed['std']:.2f} px")
    print(f"  Mín / Máx        : {ed['min']:.2f} / {ed['max']:.2f} px")
    print(f"  P25 / P75        : {ed['p25']:.2f} / {ed['p75']:.2f} px")

    # ── 10. Comparação por tipo celular ───────────────────────────
    print(f"\n{thin}")
    print("  10. COMPARAÇÃO POR TIPO CELULAR")
    print(thin)
    hdr = f"  {'Tipo':<18} {'N':>5}  {'Int.média':>9}  {'Área média':>10}  {'Circ. média':>11}  {'Grau médio':>10}"
    print(hdr)
    print(f"  {thin[:64]}")
    for ctype, m in type_comparison.items():
        print(
            f"  {ctype:<18} {m['n']:>5}"
            f"  {m['intensity_mean']:>9.2f}"
            f"  {m['area_mean']:>10.1f}"
            f"  {m['circularity_mean']:>11.4f}"
            f"  {m['degree_mean']:>10.2f}"
        )

    # ── Performance ───────────────────────────────────────────────
    print(f"\n{thin}")
    print("  DESEMPENHO — NAÏVE vs OTIMIZADO")
    print(thin)
    if time_naive is not None:
        speedup = time_naive / time_optimized if time_optimized > 0 else float("inf")
        print(f"  Naïve O(n²)      : {time_naive:.4f} s")
        print(f"  Otimizado (grid) : {time_optimized:.4f} s")
        print(f"  Speedup          : {speedup:.1f}×")
    else:
        print(f"  Otimizado (grid) : {time_optimized:.4f} s")
        print("  (use --compare para incluir versão naïve)")

    # ── Interpretação histopatológica ─────────────────────────────
    print(f"\n{thin}")
    print("  INTERPRETAÇÃO HISTOPATOLÓGICA")
    print(thin)
    infl_pct  = type_counts.get("inflammatory", 0) / len(cells) * 100
    cardio_pct = type_counts.get("cardiomyocyte", 0) / len(cells) * 100
    fibro_pct  = type_counts.get("fibroblast", 0) / len(cells) * 100
    if infl_pct > 20:
        print(f"  ⚠  Inflamatórias: {infl_pct:.1f}% — possível miocardite ativa.")
    elif infl_pct > 10:
        print(f"  ℹ  Inflamatórias: {infl_pct:.1f}% — infiltrado moderado. Monitorizar.")
    else:
        print(f"  ✓  Inflamatórias: {infl_pct:.1f}% — sem evidência de miocardite ativa.")
    if fibro_pct > cardio_pct:
        print("  ⚠  Fibroblastos superam cardiomiócitos — padrão sugestivo de fibrose.")
    print(f"{line}\n")


# ══════════════════════════════════════════════════════════════════
# PARTE 8 — CLI E PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de análise histológica de miocárdio."
    )
    parser.add_argument("--csv",      required=True, type=Path)
    parser.add_argument("--image",    required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--top-hubs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/5] A carregar células de: {args.csv}")
    cells = load_cells(args.csv)
    if not cells:
        print("ERRO: Nenhuma célula carregada.", file=sys.stderr)
        sys.exit(1)
    print(f"      {len(cells)} células carregadas.")

    print(f"[2/5] A gerar imagem sintética → {args.image}")
    img = generate_image(cells, args.image)
    print(f"      Imagem guardada ({img.size[0]}×{img.size[1]} px).")

    print(f"[3/5] A extrair métricas morfológicas (janela={WINDOW_SIZE}px)...")
    t_metrics = time.perf_counter()
    extract_metrics(cells, img, window=WINDOW_SIZE)
    print(f"      Concluído em {time.perf_counter() - t_metrics:.3f}s.")

    network = SpatialNetwork(cells)
    time_naive: Optional[float] = None
    if args.compare:
        net_naive = SpatialNetwork(cells)
        time_naive = net_naive.build_naive(args.threshold)
        print(f"      Naïve: {time_naive:.4f}s")

    auto_threshold = estimate_threshold(cells)
    threshold = min(args.threshold, auto_threshold)
    print(f"[4/5] A construir rede (otimizada, threshold={threshold:.2f}px)...")
    print(f"      Threshold automático estimado: {auto_threshold:.2f}px")
    time_opt = network.build_optimized(threshold)
    print(f"      Otimizado: {time_opt:.4f}s")

    print("[5/5] A calcular análises...")
    clusters      = network.find_clusters()
    hubs_topn     = network.find_hubs(top_n=args.top_hubs)
    deg_stats     = compute_degree_stats(cells, network)
    hubs_stat     = find_hubs_statistical(cells, network, deg_stats)
    cluster_stats = compute_cluster_stats(clusters)
    edge_dist_stats = compute_edge_distance_stats(network)
    type_comparison = compute_type_comparison(cells, network)
    density_stats   = compute_density(cells, img.size)

    print_report(
        cells=cells,
        network=network,
        clusters=clusters,
        hubs_topn=hubs_topn,
        hubs_stat=hubs_stat,
        threshold=threshold,
        time_naive=time_naive,
        time_optimized=time_opt,
        image_path=args.image,
        image_size=img.size,
        deg_stats=deg_stats,
        cluster_stats=cluster_stats,
        edge_dist_stats=edge_dist_stats,
        type_comparison=type_comparison,
        density_stats=density_stats,
    )


if __name__ == "__main__":
    main()
