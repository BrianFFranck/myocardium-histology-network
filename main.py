"""
main.py — Pipeline de Análise Histológica de Miocárdio
Programação para as Biociências II

Pipeline:
  1. Lê o CSV fornecido e instancia objetos Cell
  2. Gera uma imagem sintética PNG a partir dessas células
  3. Lê a imagem gerada e extrai métricas morfológicas por janela de 64px
  4. Constrói a rede espacial (versão naïve O(n²) e versão otimizada com grid)
  5. Deteta clusters (BFS) e hubs
  6. Gera relatório estruturado no terminal

Uso:
  python main.py --csv data/cells_small.csv --image output/slide.png --threshold 50
  python main.py --csv data/cells_large.csv --image output/slide_large.png --threshold 50 --compare
"""

import argparse
import csv
import math
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ══════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════

WINDOW_SIZE = 64   # tamanho da janela local para extração de métricas (px)

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
    """Representa uma célula individual com coordenadas, tipo e métricas."""

    def __init__(
        self,
        cell_id: str,
        x: int,
        y: int,
        cell_type: str,
    ) -> None:
        self.id = cell_id
        self.x = x
        self.y = y
        self.cell_type = cell_type.lower()

        # Métricas morfológicas — preenchidas em extract_metrics()
        self.mean_intensity: Optional[float] = None
        self.std_intensity: Optional[float] = None
        self.approx_area: Optional[int] = None
        self.circularity: Optional[float] = None

    def __repr__(self) -> str:
        return (
            f"Cell(id={self.id}, type={self.cell_type}, "
            f"x={self.x}, y={self.y}, "
            f"mean={self.mean_intensity:.2f}, std={self.std_intensity:.2f})"
            if self.mean_intensity is not None
            else f"Cell(id={self.id}, type={self.cell_type}, x={self.x}, y={self.y})"
        )


# ══════════════════════════════════════════════════════════════════
# PARTE 2 — LEITURA DO CSV (generator genérico)
# ══════════════════════════════════════════════════════════════════

def iter_csv(
    csv_path: Path,
    required_columns: List[str],
    encoding: str = "utf-8",
) -> Generator[Dict[str, Any], None, None]:
    """Lê qualquer CSV, valida colunas obrigatórias e faz yield linha a linha."""
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
    """Instancia lista de Cell a partir do CSV."""
    cells: List[Cell] = []
    required = ["cell_id", "x", "y", "cell_type"]
    for data in iter_csv(csv_path, required):
        cells.append(Cell(
            cell_id=data["cell_id"],
            x=int(float(data["x"])),
            y=int(float(data["y"])),
            cell_type=data["cell_type"],
        ))
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
    """Gera imagem sintética H&E a partir das células e guarda em out_path."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mx = max(c.x for c in cells)
    my = max(c.y for c in cells)
    W, H = max(2048, mx + 100), max(2048, my + 100)

    img = _make_background(W, H)
    draw = ImageDraw.Draw(img, "RGBA")

    for c in cells:
        seed = hash((c.id, c.x, c.y, c.cell_type)) & 0xFFFFFFFF
        _draw_cell(draw, c.x, c.y, c.cell_type, seed)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.35))
    img.save(out_path,dpi=(600,600))
    return img


# ══════════════════════════════════════════════════════════════════
# PARTE 4 — EXTRAÇÃO DE MÉTRICAS MORFOLÓGICAS
# ══════════════════════════════════════════════════════════════════

def extract_metrics(cells: List[Cell], img: Image.Image, window: int = WINDOW_SIZE) -> None:
    """
    Para cada célula, extrai uma janela de `window`×`window` px centrada
    nas suas coordenadas e calcula 4 métricas morfológicas:
      - mean_intensity   : intensidade média (escala de cinzentos)
      - std_intensity    : desvio-padrão de intensidade
      - approx_area      : número de píxeis abaixo do threshold de Otsu simplificado
      - circularity      : 4π·área/perímetro² da máscara binária
    Tudo calculado apenas com NumPy e Pillow.
    """
    # Converte para a escala de cinzentos uma única vez (otimização)
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
            cell.mean_intensity = 0.0
            cell.std_intensity = 0.0
            cell.approx_area = 0
            cell.circularity = 0.0
            continue

        cell.mean_intensity = float(np.mean(patch))
        cell.std_intensity = float(np.std(patch))

        # Threshold simples: Otsu manual com dois níveis
        thresh = float(np.mean(patch))
        binary = (patch < thresh).astype(np.uint8)

        cell.approx_area = int(np.sum(binary))

        # Circularidade: estimativa do perímetro por erosão
        area = cell.approx_area
        if area > 0:
            # perímetro ≈ diferença entre máscara original e erosão
            from numpy.lib.stride_tricks import sliding_window_view
            # erosão manual 3×3
            if binary.shape[0] > 2 and binary.shape[1] > 2:
                eroded = np.ones_like(binary)
                eroded[1:-1, 1:-1] = (
                    binary[1:-1, 1:-1] &
                    binary[:-2, 1:-1] & binary[2:, 1:-1] &
                    binary[1:-1, :-2] & binary[1:-1, 2:]
                )
                perimeter = int(np.sum(binary - eroded))
            else:
                perimeter = int(np.sum(binary))
            cell.circularity = (4 * math.pi * area / (perimeter ** 2)) if perimeter > 0 else 0.0
        else:
            cell.circularity = 0.0


# ══════════════════════════════════════════════════════════════════
# PARTE 5 — CONSTRUÇÃO DA REDE ESPACIAL
# ══════════════════════════════════════════════════════════════════
def estimate_threshold(
    cells: List[Cell],
    sample_size: int = 500,
) -> float:
    """
    Estima automaticamente um threshold espacial credível
    baseado na distância média ao vizinho mais próximo.
    """

    if len(cells) < 2:
        return 20.0

    rng = random.Random(42)

    sample = rng.sample(
        cells,
        min(sample_size, len(cells))
    )

    nearest_distances = []

    for cell in sample:

        best = float("inf")

        for other in cells:

            if cell.id == other.id:
                continue

            d = math.hypot(
                cell.x - other.x,
                cell.y - other.y
            )

            if d < best:
                best = d

        nearest_distances.append(best)

    mean_nn = np.mean(nearest_distances)

    # fator experimental estável
    threshold = mean_nn * 1.8

    # limites seguros (neighbors)
    threshold = max(12, threshold)
    threshold = min(35, threshold)

    return float(threshold)
class SpatialNetwork:
    """
    Rede espacial de células com dois modos de construção:
      - naive   : O(n²) — todos os pares
      - optimized: O(n·k) médio — grid espacial hash
    """

    def __init__(self, cells: List[Cell]) -> None:
        self.cells = cells
        # adjacency: cell_id → lista de (cell_id, distância)
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {
            c.id: [] for c in cells
        }

    def build_naive(self, threshold: float) -> float:
        """Versão ingénua O(n²)."""

        for c in self.cells:
            self.adjacency[c.id] = []

        t0 = time.perf_counter()

        n = len(self.cells)

        for i in range(n):

            for j in range(i + 1, n):

                a = self.cells[i]
                b = self.cells[j]

                dist = math.hypot(
                    a.x - b.x,
                    a.y - b.y
                )

                if dist <= threshold:
                    self.adjacency[a.id].append((b.id, dist))
                    self.adjacency[b.id].append((a.id, dist))

        return time.perf_counter() - t0

    def build_optimized(self, threshold: float) -> float:
        """
        Versão otimizada com grid hash.
        """

        for c in self.cells:
            self.adjacency[c.id] = []

        t0 = time.perf_counter()

        grid: Dict[Tuple[int, int], List[Cell]] = defaultdict(list)

        # ─────────────────────────────────────────────
        # Construção do grid espacial
        # ─────────────────────────────────────────────

        for cell in self.cells:
            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)

            grid[(gx, gy)].append(cell)

        temp_neighbors: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # ─────────────────────────────────────────────
        # Procura local
        # ─────────────────────────────────────────────

        for cell in self.cells:

            gx = int(cell.x / threshold)
            gy = int(cell.y / threshold)

            for dx in (-1, 0, 1):

                for dy in (-1, 0, 1):

                    nearby = grid.get((gx + dx, gy + dy), [])

                    for other in nearby:

                        if cell.id == other.id:
                            continue

                        dist = math.hypot(
                            cell.x - other.x,
                            cell.y - other.y
                        )

                        # Threshold reduzido para tipos diferentes
                        local_threshold = threshold

                        if cell.cell_type != other.cell_type:
                            local_threshold *= 0.75

                        if dist <= local_threshold:
                            temp_neighbors[cell.id].append(
                                (other.id, dist)
                            )

        # ─────────────────────────────────────────────
        # Limite máximo de vizinhos
        # ─────────────────────────────────────────────

        MAX_NEIGHBORS = 12

        for cid, neighbors in temp_neighbors.items():

            neighbors.sort(key=lambda x: x[1])

            limited = neighbors[:MAX_NEIGHBORS]

            for nid, dist in limited:
                self.adjacency[cid].append((nid, dist))

        return time.perf_counter() - t0

    # ── Análise estrutural ──────────────────────────────────────

    def degree(self, cell_id: str) -> int:
        return len(self.adjacency[cell_id])

    def find_clusters(self) -> List[List[str]]:
        """BFS para encontrar componentes conexas (clusters)."""
        visited: set = set()
        clusters: List[List[str]] = []
        adj = self.adjacency

        for start_id in adj:
            if start_id in visited:
                continue
            cluster: List[str] = []
            queue = deque([start_id])
            visited.add(start_id)
            while queue:
                cid = queue.popleft()
                cluster.append(cid)
                for neighbor_id, _ in adj[cid]:
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append(neighbor_id)
            clusters.append(cluster)

        return sorted(clusters, key=len, reverse=True)

    def find_hubs(self, top_n: int = 5) -> List[Tuple[str, int]]:
        """Devolve os top_n nós com maior grau (hubs)."""
        degrees = [(cid, len(neighbors)) for cid, neighbors in self.adjacency.items()]
        return sorted(degrees, key=lambda x: x[1], reverse=True)[:top_n]

    def edge_count(self) -> int:
        return sum(len(v) for v in self.adjacency.values()) // 2


# ══════════════════════════════════════════════════════════════════
# PARTE 6 — RELATÓRIO NO TERMINAL
# ══════════════════════════════════════════════════════════════════

def print_report(
    cells: List[Cell],
    network: SpatialNetwork,
    clusters: List[List[str]],
    hubs: List[Tuple[str, int]],
    threshold: float,
    time_naive: Optional[float],
    time_optimized: float,
    image_path: Path,
) -> None:
    cell_map = {c.id: c for c in cells}

    # Contagem por tipo
    type_counts: Dict[str, int] = defaultdict(int)
    for c in cells:
        type_counts[c.cell_type] += 1

    # Métricas médias por tipo
    type_metrics: Dict[str, List[float]] = defaultdict(list)
    for c in cells:
        if c.mean_intensity is not None:
            type_metrics[c.cell_type].append(c.mean_intensity)

    line = "═" * 62
    thin = "─" * 62

    print(f"\n{line}")
    print("  RELATÓRIO DE ANÁLISE HISTOLÓGICA DE MIOCÁRDIO")
    print(f"{line}")
    print(f"  Imagem analisada : {image_path}")
    print(f"  Threshold (px)   : {threshold}")
    print(f"  Total de células : {len(cells)}")

    print(f"\n{thin}")
    print("  DISTRIBUIÇÃO CELULAR")
    print(thin)
    for ctype, count in sorted(type_counts.items()):
        pct = count / len(cells) * 100
        bar = "█" * int(pct / 2)
        avg_i = (
            f"  intensidade média={sum(type_metrics[ctype])/len(type_metrics[ctype]):.1f}"
            if type_metrics[ctype] else ""
        )
        print(f"  {ctype:<16} {count:>5}  ({pct:5.1f}%)  {bar}{avg_i}")

    print(f"\n{thin}")
    print("  REDE ESPACIAL")
    print(thin)
    print(f"  Nós   : {len(cells)}")
    print(f"  Arestas: {network.edge_count()}")
    avg_deg = (
        sum(network.degree(c.id) for c in cells) / len(cells) if cells else 0
    )
    print(f"  Grau médio       : {avg_deg:.2f}")

    print(f"\n{thin}")
    print("  CLUSTERS (componentes conexas)")
    print(thin)
    print(f"  Total de clusters: {len(clusters)}")
    isolated = sum(1 for cl in clusters if len(cl) == 1)
    print(f"  Células isoladas : {isolated}")
    print(f"  Maior cluster    : {len(clusters[0])} células" if clusters else "  (sem clusters)")
    # Top 3 clusters com composição por tipo
    for i, cl in enumerate(clusters[:3], 1):
        comp: Dict[str, int] = defaultdict(int)
        for cid in cl:
            comp[cell_map[cid].cell_type] += 1
        comp_str = ", ".join(f"{k}:{v}" for k, v in sorted(comp.items()))
        print(f"  Cluster #{i}: {len(cl)} células  [{comp_str}]")

    print(f"\n{thin}")
    print("  HUBS (células com mais ligações)")
    print(thin)
    for cid, deg in hubs:
        c = cell_map[cid]
        print(f"  id={cid:<8} tipo={c.cell_type:<16} grau={deg}")

    print(f"\n{thin}")
    print("  MÉTRICAS MORFOLÓGICAS (médias globais)")
    print(thin)
    valid = [c for c in cells if c.mean_intensity is not None]
    if valid:
        print(f"  Intensidade média   : {sum(c.mean_intensity for c in valid)/len(valid):.2f}")
        print(f"  Desvio-padrão médio : {sum(c.std_intensity for c in valid)/len(valid):.2f}")
        print(f"  Área aprox. média   : {sum(c.approx_area for c in valid)/len(valid):.1f} px²")
        print(f"  Circularidade média : {sum(c.circularity for c in valid)/len(valid):.3f}")

    print(f"\n{thin}")
    print("  DESEMPENHO — COMPARAÇÃO NAÏVE vs OTIMIZADO")
    print(thin)
    if time_naive is not None:
        speedup = time_naive / time_optimized if time_optimized > 0 else float("inf")
        print(f"  Naïve O(n²)      : {time_naive:.4f} s")
        print(f"  Otimizado (grid) : {time_optimized:.4f} s")
        print(f"  Speedup          : {speedup:.1f}×")
    else:
        print(f"  Otimizado (grid) : {time_optimized:.4f} s")
        print("  (use --compare para incluir versão naïve)")

    print(f"\n{thin}")
    print("  INTERPRETAÇÃO HISTOPATOLÓGICA")
    print(thin)
    infl_pct = type_counts.get("inflammatory", 0) / len(cells) * 100
    if infl_pct > 20:
        print(f"  ⚠  Células inflamatórias: {infl_pct:.1f}% — possível miocardite ativa.")
        print("     Clusters densos de células inflamatórias são consistentes")
        print("     com infiltração linfocítica típica de miocardite.")
    elif infl_pct > 10:
        print(f"  ℹ  Células inflamatórias: {infl_pct:.1f}% — infiltrado moderado.")
        print("     Monitorizar evolução clínica.")
    else:
        print(f"  ✓  Células inflamatórias: {infl_pct:.1f}% — sem evidência de miocardite ativa.")

    cardio_pct = type_counts.get("cardiomyocyte", 0) / len(cells) * 100
    fibro_pct  = type_counts.get("fibroblast", 0) / len(cells) * 100
    if fibro_pct > cardio_pct:
        print("  ⚠  Fibroblastos superam cardiomiócitos — padrão sugestivo de fibrose.")
    print(f"{line}\n")


# ══════════════════════════════════════════════════════════════════
# PARTE 7 — CLI E PONTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de análise histológica de miocárdio."
    )
    parser.add_argument(
        "--csv", required=True, type=Path,
        help="Caminho para o ficheiro CSV de células (ex: data/cells_small.csv)"
    )
    parser.add_argument(
        "--image", required=True, type=Path,
        help="Caminho para guardar/carregar a imagem PNG (ex: output/slide.png)"
    )
    parser.add_argument(
        "--threshold", type=float, default=50.0,
        help="Limiar de distância (px) para ligar células na rede (default: 50)"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Executa também a versão naïve para comparação de desempenho"
    )
    parser.add_argument(
        "--top-hubs", type=int, default=5,
        help="Número de hubs a mostrar no relatório (default: 5)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── 1. Carregar CSV ───────────────────────────────────────────
    print(f"[1/5] A carregar células de: {args.csv}")
    cells = load_cells(args.csv)
    if not cells:
        print("ERRO: Nenhuma célula carregada. Verifica o CSV.", file=sys.stderr)
        sys.exit(1)
    print(f"      {len(cells)} células carregadas.")

    # ── 2. Gerar imagem sintética ─────────────────────────────────
    print(f"[2/5] A gerar imagem sintética → {args.image}")
    img = generate_image(cells, args.image)
    print(f"      Imagem guardada ({img.size[0]}×{img.size[1]} px).")

    # ── 3. Extrair métricas morfológicas ──────────────────────────
    print(f"[3/5] A extrair métricas morfológicas (janela={WINDOW_SIZE}px)...")
    t_metrics = time.perf_counter()
    extract_metrics(cells, img, window=WINDOW_SIZE)
    print(f"      Concluído em {time.perf_counter() - t_metrics:.3f}s.")

    # ── 4. Construir rede espacial ────────────────────────────────
    network = SpatialNetwork(cells)
    time_naive: Optional[float] = None

    if args.compare:
        print(f"[4/5] A construir rede (naïve O(n²), threshold={args.threshold})...")
        # Usamos uma cópia da rede para não contaminar a otimizada
        net_naive = SpatialNetwork(cells)
        time_naive = net_naive.build_naive(args.threshold)
        print(f"      Naïve: {time_naive:.4f}s")

    # ─────────────────────────────────────────────
    # Threshold adaptativo
    # ─────────────────────────────────────────────

    auto_threshold = estimate_threshold(cells)

    threshold = min(args.threshold, auto_threshold)

    print(
        f"[4/5] A construir rede "
        f"(otimizada, threshold={threshold:.2f}px)..."
    )

    print(
        f"      Threshold automático estimado: "
        f"{auto_threshold:.2f}px"
    )

    time_opt = network.build_optimized(threshold)
    print(f"      Otimizado: {time_opt:.4f}s")

    # ── 5. Análise estrutural ─────────────────────────────────────
    print("[5/5] A analisar clusters e hubs...")
    clusters = network.find_clusters()
    hubs = network.find_hubs(top_n=args.top_hubs)

    # ── 6. Relatório ──────────────────────────────────────────────
    print_report(
        cells=cells,
        network=network,
        clusters=clusters,
        hubs=hubs,
        threshold=threshold,
        time_naive=time_naive,
        time_optimized=time_opt,
        image_path=args.image,
    )


if __name__ == "__main__":
    main()
