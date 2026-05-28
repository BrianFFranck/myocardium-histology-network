import csv
from pathlib import Path
from typing import Any, Dict, Generator, List

from .constants import TYPE_STYLE
from .models import Cell


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
    valid_types = set(TYPE_STYLE.keys())

    for data in iter_csv(csv_path, required):
        ctype = data["cell_type"].lower().strip()
        if ctype in valid_types:
            cells.append(Cell(
                cell_id=data["cell_id"],
                x=int(float(data["x"])),
                y=int(float(data["y"])),
                cell_type=data["cell_type"],
            ))
        else:
            print(f"Aviso: Célula ignorada {data['cell_id']} por ser um tipo inválido: {ctype}")
    return cells
