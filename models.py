from typing import Optional

class Cell:
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
        return f"Cell(id={self.id}, type={self.cell_type}, x={self.x}, y={self.y})"
