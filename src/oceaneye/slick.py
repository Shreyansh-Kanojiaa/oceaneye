"""Particles to a slick mask, and the shape statistics attribution compares.

The grid is *fixed per scenario*, not per cloud. Every mask in a scenario -- the observed
slick and every predicted one -- is rasterised onto the same `Grid`, so cell (i, j) means
the same patch of ocean in all of them. Deriving extent from each particle cloud's own
bounding box would make the IoU in `similarity.py` meaningless.

`Grid` lives here rather than in `config.py` because it is scenario geometry, not a run
parameter: origin and shape depend on where the tracks are, while `Config` is the same
frozen set of scalars for every scenario in a calibration run.

Positions are projected metres. Row index is y, column index is x.
"""

from dataclasses import dataclass

import numpy as np

from .config import Config
from .drift import Particles


@dataclass(frozen=True)
class Grid:
    """A fixed raster. `origin` is the lower-left corner of cell (0, 0), in metres."""

    origin: tuple[float, float]
    shape: tuple[int, int]      # (H, W) = (rows in y, columns in x)
    res_m: float

    @classmethod
    def covering(
        cls, xy: np.ndarray, cfg: Config, pad_m: float = 5_000.0
    ) -> "Grid":
        """One grid covering all of `xy` (..., 2) plus `pad_m` of drift room on each side.

        Call this **once per scenario**, over every track and the observed slick together,
        and pass the result to every `to_mask` call in that scenario.
        """
        pts = np.asarray(xy, dtype=float).reshape(-1, 2)
        if pts.size == 0:
            raise ValueError("cannot build a grid covering no points")
        lo = pts.min(axis=0) - pad_m
        hi = pts.max(axis=0) + pad_m
        w, h = np.ceil((hi - lo) / cfg.grid_res_m).astype(int) + 1
        return cls(origin=(float(lo[0]), float(lo[1])), shape=(int(h), int(w)),
                   res_m=float(cfg.grid_res_m))

    def cell_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cell indices (row, col) for positions `xy` (..., 2). May fall outside the grid."""
        xy = np.asarray(xy, dtype=float)
        col = np.floor((xy[..., 0] - self.origin[0]) / self.res_m).astype(int)
        row = np.floor((xy[..., 1] - self.origin[1]) / self.res_m).astype(int)
        return row, col

    def centres_of(self, row: np.ndarray, col: np.ndarray) -> np.ndarray:
        """Centre positions (metres, (..., 2)) of the cells at (row, col)."""
        x = self.origin[0] + (np.asarray(col) + 0.5) * self.res_m
        y = self.origin[1] + (np.asarray(row) + 0.5) * self.res_m
        return np.stack([x, y], axis=-1)


def to_mask(p: Particles, grid: Grid) -> np.ndarray:
    """Rasterise particle cloud `p` onto `grid`. Returns a bool (H, W) array.

    A cell is True if at least one particle is inside it. Particles that have drifted off
    the grid are dropped, not clamped to the edge -- clamping would pile them onto the
    boundary and invent a slick there.
    """
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    row, col = grid.cell_of(p.xy)
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    # ponytail: one particle marks a whole cell, no kernel or closing pass. At
    # cfg.n_particles=400 a large slick rasterises speckled; if 6.4 shows IoU suffering,
    # add a binary_closing here rather than raising the particle count.
    mask[row[inside], col[inside]] = True
    return mask


def morphology(mask: np.ndarray, grid: Grid) -> dict:
    """Shape statistics of a boolean slick `mask` on `grid`.

    Returns:
        area_m2:        float. Boundary-corrected: a filled cell with any background
            neighbour is on average half covered, so it counts half. Counting every
            filled cell whole inflates area by ~perimeter*res/2 -- 17% on a 3x1 km
            ellipse at 200 m resolution.
        centroid_xy:    (2,) metres.
        orientation_rad: float in **[0, pi)** -- the axis of the major principal moment.
            Orientation from second moments is only defined **modulo pi**: an axis at
            10 degrees and one at 190 degrees are the same orientation, and both are
            returned as 10 degrees. Comparisons must respect that (see `similarity.score`,
            which uses |cos(dtheta)|).
        elongation:     float >= 1, major/minor axis ratio (1.0 for a disc).
        perimeter_m:    float, total length of mask/background cell boundaries.

    An empty mask returns area 0.0 and NaN for every shape statistic; callers must treat
    that as "no predicted slick" rather than a degenerate shape.
    """
    mask = np.asarray(mask, dtype=bool)
    row, col = np.nonzero(mask)
    cell_area = grid.res_m**2
    if row.size == 0:
        return {"area_m2": 0.0, "centroid_xy": np.full(2, np.nan), "orientation_rad": np.nan,
                "elongation": np.nan, "perimeter_m": 0.0}

    pts = grid.centres_of(row, col)
    centroid = pts.mean(axis=0)

    # Second moments of the filled cells. Each cell also has variance res^2/12 about its
    # own centre; at res=200 m against slicks of kilometres that is well inside tolerance.
    cov = np.cov(pts - centroid, rowvar=False) if row.size > 1 else np.zeros((2, 2))
    vals, vecs = np.linalg.eigh(np.atleast_2d(cov))
    major = vecs[:, -1]
    orientation = float(np.arctan2(major[1], major[0]) % np.pi)
    lo, hi = float(max(vals[0], 0.0)), float(max(vals[-1], 0.0))
    elongation = float(np.sqrt(hi / lo)) if lo > 0 else np.inf

    # Boundary length: every edge where a filled cell meets background or the grid rim.
    # ponytail: staircase perimeter, so it runs ~4/pi high on smooth curves. Fine as a
    # relative shape cue; swap for a contour tracer if an absolute length is ever claimed.
    padded = np.pad(mask, 1)
    edges = (np.diff(padded, axis=0) != 0).sum() + (np.diff(padded, axis=1) != 0).sum()

    # A cell whose four neighbours are all filled is fully covered; one touching
    # background is half covered on average. Bounded in [N/2, N] cells, so it can never
    # go negative the way subtracting a perimeter term can.
    interior = (mask & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    n_boundary = row.size - int(interior.sum())

    return {
        "area_m2": float((row.size - 0.5 * n_boundary) * cell_area),
        "centroid_xy": centroid,
        "orientation_rad": orientation,
        "elongation": elongation,
        "perimeter_m": float(edges * grid.res_m),
    }
