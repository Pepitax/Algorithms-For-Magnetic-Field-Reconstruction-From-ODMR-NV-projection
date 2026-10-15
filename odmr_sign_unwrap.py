"""
odmr_sign_unwrap.py
===================

Utilities for semi-supervised sign reconstruction of ODMR/NV projected
magnetic-field maps measured as |B_parallel|.

Main workflow
-------------
1. Build A_mT = |B_parallel| in mT for each scan.
2. Annotate + / - seed regions manually with SeedAnnotator or
   HysteresisSeedAnnotator.
3. Run sign_unwrap_graphcut(A_mT, seed_mask) to obtain B_graphcut_mT.
4. Optionally run refine_B_continuous(A_mT, B_graphcut_mT) to smooth
   interfaces while keeping |B| close to A.
5. Convert results back to Tesla for your existing reconstruction helpers.

Dependencies
------------
Required: numpy, scipy, matplotlib
Optional but required for graph cut: PyMaxflow
    pip install PyMaxflow
    conda install -c conda-forge pymaxflow

Conventions
-----------
A_mT and B_*_mT are in mT.
Your plotting/reconstruction helpers may expect Tesla, so use
results_to_B_list_T(...).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
import json
import warnings

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import minimize

try:  # optional dependency; imported lazily through _require_maxflow too
    import maxflow  # type: ignore
except Exception:  # pragma: no cover - handled explicitly at runtime
    maxflow = None


ArrayLike2D = np.ndarray
SignKind = Literal["graphcut", "refined", "init"]


@dataclass
class SignUnwrapResult:
    """Container for one scan's sign-unwrapping result."""

    index: int
    date: str | None = None
    title: str | None = None
    A_mT: np.ndarray | None = None
    seed_mask: np.ndarray | None = None
    B_graphcut_mT: np.ndarray | None = None
    sign_graphcut: np.ndarray | None = None
    graphcut_info: dict[str, Any] = field(default_factory=dict)
    B_refined_mT: np.ndarray | None = None
    refine_info: dict[str, Any] = field(default_factory=dict)

    @property
    def B_best_mT(self) -> np.ndarray:
        """Return refined map if available, otherwise graph-cut map."""
        if self.B_refined_mT is not None:
            return self.B_refined_mT
        if self.B_graphcut_mT is not None:
            return self.B_graphcut_mT
        raise ValueError("No B map available in this result.")


# -----------------------------------------------------------------------------
# Seed helpers
# -----------------------------------------------------------------------------


def make_seed_mask(A: np.ndarray) -> np.ndarray:
    """Create an empty seed mask with the same shape as A.

    Values are:
      0  = no seed
      +1 = positive sign seed
      -1 = negative sign seed
    """
    return np.zeros_like(np.asarray(A), dtype=np.int8)


def add_seed_disk(
    seed_mask: np.ndarray,
    i: int,
    j: int,
    sign: int = +1,
    radius: int = 3,
    overwrite: bool = True,
) -> np.ndarray:
    """Add a circular seed region to seed_mask in-place and return it."""
    if sign not in (+1, -1):
        raise ValueError("sign must be +1 or -1")

    ni, nj = seed_mask.shape
    ii, jj = np.ogrid[:ni, :nj]
    mask = (ii - int(i)) ** 2 + (jj - int(j)) ** 2 <= int(radius) ** 2

    if overwrite:
        seed_mask[mask] = np.int8(sign)
    else:
        seed_mask[mask & (seed_mask == 0)] = np.int8(sign)
    return seed_mask


def erase_seed_disk(seed_mask: np.ndarray, i: int, j: int, radius: int = 3) -> np.ndarray:
    """Erase seeds inside a circular region in-place and return seed_mask."""
    ni, nj = seed_mask.shape
    ii, jj = np.ogrid[:ni, :nj]
    mask = (ii - int(i)) ** 2 + (jj - int(j)) ** 2 <= int(radius) ** 2
    seed_mask[mask] = 0
    return seed_mask


def assert_seed_masks_compatible(A_maps_mT: Sequence[np.ndarray], seed_masks: Sequence[np.ndarray]) -> None:
    """Raise if a list of seed masks is missing or shape-incompatible."""
    if len(A_maps_mT) != len(seed_masks):
        raise ValueError(f"Expected {len(A_maps_mT)} seed masks, got {len(seed_masks)}.")
    for k, (A, seed) in enumerate(zip(A_maps_mT, seed_masks)):
        if np.shape(A) != np.shape(seed):
            raise ValueError(
                f"Shape mismatch at index {k}: A has shape {np.shape(A)}, "
                f"seed_mask has shape {np.shape(seed)}."
            )


def save_seed_masks(
    path: str | Path,
    seed_masks: Sequence[np.ndarray],
    *,
    dates: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save seed masks to a compressed NPZ file.

    Parameters
    ----------
    path:
        Output .npz path.
    seed_masks:
        List of 2D seed masks.
    dates, titles:
        Optional metadata for scan identification.
    extra:
        Optional JSON-serializable metadata.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    stack = np.asarray(seed_masks, dtype=np.int8)
    payload: dict[str, Any] = {"seed_masks": stack}

    if dates is not None:
        payload["dates"] = np.asarray(list(dates), dtype=object)
    if titles is not None:
        payload["titles"] = np.asarray(list(titles), dtype=object)
    if extra is not None:
        payload["extra_json"] = np.asarray(json.dumps(extra), dtype=object)

    np.savez_compressed(path, **payload)
    return path


def load_seed_masks(path: str | Path) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Load seed masks saved with save_seed_masks.

    Returns
    -------
    seed_masks, metadata
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        seed_masks = [m.astype(np.int8, copy=True) for m in data["seed_masks"]]
        metadata: dict[str, Any] = {}
        if "dates" in data:
            metadata["dates"] = [str(x) for x in data["dates"].tolist()]
        if "titles" in data:
            metadata["titles"] = [str(x) for x in data["titles"].tolist()]
        if "extra_json" in data:
            try:
                metadata["extra"] = json.loads(str(data["extra_json"].item()))
            except Exception:
                metadata["extra"] = str(data["extra_json"].item())
    return seed_masks, metadata


def make_empty_seed_masks(A_maps_mT: Sequence[np.ndarray]) -> list[np.ndarray]:
    """Create one empty seed mask per A map."""
    return [make_seed_mask(A) for A in A_maps_mT]


# -----------------------------------------------------------------------------
# Interactive annotation
# -----------------------------------------------------------------------------


class SeedAnnotator:
    """Interactive annotation of one seed mask.

    Matplotlib controls
    -------------------
    Left click   : add + seed
    Right click  : add - seed
    Middle click : erase
    + key        : set mode +
    - key        : set mode -
    e key        : erase mode
    r key        : increase radius
    f key        : decrease radius
    c key        : clear seeds

    After annotation:
        seed_mask = annot.seed_mask
    """

    def __init__(
        self,
        A: np.ndarray,
        seed_mask: np.ndarray | None = None,
        radius: int = 3,
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = "viridis",
        show_status: bool = False,
    ) -> None:
        import matplotlib.pyplot as plt

        self.A = np.asarray(A, dtype=float)
        self.seed_mask = make_seed_mask(self.A) if seed_mask is None else seed_mask.copy()
        self.radius = int(radius)
        self.mode = +1
        self.show_status = bool(show_status)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.im = self.ax.imshow(self.A, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        self.fig.colorbar(self.im, ax=self.ax, label="|B_parallel| (mT)")

        self.plus_scatter = None
        self.minus_scatter = None
        self.text = None
        if self.show_status:
            self.text = self.ax.text(
                0.02,
                0.98,
                "",
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
            )

        self.ax.set_title("Annotation des signes")
        self.ax.set_xlabel("j")
        self.ax.set_ylabel("i")

        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        self.redraw()

    def redraw(self) -> None:
        if self.plus_scatter is not None:
            self.plus_scatter.remove()
        if self.minus_scatter is not None:
            self.minus_scatter.remove()

        plus = np.where(self.seed_mask == +1)
        minus = np.where(self.seed_mask == -1)

        self.plus_scatter = self.ax.scatter(
            plus[1],
            plus[0],
            s=25,
            marker="o",
            c="r",
            edgecolors="white",
            linewidths=0.5,
        )
        self.minus_scatter = self.ax.scatter(
            minus[1],
            minus[0],
            s=25,
            marker="x",
            c="b",
            linewidths=1.5,
        )

        if self.text is not None:
            mode_str = {+1: "+", -1: "-", 0: "erase"}[self.mode]
            self.text.set_text(f"mode={mode_str}, radius={self.radius}")

        self.fig.canvas.draw_idle()

    def onclick(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if not (0 <= i < self.A.shape[0] and 0 <= j < self.A.shape[1]):
            return

        if event.button == 1:
            add_seed_disk(self.seed_mask, i, j, sign=+1, radius=self.radius)
        elif event.button == 3:
            add_seed_disk(self.seed_mask, i, j, sign=-1, radius=self.radius)
        elif event.button == 2:
            erase_seed_disk(self.seed_mask, i, j, radius=self.radius)
        else:
            if self.mode == 0:
                erase_seed_disk(self.seed_mask, i, j, radius=self.radius)
            else:
                add_seed_disk(self.seed_mask, i, j, sign=self.mode, radius=self.radius)
        self.redraw()

    def onkey(self, event: Any) -> None:
        if event.key == "+":
            self.mode = +1
        elif event.key == "-":
            self.mode = -1
        elif event.key == "e":
            self.mode = 0
        elif event.key == "r":
            self.radius += 1
        elif event.key == "f":
            self.radius = max(1, self.radius - 1)
        elif event.key == "c":
            self.seed_mask[:] = 0
        self.redraw()


class HysteresisSeedAnnotator:
    """Interactive annotation of seed masks for a list of hysteresis scans.

    Controls
    --------
    Left click   : + seed on current scan
    Right click  : - seed on current scan
    Middle click : erase on current scan
    n / right    : next scan
    p / left     : previous scan
    +            : mode +
    -            : mode -
    e            : erase mode
    r            : increase radius
    f            : decrease radius
    c            : clear seeds for current scan
    C            : clear seeds for all scans
    s            : save seed masks if autosave_path is provided
    q            : close figure

    After annotation:
        seed_masks = annot.seed_masks
    """

    def __init__(
        self,
        A_maps_mT: Sequence[np.ndarray],
        *,
        seed_masks: Sequence[np.ndarray] | None = None,
        dates: Sequence[str] | None = None,
        titles: Sequence[str] | None = None,
        radius: int = 4,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        autosave_path: str | Path | None = None,
        show_status: bool = True,
    ) -> None:
        import matplotlib.pyplot as plt

        if len(A_maps_mT) == 0:
            raise ValueError("A_maps_mT is empty.")

        self.A_maps_mT = [np.asarray(A, dtype=float) for A in A_maps_mT]
        shape0 = self.A_maps_mT[0].shape
        for k, A in enumerate(self.A_maps_mT):
            if A.shape != shape0:
                raise ValueError(
                    "All maps must have the same shape for HysteresisSeedAnnotator. "
                    f"Map 0 has {shape0}, map {k} has {A.shape}."
                )

        if seed_masks is None:
            self.seed_masks = make_empty_seed_masks(self.A_maps_mT)
        else:
            self.seed_masks = [np.asarray(s, dtype=np.int8).copy() for s in seed_masks]
            assert_seed_masks_compatible(self.A_maps_mT, self.seed_masks)

        self.dates = list(dates) if dates is not None else [None] * len(self.A_maps_mT)
        self.titles = list(titles) if titles is not None else [None] * len(self.A_maps_mT)
        self.radius = int(radius)
        self.mode = +1
        self.index = 0
        self.autosave_path = Path(autosave_path) if autosave_path is not None else None
        self.show_status = bool(show_status)

        self.fig, self.ax = plt.subplots(figsize=(7, 6))
        self.im = self.ax.imshow(self.A_maps_mT[0], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        self.cbar = self.fig.colorbar(self.im, ax=self.ax, label="|B_parallel| (mT)")
        self.plus_scatter = None
        self.minus_scatter = None
        self.text = None
        if self.show_status:
            self.text = self.ax.text(
                0.02,
                0.98,
                "",
                transform=self.ax.transAxes,
                va="top",
                ha="left",
                fontsize=8,
                color="white",
                bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2),
            )

        self.ax.set_xlabel("j")
        self.ax.set_ylabel("i")
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self.onkey)
        self.redraw()

    @property
    def n_maps(self) -> int:
        return len(self.A_maps_mT)

    @property
    def current_A(self) -> np.ndarray:
        return self.A_maps_mT[self.index]

    @property
    def current_seed_mask(self) -> np.ndarray:
        return self.seed_masks[self.index]

    def _title(self) -> str:
        base = self.titles[self.index]
        if base is None:
            date = self.dates[self.index]
            base = str(date) if date is not None else f"scan {self.index}"
        mode_str = {+1: "+", -1: "-", 0: "erase"}[self.mode]
        return f"[{self.index + 1}/{self.n_maps}] {base} | mode={mode_str}, r={self.radius}"

    def redraw(self) -> None:
        self.im.set_data(self.current_A)
        self.ax.set_title(self._title())

        # Per-scan autoscale if no fixed clim was provided at construction.
        if self.im.norm.vmin is None or self.im.norm.vmax is None:
            vals = self.current_A[np.isfinite(self.current_A)]
            if vals.size:
                self.im.set_clim(np.nanpercentile(vals, 1), np.nanpercentile(vals, 99))

        if self.plus_scatter is not None:
            self.plus_scatter.remove()
        if self.minus_scatter is not None:
            self.minus_scatter.remove()

        plus = np.where(self.current_seed_mask == +1)
        minus = np.where(self.current_seed_mask == -1)
        self.plus_scatter = self.ax.scatter(
            plus[1], plus[0], s=25, marker="o", c="r", edgecolors="white", linewidths=0.5
        )
        self.minus_scatter = self.ax.scatter(
            minus[1], minus[0], s=25, marker="x", c="b", linewidths=1.5
        )

        if self.text is not None:
            mode_str = {+1: "+", -1: "-", 0: "erase"}[self.mode]
            self.text.set_text(
                f"scan {self.index + 1}/{self.n_maps}\n"
                f"mode={mode_str}, radius={self.radius}\n"
                "left:+ right:- middle:erase\n"
                "n/p: next/prev, s: save"
            )

        self.fig.canvas.draw_idle()

    def set_index(self, index: int) -> None:
        self.index = int(np.clip(index, 0, self.n_maps - 1))
        self.redraw()

    def save(self) -> None:
        if self.autosave_path is None:
            print("autosave_path is None; nothing saved.")
            return
        save_seed_masks(
            self.autosave_path,
            self.seed_masks,
            dates=["" if d is None else str(d) for d in self.dates],
            titles=["" if t is None else str(t) for t in self.titles],
        )
        print(f"Seed masks saved to {self.autosave_path}")

    def onclick(self, event: Any) -> None:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        j = int(round(event.xdata))
        i = int(round(event.ydata))
        if not (0 <= i < self.current_A.shape[0] and 0 <= j < self.current_A.shape[1]):
            return

        if event.button == 1:
            add_seed_disk(self.current_seed_mask, i, j, sign=+1, radius=self.radius)
        elif event.button == 3:
            add_seed_disk(self.current_seed_mask, i, j, sign=-1, radius=self.radius)
        elif event.button == 2:
            erase_seed_disk(self.current_seed_mask, i, j, radius=self.radius)
        else:
            if self.mode == 0:
                erase_seed_disk(self.current_seed_mask, i, j, radius=self.radius)
            else:
                add_seed_disk(self.current_seed_mask, i, j, sign=self.mode, radius=self.radius)
        self.redraw()

    def onkey(self, event: Any) -> None:
        key = event.key
        if key in ("n", "right"):
            self.set_index(self.index + 1)
            return
        if key in ("p", "left"):
            self.set_index(self.index - 1)
            return
        if key == "+":
            self.mode = +1
        elif key == "-":
            self.mode = -1
        elif key == "e":
            self.mode = 0
        elif key == "r":
            self.radius += 1
        elif key == "f":
            self.radius = max(1, self.radius - 1)
        elif key == "c":
            self.current_seed_mask[:] = 0
        elif key == "C":
            for s in self.seed_masks:
                s[:] = 0
        elif key == "s":
            self.save()
        elif key == "q":
            import matplotlib.pyplot as plt

            plt.close(self.fig)
            return
        self.redraw()


# -----------------------------------------------------------------------------
# Graph cut sign reconstruction
# -----------------------------------------------------------------------------


def _require_maxflow() -> Any:
    if maxflow is None:
        raise ImportError(
            "PyMaxflow is required for sign_unwrap_graphcut. Install it with: "
            "pip install PyMaxflow or conda install -c conda-forge pymaxflow"
        )
    return maxflow


def sign_unwrap_graphcut(
    A: np.ndarray,
    seed_mask: np.ndarray,
    valid_mask: np.ndarray | None = None,
    lambda_pair: float = 1.0,
    connectivity: int = 4,
    normalize_percentile: float = 95,
    seed_cost: float = 1e6,
    confidence: np.ndarray | None = None,
    eps: float = 1e-12,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct signed B = s*A with a semi-supervised graph cut.

    The pairwise term penalizes sign flips proportionally to 4*A_p*A_q,
    so flips are cheap near A ~= 0 and expensive where |B| is large.

    Parameters
    ----------
    A:
        Positive |B_parallel| map in mT.
    seed_mask:
        0 = unlabeled, +1 = positive seed, -1 = negative seed.
    valid_mask:
        Optional boolean mask. Defaults to finite pixels of A.
    lambda_pair:
        Spatial regularization strength.
    connectivity:
        4 or 8-neighbour graph.
    normalize_percentile:
        Robust percentile used to normalize A internally.
    seed_cost:
        Hard seed penalty. Keep large.
    confidence:
        Optional 0..1 confidence map. Low confidence lowers edge costs.
    eps:
        Numerical floor.
    verbose:
        Print summary.

    Returns
    -------
    B_signed_mT, sign_map, info
    """
    mf = _require_maxflow()
    A = np.asarray(A, dtype=float)
    seeds = np.asarray(seed_mask, dtype=np.int8)

    if A.ndim != 2:
        raise ValueError("A must be 2D.")
    if seeds.shape != A.shape:
        raise ValueError("seed_mask must have the same shape as A.")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    if valid_mask is None:
        valid_mask = np.isfinite(A)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(A)

    if confidence is None:
        confidence = np.ones_like(A, dtype=float)
    else:
        confidence = np.asarray(confidence, dtype=float)
        if confidence.shape != A.shape:
            raise ValueError("confidence must have the same shape as A.")
        confidence = np.clip(confidence, 0.0, 1.0)
        confidence[~np.isfinite(confidence)] = 0.0

    vals = np.abs(A[valid_mask])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("No valid finite pixels in A.")

    scale = np.nanpercentile(vals, normalize_percentile)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(vals)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    A_norm = np.zeros_like(A, dtype=float)
    A_norm[valid_mask] = np.abs(A[valid_mask]) / (scale + eps)

    g = mf.Graph[float]()
    nodeids = g.add_grid_nodes(A.shape)
    n_edges = 0

    def add_edge_between(p: tuple[int, int], q: tuple[int, int], distance_factor: float = 1.0) -> None:
        nonlocal n_edges
        i1, j1 = p
        i2, j2 = q
        if not (valid_mask[i1, j1] and valid_mask[i2, j2]):
            return

        Ap = A_norm[i1, j1]
        Aq = A_norm[i2, j2]
        conf = float(np.sqrt(confidence[i1, j1] * confidence[i2, j2]))
        cap = lambda_pair * 4.0 * Ap * Aq * conf / distance_factor
        if np.isfinite(cap) and cap > 0:
            g.add_edge(int(nodeids[i1, j1]), int(nodeids[i2, j2]), float(cap), float(cap))
            n_edges += 1

    ni, nj = A.shape
    for i in range(ni):
        for j in range(nj):
            if not valid_mask[i, j]:
                continue
            if i + 1 < ni:
                add_edge_between((i, j), (i + 1, j), 1.0)
            if j + 1 < nj:
                add_edge_between((i, j), (i, j + 1), 1.0)
            if connectivity == 8:
                if i + 1 < ni and j + 1 < nj:
                    add_edge_between((i, j), (i + 1, j + 1), np.sqrt(2.0))
                if i + 1 < ni and j - 1 >= 0:
                    add_edge_between((i, j), (i + 1, j - 1), np.sqrt(2.0))

    plus_mask = (seeds == +1) & valid_mask
    minus_mask = (seeds == -1) & valid_mask

    # Convention internally corrected below using seed agreement.
    for i, j in zip(*np.where(plus_mask)):
        g.add_tedge(int(nodeids[i, j]), float(seed_cost), 0.0)
    for i, j in zip(*np.where(minus_mask)):
        g.add_tedge(int(nodeids[i, j]), 0.0, float(seed_cost))

    flow = g.maxflow()
    segments = g.get_grid_segments(nodeids)
    sign_candidate = np.where(segments, +1.0, -1.0)
    sign_candidate[~valid_mask] = np.nan

    # Correct global convention from seeds.
    seed_pixels = (seeds != 0) & valid_mask
    if np.any(seed_pixels):
        score = np.sum(sign_candidate[seed_pixels] == seeds[seed_pixels])
        score_flip = np.sum((-sign_candidate[seed_pixels]) == seeds[seed_pixels])
        if score_flip > score:
            sign_candidate *= -1.0
    else:
        warnings.warn(
            "No seeds were provided. The global sign and segmentation may be arbitrary.",
            RuntimeWarning,
            stacklevel=2,
        )

    sign_map = sign_candidate
    B_signed = sign_map * A

    info = {
        "flow": float(flow),
        "scale_mT": float(scale),
        "n_valid": int(np.sum(valid_mask)),
        "n_edges": int(n_edges),
        "n_plus_seeds": int(np.sum(plus_mask)),
        "n_minus_seeds": int(np.sum(minus_mask)),
        "lambda_pair": float(lambda_pair),
        "connectivity": int(connectivity),
        "seed_cost": float(seed_cost),
    }
    if verbose:
        print("Graph cut terminé.")
        print(f"  flow              = {flow:.6g}")
        print(f"  scale_mT          = {scale:.6g}")
        print(f"  pixels valides    = {info['n_valid']}")
        print(f"  arêtes            = {info['n_edges']}")
        print(f"  graines +         = {info['n_plus_seeds']}")
        print(f"  graines -         = {info['n_minus_seeds']}")
        print(f"  lambda_pair       = {lambda_pair}")
        print(f"  connectivité      = {connectivity}")

    return B_signed, sign_map, info


# -----------------------------------------------------------------------------
# Continuous refinement
# -----------------------------------------------------------------------------


def laplacian_nearest(X: np.ndarray) -> np.ndarray:
    """2D 4-neighbour Laplacian with nearest boundary mode."""
    kernel = np.array(
        [
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    return ndi.convolve(np.asarray(X, dtype=float), kernel, mode="nearest")


# Backward-compatible private alias used in earlier notebooks.
_laplacian_nearest = laplacian_nearest


def refine_B_continuous(
    A: np.ndarray,
    B_init: np.ndarray,
    seed_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    tau_mT: float = 0.15,
    mu_data: float = 1.0,
    lambda_lap: float = 0.03,
    lambda_anchor: float = 0.05,
    lambda_seed: float = 10.0,
    data_weight_floor: float = 0.05,
    anchor_power: float = 2.0,
    maxiter: int = 300,
    gtol: float = 1e-6,
    verbose: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Refine a signed B map by optimizing B directly.

    The normalized energy is roughly:
        0.5*mu_data * sum w*(B^2 - A^2)^2
      + 0.5*lambda_lap * sum (Delta B)^2
      + 0.5*lambda_anchor * sum q*(B - B_init)^2
      + 0.5*lambda_seed * sum_seeds (B - sign_seed*A)^2

    This is non-convex, so B_init should come from a good graph cut.
    """
    A = np.asarray(A, dtype=float)
    B_init = np.asarray(B_init, dtype=float)
    if A.shape != B_init.shape:
        raise ValueError("A and B_init must have the same shape.")

    if valid_mask is None:
        valid_mask = np.isfinite(A) & np.isfinite(B_init)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(A) & np.isfinite(B_init)

    if seed_mask is None:
        seed_mask = np.zeros_like(A, dtype=np.int8)
    else:
        seed_mask = np.asarray(seed_mask, dtype=np.int8)
        if seed_mask.shape != A.shape:
            raise ValueError("seed_mask must have the same shape as A.")

    vals = np.abs(A[valid_mask])
    if vals.size == 0:
        raise ValueError("No valid pixels.")

    scale = np.nanpercentile(vals, 95)
    if not np.isfinite(scale) or scale <= 0:
        scale = np.nanmax(vals)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    a = np.zeros_like(A, dtype=float)
    x0 = np.zeros_like(A, dtype=float)
    a[valid_mask] = np.abs(A[valid_mask]) / scale
    x0[valid_mask] = B_init[valid_mask] / scale

    tau = tau_mT / scale
    valid_float = valid_mask.astype(float)

    w_data = np.zeros_like(A, dtype=float)
    w_data[valid_mask] = data_weight_floor + (1.0 - data_weight_floor) * (
        a[valid_mask] ** 2 / (a[valid_mask] ** 2 + tau**2)
    )

    q_anchor = np.zeros_like(A, dtype=float)
    q_anchor[valid_mask] = (
        a[valid_mask] ** 2 / (a[valid_mask] ** 2 + tau**2)
    ) ** anchor_power

    seed_pixels = (seed_mask != 0) & valid_mask
    seed_target = np.zeros_like(A, dtype=float)
    seed_target[seed_pixels] = seed_mask[seed_pixels] * a[seed_pixels]
    seed_float = seed_pixels.astype(float)

    x0_flat = x0.ravel()

    def energy_and_grad(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
        x = x_flat.reshape(A.shape)

        r = x * x - a * a
        E_data = 0.5 * mu_data * np.sum(valid_float * w_data * r * r)
        G_data = mu_data * valid_float * w_data * 2.0 * x * r

        Lx = laplacian_nearest(x)
        E_lap = 0.5 * lambda_lap * np.sum(valid_float * Lx * Lx)
        G_lap = lambda_lap * laplacian_nearest(valid_float * Lx)

        diff_anchor = x - x0
        E_anchor = 0.5 * lambda_anchor * np.sum(valid_float * q_anchor * diff_anchor * diff_anchor)
        G_anchor = lambda_anchor * valid_float * q_anchor * diff_anchor

        diff_seed = x - seed_target
        E_seed = 0.5 * lambda_seed * np.sum(seed_float * diff_seed * diff_seed)
        G_seed = lambda_seed * seed_float * diff_seed

        invalid = 1.0 - valid_float
        E_invalid = 0.5 * 100.0 * np.sum(invalid * (x - x0) ** 2)
        G_invalid = 100.0 * invalid * (x - x0)

        E = E_data + E_lap + E_anchor + E_seed + E_invalid
        G = G_data + G_lap + G_anchor + G_seed + G_invalid
        return float(E), G.ravel().astype(float)

    res = minimize(
        fun=lambda z: energy_and_grad(z),
        x0=x0_flat,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": int(maxiter), "gtol": float(gtol), "ftol": 1e-10, "maxls": 50, "disp": False},
    )

    x_opt = res.x.reshape(A.shape)
    B_refined = x_opt * scale
    B_refined[~valid_mask] = np.nan

    info = {
        "success": bool(res.success),
        "message": str(res.message),
        "niter": int(res.nit),
        "final_energy": float(res.fun),
        "scale_mT": float(scale),
        "tau_norm": float(tau),
        "mu_data": float(mu_data),
        "lambda_lap": float(lambda_lap),
        "lambda_anchor": float(lambda_anchor),
        "lambda_seed": float(lambda_seed),
    }
    if verbose:
        print("Raffinement continu terminé.")
        print(f"  success       = {res.success}")
        print(f"  message       = {res.message}")
        print(f"  niter         = {res.nit}")
        print(f"  final energy  = {res.fun:.6g}")
        print(f"  scale_mT      = {scale:.6g}")
        print(f"  tau_mT        = {tau_mT}")

    return B_refined, info


# -----------------------------------------------------------------------------
# Optional sign-map cleaning
# -----------------------------------------------------------------------------


def clean_small_sign_islands(
    sign_map: np.ndarray,
    seed_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    min_size: int = 8,
    n_iter: int = 1,
) -> np.ndarray:
    """Remove small isolated sign components by flipping them.

    Components containing a same-sign seed are preserved.
    """
    sign_clean = np.array(sign_map, dtype=float, copy=True)
    if valid_mask is None:
        valid_mask = np.isfinite(sign_clean)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(sign_clean)

    if seed_mask is None:
        seed_mask = np.zeros_like(sign_clean, dtype=np.int8)
    else:
        seed_mask = np.asarray(seed_mask, dtype=np.int8)

    structure = ndi.generate_binary_structure(2, 2)
    for _ in range(int(n_iter)):
        for s in (+1, -1):
            mask = (sign_clean == s) & valid_mask
            labels, nlab = ndi.label(mask, structure=structure)
            for lab in range(1, nlab + 1):
                comp = labels == lab
                if int(np.sum(comp)) >= int(min_size):
                    continue
                if np.any((seed_mask == s) & comp):
                    continue
                sign_clean[comp] = -s

    sign_clean[~valid_mask] = np.nan
    return sign_clean


def rebuild_B_from_sign(A: np.ndarray, sign_map: np.ndarray) -> np.ndarray:
    """Return B = sign_map * A."""
    return np.asarray(sign_map, dtype=float) * np.asarray(A, dtype=float)


# -----------------------------------------------------------------------------
# Batch hysteresis utilities
# -----------------------------------------------------------------------------


def run_sign_unwrap_for_hysteresis(
    A_maps_mT: Sequence[np.ndarray],
    seed_masks: Sequence[np.ndarray],
    *,
    dates: Sequence[str] | None = None,
    titles: Sequence[str] | None = None,
    valid_masks: Sequence[np.ndarray] | None = None,
    graphcut_kwargs: dict[str, Any] | None = None,
    refine: bool = True,
    refine_kwargs: dict[str, Any] | None = None,
    clean_islands: bool = False,
    clean_kwargs: dict[str, Any] | None = None,
    verbose: bool = True,
) -> list[SignUnwrapResult]:
    """Run graph cut and optional continuous refinement for many scans."""
    A_maps = [np.asarray(A, dtype=float) for A in A_maps_mT]
    seeds = [np.asarray(s, dtype=np.int8) for s in seed_masks]
    assert_seed_masks_compatible(A_maps, seeds)

    n = len(A_maps)
    dates_list = list(dates) if dates is not None else [None] * n
    titles_list = list(titles) if titles is not None else [None] * n
    valid_list = list(valid_masks) if valid_masks is not None else [None] * n

    graphcut_kwargs = dict(graphcut_kwargs or {})
    refine_kwargs = dict(refine_kwargs or {})
    clean_kwargs = dict(clean_kwargs or {})

    results: list[SignUnwrapResult] = []
    for k, (A, seed, valid) in enumerate(zip(A_maps, seeds, valid_list)):
        if verbose:
            print("=" * 72)
            print(f"Scan {k + 1}/{n}: {titles_list[k] or dates_list[k] or k}")

        B_gc, sign_gc, info_gc = sign_unwrap_graphcut(
            A,
            seed,
            valid_mask=valid,
            verbose=verbose,
            **graphcut_kwargs,
        )

        if clean_islands:
            sign_gc = clean_small_sign_islands(sign_gc, seed_mask=seed, valid_mask=valid, **clean_kwargs)
            B_gc = rebuild_B_from_sign(A, sign_gc)

        B_ref = None
        info_ref: dict[str, Any] = {}
        if refine:
            B_ref, info_ref = refine_B_continuous(
                A,
                B_gc,
                seed_mask=seed,
                valid_mask=valid,
                verbose=verbose,
                **refine_kwargs,
            )

        results.append(
            SignUnwrapResult(
                index=k,
                date=dates_list[k],
                title=titles_list[k],
                A_mT=A,
                seed_mask=seed,
                B_graphcut_mT=B_gc,
                sign_graphcut=sign_gc,
                graphcut_info=info_gc,
                B_refined_mT=B_ref,
                refine_info=info_ref,
            )
        )
    return results


def results_to_B_list_mT(results: Sequence[SignUnwrapResult], kind: SignKind = "refined") -> list[np.ndarray]:
    """Extract graphcut/refined/init maps from results in mT."""
    out: list[np.ndarray] = []
    for r in results:
        if kind == "graphcut":
            if r.B_graphcut_mT is None:
                raise ValueError(f"Result {r.index} has no graphcut map.")
            out.append(r.B_graphcut_mT)
        elif kind == "refined":
            if r.B_refined_mT is None:
                raise ValueError(f"Result {r.index} has no refined map.")
            out.append(r.B_refined_mT)
        elif kind == "init":
            if r.A_mT is None:
                raise ValueError(f"Result {r.index} has no initial A map.")
            out.append(r.A_mT)
        else:
            raise ValueError("kind must be 'graphcut', 'refined', or 'init'.")
    return out


def results_to_B_list_T(results: Sequence[SignUnwrapResult], kind: SignKind = "refined") -> list[np.ndarray]:
    """Extract graphcut/refined/init maps from results and convert mT -> T."""
    return [B_mT * 1e-3 for B_mT in results_to_B_list_mT(results, kind=kind)]


def save_unwrap_results_npz(
    path: str | Path,
    results: Sequence[SignUnwrapResult],
    *,
    include_A: bool = True,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Save batch sign-unwrapping arrays and lightweight metadata to NPZ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "B_graphcut_mT": np.asarray([r.B_graphcut_mT for r in results], dtype=float),
        "sign_graphcut": np.asarray([r.sign_graphcut for r in results], dtype=float),
        "seed_masks": np.asarray([r.seed_mask for r in results], dtype=np.int8),
        "dates": np.asarray(["" if r.date is None else r.date for r in results], dtype=object),
        "titles": np.asarray(["" if r.title is None else r.title for r in results], dtype=object),
    }
    if include_A:
        payload["A_mT"] = np.asarray([r.A_mT for r in results], dtype=float)
    if any(r.B_refined_mT is not None for r in results):
        payload["B_refined_mT"] = np.asarray(
            [np.full_like(r.B_graphcut_mT, np.nan) if r.B_refined_mT is None else r.B_refined_mT for r in results],
            dtype=float,
        )
    if extra is not None:
        payload["extra_json"] = np.asarray(json.dumps(extra), dtype=object)

    np.savez_compressed(path, **payload)
    return path


# -----------------------------------------------------------------------------
# Lightweight plotting utilities (optional; you can use your helpers instead)
# -----------------------------------------------------------------------------


def plot_seed_map(A_mT: np.ndarray, seed_mask: np.ndarray | None = None, title: str = "Graines de signe"):
    """Quick plot of A_mT with fixed red/blue seed markers."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A_mT, origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax, label="|B_parallel| (mT)")
    if seed_mask is not None:
        plus = np.where(seed_mask == +1)
        minus = np.where(seed_mask == -1)
        ax.scatter(plus[1], plus[0], s=20, marker="o", c="r", edgecolors="white", linewidths=0.5)
        ax.scatter(minus[1], minus[0], s=20, marker="x", c="b", linewidths=1.2)
    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.tight_layout()
    return fig, ax


# Backward-compatible alias with user's previous naming.
plot_seeds = plot_seed_map


def plot_graphcut_result(
    A: np.ndarray,
    B_signed: np.ndarray,
    sign_map: np.ndarray | None = None,
    seed_mask: np.ndarray | None = None,
    title_prefix: str = "Graph cut sign unwrapping",
    symmetric_B: bool = True,
):
    """Plot A, signed B and sign map for one scan."""
    import matplotlib.pyplot as plt

    A = np.asarray(A, dtype=float)
    B_signed = np.asarray(B_signed, dtype=float)
    if sign_map is None:
        sign_map = np.sign(B_signed)

    if symmetric_B:
        vmaxB = np.nanpercentile(np.abs(B_signed), 99)
        if not np.isfinite(vmaxB) or vmaxB == 0:
            vmaxB = 1.0
        vminB = -vmaxB
    else:
        vminB = np.nanpercentile(B_signed, 1)
        vmaxB = np.nanpercentile(B_signed, 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    im0 = axes[0].imshow(A, origin="upper", cmap="viridis")
    axes[0].set_title(r"$A=|B_\parallel|$ (mT)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(B_signed, origin="upper", cmap="coolwarm", vmin=vminB, vmax=vmaxB)
    axes[1].set_title(r"$B_\parallel$ signé (mT)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(sign_map, origin="upper", cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("Signe reconstruit")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if seed_mask is not None:
        plus = np.where(seed_mask == +1)
        minus = np.where(seed_mask == -1)
        for ax in axes:
            ax.scatter(plus[1], plus[0], s=18, marker="o", c="r", edgecolors="white", linewidths=0.5)
            ax.scatter(minus[1], minus[0], s=18, marker="x", c="b", linewidths=1.2)

    for ax in axes:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
    fig.suptitle(title_prefix)
    plt.tight_layout()
    return fig, axes


def plot_zero_and_boundaries(
    A: np.ndarray,
    B_signed: np.ndarray,
    tau_zero_mT: float = 0.15,
    title: str = "Zéro et frontières de signe",
):
    """Plot zero band A < tau and reconstructed sign boundary."""
    import matplotlib.pyplot as plt

    A = np.asarray(A, dtype=float)
    B_signed = np.asarray(B_signed, dtype=float)
    zero_mask = A < tau_zero_mT
    sign_map = np.sign(B_signed)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, origin="upper", cmap="viridis")
    fig.colorbar(im, ax=ax, label="|B_parallel| (mT)")
    ax.contour(zero_mask.astype(float), levels=[0.5], linewidths=1.5)
    ax.contour(sign_map, levels=[0], linewidths=1.5)
    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.tight_layout()
    return fig, ax


def plot_refinement_comparison(
    A: np.ndarray,
    B_gc: np.ndarray,
    B_refined: np.ndarray,
    seed_mask: np.ndarray | None = None,
    title: str = "Raffinement continu",
):
    """Compare A, graph-cut B, refined B and difference for one scan."""
    import matplotlib.pyplot as plt

    vmax = np.nanpercentile(np.abs(np.concatenate([B_gc.ravel(), B_refined.ravel()])), 99)
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    im0 = axes[0].imshow(A, origin="upper", cmap="viridis")
    axes[0].set_title(r"$A=|B|$ (mT)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(B_gc, origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title(r"$B_\mathrm{gc}$ (mT)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(B_refined, origin="upper", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title(r"$B_\mathrm{refined}$ (mT)")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    diff = B_refined - B_gc
    vmax_diff = np.nanpercentile(np.abs(diff), 99)
    if not np.isfinite(vmax_diff) or vmax_diff == 0:
        vmax_diff = 1.0
    im3 = axes[3].imshow(diff, origin="upper", cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
    axes[3].set_title(r"$B_\mathrm{refined}-B_\mathrm{gc}$ (mT)")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    if seed_mask is not None:
        plus = np.where(seed_mask == +1)
        minus = np.where(seed_mask == -1)
        for ax in axes:
            ax.scatter(plus[1], plus[0], s=18, marker="o", c="r", edgecolors="white", linewidths=0.5)
            ax.scatter(minus[1], minus[0], s=18, marker="x", c="b", linewidths=1.2)

    for ax in axes:
        ax.set_xlabel("j")
        ax.set_ylabel("i")
    fig.suptitle(title)
    plt.tight_layout()
    return fig, axes


__all__ = [
    "SignUnwrapResult",
    "make_seed_mask",
    "make_empty_seed_masks",
    "add_seed_disk",
    "erase_seed_disk",
    "assert_seed_masks_compatible",
    "save_seed_masks",
    "load_seed_masks",
    "SeedAnnotator",
    "HysteresisSeedAnnotator",
    "sign_unwrap_graphcut",
    "laplacian_nearest",
    "_laplacian_nearest",
    "refine_B_continuous",
    "clean_small_sign_islands",
    "rebuild_B_from_sign",
    "run_sign_unwrap_for_hysteresis",
    "results_to_B_list_mT",
    "results_to_B_list_T",
    "save_unwrap_results_npz",
    "plot_seed_map",
    "plot_seeds",
    "plot_graphcut_result",
    "plot_zero_and_boundaries",
    "plot_refinement_comparison",
]
