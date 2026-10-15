"""
Helpers for testing ODMR sign unwrapping on simulated NV magnetic-field maps.

This helper is designed to be used together with odmr_sign_unwrap.py.
It contains only simulation, Fourier reconstruction, metrics, and plotting utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift
from scipy.signal import convolve2d


@dataclass
class SimulationConfig:
    NX: int = 110
    NY: int = 110
    Lx: float = 3e-6
    Ly: float = 3e-6
    z_meas: float = 100e-9
    Ms: float = 8e5
    t_film: float = 10e-9
    theta_nv_deg: float = 125.25
    phi_nv_deg: float = 90.0

    @property
    def dx(self) -> float:
        return self.Lx / self.NX

    @property
    def z_meas_idx(self) -> int:
        return max(1, int(round(self.z_meas / self.dx)))

    @property
    def z_val(self) -> float:
        return self.z_meas_idx * self.dx

    @property
    def C_dip(self) -> float:
        # Dipole prefactor [T m^3] for a z-polarised film pixel.
        return 1e-7 * self.Ms * self.dx**2 * self.t_film

    @property
    def extent_um(self) -> list[float]:
        return [0, self.Lx * 1e6, 0, self.Ly * 1e6]

    @property
    def nv_hat(self) -> np.ndarray:
        theta = np.deg2rad(self.theta_nv_deg)
        phi = np.deg2rad(self.phi_nv_deg)
        return np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ], dtype=float)


def generate_spin_lattice(nx: int, ny: int, n_domains: int = 12, r_min: int = 6,
                          r_max: int = 20, seed: int = 4) -> np.ndarray:
    """Random magnetic domains (+/-1) with smoothed boundaries."""
    rng = np.random.default_rng(seed)
    spins = rng.choice([-1.0, 1.0], size=(ny, nx), p=[0.48, 0.52]) * 0.1
    yy, xx = np.ogrid[:ny, :nx]

    for _ in range(n_domains):
        cx, cy = rng.integers(0, nx), rng.integers(0, ny)
        r = rng.integers(r_min, r_max)
        val = rng.choice([-1.0, 1.0])
        spins[(xx - cx) ** 2 + (yy - cy) ** 2 <= r**2] = val

    kernel = np.array([
        [0.5, 1.0, 0.5],
        [1.0, 2.0, 1.0],
        [0.5, 1.0, 0.5],
    ]) / 9.0
    kernel /= kernel.sum()
    spins = np.sign(convolve2d(spins, kernel, mode="same", boundary="wrap") + 0.005)
    return spins


def dipole_kernels(nx: int, ny: int, dx: float, z: float, C: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real-space convolution kernels Kx, Ky, Kz for a z-polarised dipole film."""
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dx
    X, Y = np.meshgrid(x, y, indexing="xy")
    R2 = X**2 + Y**2 + z**2
    R = np.sqrt(R2)

    with np.errstate(divide="ignore", invalid="ignore"):
        Kx = C * 3 * X * z / R2**2.5
        Ky = C * 3 * Y * z / R2**2.5
        Kz = C * (3 * z**2 / R2**2.5 - 1 / R2**1.5)

    Kx[R == 0] = 0.0
    Ky[R == 0] = 0.0
    Kz[R == 0] = 0.0
    return Kx, Ky, Kz


def field_at_z(spins: np.ndarray, z: float, dx: float, C: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bx, By, Bz at height z via FFT convolution."""
    ny, nx = spins.shape
    Kx, Ky, Kz = dipole_kernels(nx, ny, dx, z, C)
    S = fft2(spins)

    def conv(K: np.ndarray) -> np.ndarray:
        return np.real(ifft2(S * fft2(ifftshift(K))))

    return conv(Kx), conv(Ky), conv(Kz)


def nv_projection(Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray, nv_hat: np.ndarray) -> np.ndarray:
    """Projection of the vector field on the NV axis."""
    return Bx * nv_hat[0] + By * nv_hat[1] + Bz * nv_hat[2]


def simulate_classic_lattice(cfg: SimulationConfig, spin_seed: int = 4) -> dict[str, Any]:
    """Run the single classic lattice simulation used for sign-unwrapping tests."""
    spins = generate_spin_lattice(cfg.NX, cfg.NY, seed=spin_seed)
    Bx, By, Bz = field_at_z(spins, cfg.z_val, dx=cfg.dx, C=cfg.C_dip)
    Bpar = nv_projection(Bx, By, Bz, cfg.nv_hat)
    return {
        "spins": spins,
        "Bx": Bx,
        "By": By,
        "Bz": Bz,
        "Bpar": Bpar,
        "Bpar_gt_mT": Bpar * 1e3,
        "A_sim_mT": np.abs(Bpar) * 1e3,
    }


def print_simulation_summary(cfg: SimulationConfig, sim: dict[str, Any]) -> None:
    Bx, By, Bz = sim["Bx"], sim["By"], sim["Bz"]
    Bmax_mT = np.sqrt(Bx**2 + By**2 + Bz**2).max() * 1e3
    ux, uy, uz = cfg.nv_hat
    print(f"dx       = {cfg.dx * 1e9:.2f} nm")
    print(f"z_meas   = {cfg.z_val * 1e9:.1f} nm  (index {cfg.z_meas_idx})")
    print(f"NV axis  = [{ux:.4f}, {uy:.4f}, {uz:.4f}]")
    print(f"|B|_max  = {Bmax_mT:.3f} mT")
    print(f"Bpar min/max = {np.nanmin(sim['Bpar_gt_mT']):+.3f} / {np.nanmax(sim['Bpar_gt_mT']):+.3f} mT")


def align_global_sign(B_rec: np.ndarray, B_gt: np.ndarray) -> tuple[np.ndarray, int]:
    """Correct the remaining global sign ambiguity using ground truth."""
    mask = np.isfinite(B_rec) & np.isfinite(B_gt)
    if not np.any(mask):
        return B_rec, +1
    corr = np.nansum(B_rec[mask] * B_gt[mask])
    s = +1 if corr >= 0 else -1
    return s * B_rec, s


def reconstruct_vector_field_from_nv_projection(Bnv_T: np.ndarray, dx: float, nv_hat: np.ndarray,
                                                 regularization: float | None = None,
                                                 subtract_mean: bool = True,
                                                 known_dc: tuple[float, float, float] | None = None,
                                                 d_meas: float = 100e-9,
                                                 lambda0: float = 1e-8,
                                                 p: float = 10,
                                                 alpha: float = 1.0,
                                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct Bx, By, Bz from one signed NV projection in a source-free half-space.

    The k=0/DC component is not determined by a single projection. For simulation
    benchmarking, pass known_dc=(mean(Bx_gt), mean(By_gt), mean(Bz_gt)). For real
    data, leave known_dc=None and compare AC components.
    """
    from helpers import reconstruct_B_from_Bparallel_Fourier

    Bnv_work = np.asarray(Bnv_T, dtype=float).copy()
    if subtract_mean:
        Bnv_work -= np.nanmean(Bnv_work)
    if regularization is not None:
        # Compatibility with the former constant penalty, only when requested.
        lambda0, p = float(regularization)**2 / 2, 0
    Bx_rec, By_rec, Bz_rec = reconstruct_B_from_Bparallel_Fourier(
        Bnv_work, nv_hat, dx, d_meas, lambda0=lambda0, p=p,
        alpha=alpha, dc_mode="zero")

    if known_dc is not None:
        Bx0, By0, Bz0 = known_dc
        Bx_rec = Bx_rec + Bx0
        By_rec = By_rec + By0
        Bz_rec = Bz_rec + Bz0

    return Bx_rec, By_rec, Bz_rec


def field_rms_errors(B_rec: tuple[np.ndarray, np.ndarray, np.ndarray],
                     B_gt: tuple[np.ndarray, np.ndarray, np.ndarray],
                     subtract_mean: bool = False) -> dict[str, float]:
    """RMS errors for Bx, By, Bz and vector magnitude, returned in mT."""
    out: dict[str, float] = {}
    labels = ["Bx", "By", "Bz"]
    err_stack = []

    for label, R, G in zip(labels, B_rec, B_gt):
        R2 = np.asarray(R, dtype=float)
        G2 = np.asarray(G, dtype=float)
        if subtract_mean:
            R2 = R2 - np.nanmean(R2)
            G2 = G2 - np.nanmean(G2)
        err = R2 - G2
        err_stack.append(err)
        out[label] = float(np.sqrt(np.nanmean(err**2)) * 1e3)

    err_stack = np.stack(err_stack, axis=0)
    out["B_vector"] = float(np.sqrt(np.nanmean(np.sum(err_stack**2, axis=0))) * 1e3)
    return out


def finite_fov_diagnostic(cfg: SimulationConfig, border_px: int = 15) -> pd.DataFrame:
    """Compare cropped-window inversion with/without zero padding on one larger map.

    The larger forward domain remains periodic. Cropping breaks the shared forward/
    inverse boundary assumption. AC reference and errors use the crop-wide means;
    padding is only a boundary convention and supplies no measured information.
    """
    if not 0 < border_px < min(cfg.NX, cfg.NY) / 2:
        raise ValueError("border_px must leave a nonempty interior.")
    spins = generate_spin_lattice(2*cfg.NX, 2*cfg.NY, seed=19, n_domains=24)
    large = field_at_z(spins, cfg.z_val, cfg.dx, cfg.C_dip)
    crop = (slice(cfg.NY//2, cfg.NY//2+cfg.NY), slice(cfg.NX//2, cfg.NX//2+cfg.NX))
    truth = tuple(b[crop] for b in large)
    truth_ac = tuple(b - np.mean(b) for b in truth)
    projection = nv_projection(*truth, cfg.nv_hat)
    projection -= np.mean(projection)
    interior = np.zeros_like(projection, dtype=bool)
    interior[border_px:-border_px, border_px:-border_px] = True
    rows = []
    for padding in (False, True):
        pad = ((cfg.NY//2, cfg.NY//2), (cfg.NX//2, cfg.NX//2)) if padding else ((0,0),(0,0))
        data = np.pad(projection, pad, mode='constant')
        rec = reconstruct_vector_field_from_nv_projection(
            data, cfg.dx, cfg.nv_hat, d_meas=cfg.z_val, subtract_mean=False)
        if padding:
            rec = tuple(b[pad[0][0]:pad[0][0]+cfg.NY, pad[1][0]:pad[1][0]+cfg.NX] for b in rec)
        rec = tuple(b - np.mean(b) for b in rec)
        error2 = sum((r-g)**2 for r,g in zip(rec, truth_ac))
        for region, mask in [('all', np.ones_like(interior)), ('interior', interior), ('border', ~interior)]:
            rows.append(dict(zero_padding=padding, region=region, border_px=border_px,
                             vector_RMS_mT=float(np.sqrt(np.mean(error2[mask]))*1e3),
                             max_vector_error_mT=float(np.sqrt(np.max(error2[mask]))*1e3)))
    return pd.DataFrame(rows)


def make_error_dataframe(Bpar_rec_mT: np.ndarray, Bpar_gt_mT: np.ndarray,
                         B_rec: tuple[np.ndarray, np.ndarray, np.ndarray],
                         B_gt: tuple[np.ndarray, np.ndarray, np.ndarray]) -> pd.DataFrame:
    """Create a compact error table."""
    bpar_rms = float(np.sqrt(np.nanmean((Bpar_rec_mT - Bpar_gt_mT) ** 2)))
    err_full = field_rms_errors(B_rec, B_gt, subtract_mean=False)
    err_ac = field_rms_errors(B_rec, B_gt, subtract_mean=True)

    bpar_ac_error = ((Bpar_rec_mT - np.nanmean(Bpar_rec_mT))
                    - (Bpar_gt_mT - np.nanmean(Bpar_gt_mT)))
    rows = [{"quantity": "B_parallel", "full RMS [mT]": bpar_rms,
             "AC RMS [mT]": float(np.sqrt(np.nanmean(bpar_ac_error**2)))}]
    for comp in ["Bx", "By", "Bz", "B_vector"]:
        rows.append({"quantity": comp, "full RMS [mT]": err_full[comp], "AC RMS [mT]": err_ac[comp]})
    pairs = [(Bpar_rec_mT, Bpar_gt_mT)] + [(r*1e3, g*1e3) for r, g in zip(B_rec, B_gt)]
    for row, (rec, gt) in zip(rows, pairs):
        err = np.asarray(rec) - np.asarray(gt)
        peak = float(np.nanmax(np.abs(gt)))
        row["max |GT| [mT]"] = peak
        row["max |error| [mT]"] = float(np.nanmax(np.abs(err)))
        row["max error / max GT"] = row["max |error| [mT]"] / peak if peak > 0 else np.nan
        row["max |AC GT| [mT]"] = float(np.nanmax(np.abs(gt-np.nanmean(gt))))
        row["max |AC error| [mT]"] = float(np.nanmax(np.abs(err-np.nanmean(err))))
    vector_gt = np.sqrt(sum(np.asarray(g)**2 for g in B_gt)) * 1e3
    vector_error = np.sqrt(sum((np.asarray(r)-g)**2 for r,g in zip(B_rec, B_gt))) * 1e3
    rows[-1].update({"max |GT| [mT]": float(np.nanmax(vector_gt)),
                     "max |error| [mT]": float(np.nanmax(vector_error)),
                     "max error / max GT": float(np.nanmax(vector_error)/np.nanmax(vector_gt))})
    vector_gt_ac = np.sqrt(sum((g-np.nanmean(g))**2 for g in B_gt))*1e3
    vector_error_ac = np.sqrt(sum(((r-np.nanmean(r))-(g-np.nanmean(g)))**2
                                  for r,g in zip(B_rec,B_gt)))*1e3
    rows[-1].update({"max |AC GT| [mT]": float(np.nanmax(vector_gt_ac)),
                     "max |AC error| [mT]": float(np.nanmax(vector_error_ac))})
    return pd.DataFrame(rows)


# -------------------------- plotting helpers -------------------------------

def plot_spin_lattice(spins: np.ndarray, cfg: SimulationConfig, title: str = "Spin lattice"):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(spins, origin="lower", extent=cfg.extent_um, cmap="viridis",
                   vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="spin")
    plt.tight_layout()
    return fig, ax


def plot_bpar_abs(Bpar_gt_mT: np.ndarray, A_sim_mT: np.ndarray, cfg: SimulationConfig):
    vmax = max(float(np.nanmax(np.abs(Bpar_gt_mT))), 1e-9)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.5), constrained_layout=True)
    im0 = axes[0].imshow(Bpar_gt_mT, origin="lower", extent=cfg.extent_um,
                         cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest")
    axes[0].set_title(r"Ground truth $B_{\parallel}$")
    fig.colorbar(im0, ax=axes[0], label="mT", shrink=0.85)

    im1 = axes[1].imshow(A_sim_mT, origin="lower", extent=cfg.extent_um,
                         cmap="plasma", vmin=0, vmax=vmax, interpolation="nearest")
    axes[1].set_title(r"Simulated $|B_{\parallel}|$")
    fig.colorbar(im1, ax=axes[1], label="mT", shrink=0.85)

    for ax in axes:
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
    return fig, axes


def plot_projection_comparison(
    Bpar_gt_mT,
    B_gc_mT,
    B_ref_mT,
    cfg=None,
    extent_um=None,
    title="Projection NV : GT, graph cut, refined, erreur",
    percentile=100,
    figsize=(10.5, 8.5),
):
    """
    Plot 2x2 :
      [0,0] GT B_parallel
      [0,1] Graph cut B_parallel
      [1,0] Refined B_parallel
      [1,1] Refined - GT

    Paramètres
    ----------
    Bpar_gt_mT : array 2D
        Ground truth B_parallel en mT.
    B_gc_mT : array 2D
        Reconstruction graph cut en mT.
    B_ref_mT : array 2D
        Reconstruction refined en mT.
    cfg : dict-like ou objet, optionnel
        Si cfg contient Lx, Ly ou extent_um, utilisé pour l'extent.
    extent_um : list, optionnel
        Extent matplotlib [xmin, xmax, ymin, ymax] en µm.
    title : str
        Titre global.
    percentile : float
        Percentile robuste pour les échelles couleur.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    Bpar_gt_mT = np.asarray(Bpar_gt_mT, dtype=float)
    B_gc_mT = np.asarray(B_gc_mT, dtype=float)
    B_ref_mT = np.asarray(B_ref_mT, dtype=float)

    if extent_um is None:
        if cfg is not None:
            if isinstance(cfg, dict) and "extent_um" in cfg:
                extent_um = cfg["extent_um"]
            elif hasattr(cfg, "extent_um"):
                extent_um = cfg.extent_um
            elif isinstance(cfg, dict) and ("Lx" in cfg) and ("Ly" in cfg):
                extent_um = [0, cfg["Lx"] * 1e6, 0, cfg["Ly"] * 1e6]
            elif hasattr(cfg, "Lx") and hasattr(cfg, "Ly"):
                extent_um = [0, cfg.Lx * 1e6, 0, cfg.Ly * 1e6]

    err_ref_mT = B_ref_mT - Bpar_gt_mT

    vmax_B = np.nanpercentile(
        np.abs(np.concatenate([
            Bpar_gt_mT.ravel(),
            B_gc_mT.ravel(),
            B_ref_mT.ravel(),
        ])),
        percentile,
    )
    if not np.isfinite(vmax_B) or vmax_B == 0:
        vmax_B = 1.0

    vmax_err = np.nanpercentile(np.abs(err_ref_mT), percentile)
    if not np.isfinite(vmax_err) or vmax_err == 0:
        vmax_err = 1.0

    fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)

    panels = [
        (axes[0, 0], Bpar_gt_mT, r"GT $B_\parallel$", vmax_B, "mT"),
        (axes[0, 1], B_gc_mT, r"Graph cut $B_\parallel$", vmax_B, "mT"),
        (axes[1, 0], B_ref_mT, r"Refined $B_\parallel$", vmax_B, "mT"),
        (axes[1, 1], err_ref_mT, r"Refined $-$ GT", vmax_err, "mT"),
    ]

    for ax, data, panel_title, vmax, cbar_label in panels:
        im = ax.imshow(
            data,
            origin="lower",
            extent=extent_um,
            cmap="bwr",
            vmin=-vmax,
            vmax=+vmax,
            interpolation="nearest",
        )
        ax.set_title(panel_title)
        ax.set_xlabel("x [µm]" if extent_um is not None else "j")
        ax.set_ylabel("y [µm]" if extent_um is not None else "i")
        fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)

    fig.suptitle(title)
    return fig, axes


def plot_field_bwr(ax: plt.Axes, Bx: np.ndarray, By: np.ndarray, Bz: np.ndarray,
                   cfg: SimulationConfig, title: str, fig: plt.Figure | None = None,
                   vmax_override: float | None = None, qstep: int = 6,
                   quiver_ref: float | None = None):
    """Plot Bz as colour map + normalised Bx/By quiver arrows."""
    vmax = vmax_override if vmax_override is not None else max(np.nanpercentile(np.abs(Bz * 1e3), 99), 1e-9)
    im = ax.imshow(Bz * 1e3, origin="lower", extent=cfg.extent_um,
                   cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest")

    nx, ny = cfg.NX, cfg.NY
    xs_q = ((np.arange(nx) + .5) * cfg.Lx * 1e6 / nx)[::qstep]
    ys_q = ((np.arange(ny) + .5) * cfg.Ly * 1e6 / ny)[::qstep]
    Xs_q, Ys_q = np.meshgrid(xs_q, ys_q, indexing="xy")
    U = Bx[::qstep, ::qstep]
    V = By[::qstep, ::qstep]
    mag = np.sqrt(U**2 + V**2)
    ref = quiver_ref if quiver_ref is not None else (np.percentile(mag[mag > 0], 90) if (mag > 0).any() else 1.0)
    ax.quiver(Xs_q, Ys_q, U / ref, V / ref, color="k", alpha=0.95,
              scale=18, width=0.004, headwidth=3.5)

    ax.set_title(title)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    if fig is not None:
        fig.colorbar(im, ax=ax, label="Bz [mT]", shrink=0.85)
    return im


def plot_reconstruction_3D_comparison(B_gt: tuple[np.ndarray, np.ndarray, np.ndarray],
                                      B_rec: tuple[np.ndarray, np.ndarray, np.ndarray],
                                      cfg: SimulationConfig,
                                      title_rec: str = "Reconstructed",
                                      error_arrow_gain: float = 10.0):
    """Compare AC fields; error arrows are enlarged by the stated relative factor."""
    if not np.isfinite(error_arrow_gain) or error_arrow_gain <= 0:
        raise ValueError("error_arrow_gain must be finite and positive.")
    Bx_gt, By_gt, Bz_gt = B_gt
    Bx_rec, By_rec, Bz_rec = B_rec
    dBx, dBy, dBz = Bx_rec - Bx_gt, By_rec - By_gt, Bz_rec - Bz_gt

    vmax_gt = max(float(np.nanpercentile(np.abs(Bz_gt * 1e3), 99)), 1e-9)
    vmax_err = max(float(np.nanpercentile(np.abs(dBz * 1e3), 99)), 1e-9)
    mags = np.concatenate([np.hypot(x[::6, ::6], y[::6, ::6]).ravel()
                           for x, y in [(Bx_gt, By_gt), (Bx_rec, By_rec)]])
    ref = max(float(np.nanpercentile(mags, 90)), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
    plot_field_bwr(axes[0], Bx_gt, By_gt, Bz_gt, cfg, "Ground truth", fig=fig, vmax_override=vmax_gt, quiver_ref=ref)
    plot_field_bwr(axes[1], Bx_rec, By_rec, Bz_rec, cfg, title_rec, fig=fig, vmax_override=vmax_gt, quiver_ref=ref)
    plot_field_bwr(axes[2], dBx, dBy, dBz, cfg,
                   f"Error rec - GT (arrows x{error_arrow_gain:g})",
                   fig=fig, vmax_override=vmax_err, quiver_ref=ref/error_arrow_gain)
    return fig, axes


def plot_component_errors(B_rec: tuple[np.ndarray, np.ndarray, np.ndarray],
                          B_gt: tuple[np.ndarray, np.ndarray, np.ndarray],
                          cfg: SimulationConfig,
                          title: str = "Component errors"):
    Bx_rec, By_rec, Bz_rec = B_rec
    Bx_gt, By_gt, Bz_gt = B_gt
    comps = [("Bx", Bx_rec - Bx_gt), ("By", By_rec - By_gt), ("Bz", Bz_rec - Bz_gt)]
    all_vals = np.concatenate([err.ravel() for _, err in comps]) * 1e3
    vmax = max(float(np.nanpercentile(np.abs(all_vals), 99)), 1e-9)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    for ax, (name, err) in zip(axes, comps):
        im = ax.imshow(err * 1e3, origin="lower", extent=cfg.extent_um,
                       cmap="bwr", vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{name} rec - GT")
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        fig.colorbar(im, ax=ax, label="[mT]", shrink=0.85)
    fig.suptitle(title)
    return fig, axes


def plot_seed_map_flipy(osu: Any, A_mT: np.ndarray, seed_mask: np.ndarray,
                        title: str = "Seeds — flipped y"):
    """Plot only: vertical flip of A and seeds, without modifying originals."""
    return osu.plot_seed_map(np.flipud(A_mT), np.flipud(seed_mask), title=title)
