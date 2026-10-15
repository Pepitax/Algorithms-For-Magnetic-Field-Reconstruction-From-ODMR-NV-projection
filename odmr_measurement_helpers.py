"""Spectrum fitting and figure helpers for experimental ODMR sign reconstruction.

Vector-field and plot_field inputs are in Tesla, with colorbars in mT.
Seed and graph-cut diagnostic inputs are already expressed in mT.
Plot labels and conventions follow the QZabre Lorentzian-fit workflow.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from helpers import (build_x_axis, maybe_normalize, maybe_smooth,
                     nv_axis_from_angles, reconstruct_B_from_Bparallel_Fourier)
from fits import fit_one_peak, _subset_by_D

def fit_spectrum(sub, *, normalize=True, smooth_win=3, D_Hz=2.87e9,
                 crop_around_D=True, keep_global_min=False):
    """Fit a single Lorentzian dip after normalization, smoothing and optional cropping."""
    sub = sub.sort_values('k')
    x, _ = build_x_axis(sub, freq_mode='frequences_odmr')
    y_raw = maybe_normalize(sub['sig'].to_numpy(float), normalize)
    y = maybe_smooth(y_raw, smooth_win)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or len(x) < 5:
        return x, y_raw, x, y, dict(success=False, reason='Spectre incomplet/non fini')
    xf, yf = x, y
    if crop_around_D:
        xf, yf, _, _ = _subset_by_D(x, y, D_Hz, side='auto',
                                    keep_if_global_min=keep_global_min)
    fit = fit_one_peak(xf, yf, model='lorentz', baseline_to_0=True)
    return x, y_raw, xf, yf, fit

def plot_field(B_T, extent=None, title="Champ B", png_path=None,
               clim_mT=None, ax=None, show=True, add_colorbar=True, cmap="plasma",
               scalebar=False, Size_um=None, scalebar_frac=0.25,
               scalebar_xy=(0.05, 0.05), scalebar_lw_outer=4,
               scalebar_lw_inner=2, scalebar_fontsize=9):
    """
    Plot simple d'un champ magnétique, compatible subplot.

    Parameters
    ----------
    B_T : 2D array
        Champ en Tesla.
    extent : tuple, optional
        Extent matplotlib : (xmin, xmax, ymin, ymax).
    title : str
        Titre du plot.
    png_path : str, optional
        Chemin de sauvegarde PNG.
    clim_mT : tuple, optional
        Limites couleur en mT : (vmin, vmax).
    ax : matplotlib.axes.Axes, optional
        Axe existant pour afficher dans une subfigure.
    show : bool
        Affiche la figure si True. À mettre False pour des subplots.
    add_colorbar : bool
        Ajoute une colorbar si True.
    cmap : str
        Colormap matplotlib.

    scalebar : bool
        Ajoute une barre d'échelle si True.
    Size_um : float, optional
        Taille physique du côté affiché, en micromètres.
        Exemple : Size_um = param_exp["rect"]["size"][0] * 1e6
    scalebar_frac : float
        Fraction du côté affiché utilisée pour la barre d'échelle.
        Exemple : 0.25 donne une barre de longueur 25% du champ affiché.
    scalebar_xy : tuple
        Position de départ de la barre en coordonnées d'axe, entre 0 et 1.
        Exemple : (0.05, 0.05) place la barre en bas à gauche.
    """

    def _format_length_um(length_um):
        if length_um >= 1000:
            return length_um / 1000, "mm"
        elif length_um >= 1:
            return length_um, "µm"
        else:
            return length_um * 1000, "nm"

    def _add_scalebar(ax, Size_um, scalebar_frac=0.25, scalebar_xy=(0.05, 0.05)):
        bar_um = scalebar_frac * Size_um

        x0, y0 = scalebar_xy
        x1, y1 = x0 + scalebar_frac, y0

        # Trait noir épais en dessous
        ann1 = ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-",
                lw=scalebar_lw_outer,
                color="k",
            ),
            annotation_clip=False,
        )
        ann1.set_zorder(10000)

        # Trait blanc pointillé par-dessus
        ann2 = ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-",
                lw=scalebar_lw_inner,
                color="w",
                linestyle=(0, (4, 4)),
            ),
            annotation_clip=False,
        )
        ann2.set_zorder(10001)

        val, unit = _format_length_um(bar_um)
        txt_str = f"{val:.0f} {unit}" if val >= 10 else f"{val:.1f} {unit}"

        txt = ax.text(
            x0 + scalebar_frac / 2,
            y0 + 0.03,
            txt_str,
            transform=ax.transAxes,
            color="w",
            ha="center",
            va="bottom",
            fontsize=scalebar_fontsize,
            bbox=dict(
                facecolor="black",
                alpha=0.5,
                edgecolor="none",
                pad=1.5,
            ),
        )
        txt.set_zorder(10002)
        txt.set_clip_on(False)

    B_mT = np.ma.masked_invalid(B_T * 1e3)
    vmin, vmax = clim_mT if clim_mT is not None else (None, None)

    created_fig = ax is None

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    else:
        fig = ax.figure

    im = ax.imshow(
        B_mT,
        origin="lower",
        interpolation="nearest",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    ax.invert_yaxis()

    if scalebar:
        if Size_um is None:
            raise ValueError("Pour afficher la barre d'échelle, il faut fournir Size_um.")
        _add_scalebar(
            ax=ax,
            Size_um=Size_um,
            scalebar_frac=scalebar_frac,
            scalebar_xy=scalebar_xy,
        )

    if add_colorbar:
        fig.colorbar(im, ax=ax, label="mT")

    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")

    if show and created_fig:
        plt.show()

    return im

def plot_reconstructed_B_quiver_with_zoom(
    Bx_T,
    By_T,
    Bz_T,
    *,
    center_ij,
    side_px,
    extent=None,
    title_full=r"Champ reconstruit : $B_z$ + $(B_x, B_y)$",
    title_zoom=r"Zoom",
    cmap="bwr",
    clim_mT=None,
    clip_percentile=None,
    symmetric=True,
    qstep_full=6,
    qstep_zoom=2,
    quiver_scale_full=18,
    quiver_scale_zoom=18,
    quiver_width_full=0.004,
    quiver_width_zoom=0.006,
    quiver_headwidth=3.5,
    quiver_alpha=0.95,
    normalize_quiver=True,
    quiver_percentile=90,
    flip_quiver_y=True,
    figsize=(11, 4.5),
    add_colorbar=True,
    colorbar_label=r"$B_z$ reconstruit [mT]",
    save_pdf=None,
    save_png=None,
    save_dpi=400,
    show=True,
):
    """
    Plot du champ reconstruit en deux panneaux :
      - gauche : vue globale Bz en bwr + quiver(Bx, By) + carré rouge
      - droite : zoom dans le carré rouge, même représentation

    Parameters
    ----------
    Bx_T, By_T, Bz_T : 2D arrays
        Composantes du champ reconstruit, en Tesla.
    center_ij : tuple
        Centre du carré en pixels : (i0, j0).
    side_px : int
        Côté du carré en pixels.
    extent : list or None
        Extent du plot complet : [xmin, xmax, ymin, ymax].
        Exemple : [0, 7, 0, 7].
    clim_mT : tuple or None
        Limites couleur communes en mT : (vmin, vmax).
    clip_percentile : float or None
        Si None : utilise min/max ou limites symétriques min/max.
        Si par exemple 99.4 : coupe les extrêmes de Bz.
    symmetric : bool
        Si True : colorbar symétrique autour de 0.
    flip_quiver_y : bool
        True correspond à la convention image :
        i=0 en haut, i augmente vers le bas.
    """

    Bx_T = np.asarray(Bx_T, dtype=float)
    By_T = np.asarray(By_T, dtype=float)
    Bz_T = np.asarray(Bz_T, dtype=float)

    NY, NX = Bz_T.shape

    if extent is None:
        extent = [0, NX, 0, NY]

    x_min, x_max, y_min, y_max = extent

    # ---------- ROI en pixels ----------
    i0, j0 = center_ij
    half = side_px / 2

    i1 = max(0, int(np.floor(i0 - half)))
    i2 = min(NY, int(np.ceil(i0 + half)))
    j1 = max(0, int(np.floor(j0 - half)))
    j2 = min(NX, int(np.ceil(j0 + half)))

    if i2 <= i1 or j2 <= j1:
        raise ValueError("ROI vide pour les paramètres center_ij et side_px.")

    # Conversion pixels -> coordonnées affichées
    def pix_j_to_x(j):
        return x_min + (j / NX) * (x_max - x_min)

    def pix_i_to_y(i):
        return y_min + (i / NY) * (y_max - y_min)

    x1 = pix_j_to_x(j1)
    x2 = pix_j_to_x(j2)
    y1 = pix_i_to_y(i1)
    y2 = pix_i_to_y(i2)

    extent_zoom = [x1, x2, y1, y2]

    # ---------- limites couleur communes ----------
    Bz_mT = np.ma.masked_invalid(Bz_T * 1e3)
    vals = np.asarray(Bz_mT.compressed())

    if clim_mT is not None:
        vmin, vmax = clim_mT

    else:
        if vals.size == 0:
            vmin, vmax = -1.0, 1.0

        elif clip_percentile is None:
            if symmetric:
                vmax_abs = np.nanmax(np.abs(vals))
                if not np.isfinite(vmax_abs) or vmax_abs == 0:
                    vmax_abs = 1.0
                vmin, vmax = -vmax_abs, vmax_abs
            else:
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)

        else:
            if symmetric:
                vmax_abs = np.nanpercentile(np.abs(vals), clip_percentile)
                if not np.isfinite(vmax_abs) or vmax_abs == 0:
                    vmax_abs = np.nanmax(np.abs(vals))
                if not np.isfinite(vmax_abs) or vmax_abs == 0:
                    vmax_abs = 1.0
                vmin, vmax = -vmax_abs, vmax_abs
            else:
                lower_pct = 100 - clip_percentile
                upper_pct = clip_percentile
                vmin = np.nanpercentile(vals, lower_pct)
                vmax = np.nanpercentile(vals, upper_pct)

    # ---------- helper interne ----------
    def _plot_quiver_panel(
        ax,
        Bx,
        By,
        Bz,
        extent_panel,
        qstep,
        quiver_scale,
        quiver_width,
        title,
    ):
        Bz_panel_mT = np.ma.masked_invalid(Bz * 1e3)

        im = ax.imshow(
            Bz_panel_mT,
            origin="lower",
            extent=extent_panel,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )

        ax.invert_yaxis()

        ny, nx = Bz.shape
        xp_min, xp_max, yp_min, yp_max = extent_panel

        x = np.linspace(xp_min, xp_max, nx)
        y = np.linspace(yp_min, yp_max, ny)
        X, Y = np.meshgrid(x, y, indexing="xy")

        Xq = X[::qstep, ::qstep]
        Yq = Y[::qstep, ::qstep]

        U = Bx[::qstep, ::qstep]
        V = By[::qstep, ::qstep]

        if flip_quiver_y:
            V = -V

        if normalize_quiver:
            mag = np.sqrt(U**2 + V**2)
            valid = np.isfinite(mag) & (mag > 0)

            if np.any(valid):
                ref = np.nanpercentile(mag[valid], quiver_percentile)
            else:
                ref = 1.0

            U_plot = U / ref
            V_plot = V / ref
        else:
            U_plot = U
            V_plot = V

        q = ax.quiver(
            Xq,
            Yq,
            U_plot,
            V_plot,
            color="k",
            alpha=quiver_alpha,
            scale=quiver_scale,
            width=quiver_width,
            headwidth=quiver_headwidth,
        )

        ax.set_title(title)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_aspect("equal")

        return im, q

    # ---------- figure ----------
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # Panneau gauche : champ reconstruit complet
    im_full, q_full = _plot_quiver_panel(
        axes[0],
        Bx_T,
        By_T,
        Bz_T,
        extent,
        qstep_full,
        quiver_scale_full,
        quiver_width_full,
        title_full,
    )

    # Carré rouge sur le panneau gauche
    rect = Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor="lime",
        linewidth=2.0,
        zorder=1000,
    )
    axes[0].add_patch(rect)

    # Panneau droit : zoom dans la ROI
    Bx_zoom = Bx_T[i1:i2, j1:j2]
    By_zoom = By_T[i1:i2, j1:j2]
    Bz_zoom = Bz_T[i1:i2, j1:j2]

    im_zoom, q_zoom = _plot_quiver_panel(
        axes[1],
        Bx_zoom,
        By_zoom,
        Bz_zoom,
        extent_zoom,
        qstep_zoom,
        quiver_scale_zoom,
        quiver_width_zoom,
        title_zoom,
    )

    if add_colorbar:
        fig.colorbar(
            im_full,
            ax=axes,
            label=colorbar_label,
            fraction=0.046,
            pad=0.04,
        )

    if save_pdf is not None:
        Path(save_pdf).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_pdf, bbox_inches="tight", facecolor="white")

    if save_png is not None:
        Path(save_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_png, dpi=save_dpi, bbox_inches="tight", facecolor="white")

    if show:
        plt.show()

    return fig, axes, rect, im_full, im_zoom, q_full, q_zoom, (i1, i2, j1, j2)



def plot_seeds(A, seed_mask=None, title="Graines de signe", vmin=None, vmax=None):
    """
    Affiche A avec les graines + et - superposées.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="|B_parallel| (mT)")

    if seed_mask is not None:
        plus = np.where(seed_mask == +1)
        minus = np.where(seed_mask == -1)

        ax.scatter(plus[1], plus[0], s=20, marker="o", label="+ seeds", color="r")
        ax.scatter(minus[1], minus[0], s=20, marker="x", label="- seeds", color="b")
        ax.legend(loc="upper right")

    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.tight_layout()
    plt.show()


def plot_graphcut_result(
    A,
    B_signed,
    sign_map=None,
    seed_mask=None,
    title_prefix="Graph cut sign unwrapping",
    symmetric_B=True,
):
    """
    Affiche A, B signé, signe reconstruit, et graines.
    """
    A = np.asarray(A, dtype=float)
    B_signed = np.asarray(B_signed, dtype=float)

    if sign_map is None:
        sign_map = np.sign(B_signed)

    if symmetric_B:
        vmaxB = np.nanpercentile(np.abs(B_signed), 99)
        vminB = -vmaxB
    else:
        vminB = np.nanpercentile(B_signed, 1)
        vmaxB = np.nanpercentile(B_signed, 99)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    im0 = axes[0].imshow(A, origin="upper", cmap="viridis")
    axes[0].set_title(r"$A=|B_\parallel|$ (mT)")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(B_signed, origin="upper", cmap="coolwarm", vmin=vminB, vmax=vmaxB)
    axes[1].set_title(r"$B_\parallel$ signé (mT)")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(sign_map, origin="upper", cmap="bwr", vmin=-1, vmax=1)
    axes[2].set_title("Signe reconstruit")
    axes[2].set_xlabel("j")
    axes[2].set_ylabel("i")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    if seed_mask is not None:
        for ax in axes:
            plus = np.where(seed_mask == +1)
            minus = np.where(seed_mask == -1)
            ax.scatter(plus[1], plus[0], s=20, marker="o", label="+ seeds")
            ax.scatter(minus[1], minus[0], s=20, marker="x", label="- seeds")

    plt.suptitle(title_prefix)
    plt.tight_layout()
    plt.show()


def plot_zero_and_boundaries(A, B_signed, tau_zero_mT=0.15, title="Zéro et frontières de signe"):
    """
    Affiche les bandes A < tau_zero_mT et les frontières du signe reconstruit.
    """
    A = np.asarray(A, dtype=float)
    B_signed = np.asarray(B_signed, dtype=float)

    zero_mask = A < tau_zero_mT
    sign_map = np.sign(B_signed)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(A, origin="upper", cmap="viridis")
    plt.colorbar(im, ax=ax, label="|B_parallel| (mT)")

    ax.contour(zero_mask.astype(float), levels=[0.5], linewidths=1.5)
    ax.contour(sign_map, levels=[0], linewidths=1.5)

    ax.set_title(title)
    ax.set_xlabel("j")
    ax.set_ylabel("i")
    plt.tight_layout()
    plt.show()


def plot_reconstructed_B_quiver(
    Bpar_T,
    dx,
    d_meas,
    *,
    theta_deg=54.75,
    phi_deg=0.0,
    nv_axis=None,
    lambda0=None,
    p=10,
    dc_mode="zero",
    extent_um=None,
    title=None,
    qstep=6,
    quiver_scale=18,
    quiver_width=0.004,
    quiver_headwidth=3.5,
    quiver_alpha=0.95,
    quiver_percentile=90,
    normalize_quiver=True,
    cmap="bwr",
    clim_mT=None,
    clip_percentile=99.4,
    color_scale_mode="symmetric",  # "symmetric" ou "minmax"
    mirror_y=True,
    figsize=(5.5, 4.5),
    ax=None,
    add_colorbar=True,
    colorbar_label=r"$B_z$ [mT]",
    show=True,
    pdf_path=None,
    png_path=None,
    save_dpi=400,
):
    """
    Reconstruit Bx, By, Bz depuis B∥ puis affiche :
      - colormap : Bz reconstruit en mT
      - quiver   : Bx, By reconstruits

    mirror_y=True :
      Affiche y=0 en haut et y négatif vers le bas,
      cohérent avec des images indexées comme i,j.
    """

    Bpar_T = np.asarray(Bpar_T, dtype=float)
    NY, NX = Bpar_T.shape

    if nv_axis is None:
        nv_axis = nv_axis_from_angles(theta_deg=theta_deg, phi_deg=phi_deg)

    if extent_um is None:
        extent_um = [0, NX * dx * 1e6, 0, NY * dx * 1e6]

    x_min, x_max, y_min, y_max = extent_um

    # Reconstruction Fourier
    Bx_rec, By_rec, Bz_rec = reconstruct_B_from_Bparallel_Fourier(
        Bpar_T=Bpar_T,
        nv_axis=nv_axis,
        dx=dx,
        d_meas=d_meas,
        lambda0=lambda0,
        p=p,
        dc_mode=dc_mode,
    )

    created_fig = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    Bz_mT = np.ma.masked_invalid(Bz_rec * 1e3)

    # ---------- limites couleur ----------
    if clim_mT is not None:
        vmin, vmax = clim_mT
    else:
        vals = np.asarray(Bz_mT.compressed())

        if vals.size == 0:
            vmin, vmax = -1.0, 1.0

        elif color_scale_mode == "symmetric":
            vmax_abs = np.nanpercentile(np.abs(vals), clip_percentile)
            if not np.isfinite(vmax_abs) or vmax_abs == 0:
                vmax_abs = np.nanmax(np.abs(vals))
            if not np.isfinite(vmax_abs) or vmax_abs == 0:
                vmax_abs = 1.0
            vmin, vmax = -vmax_abs, vmax_abs

        elif color_scale_mode == "minmax":
            lower_pct = 100 - clip_percentile
            upper_pct = clip_percentile
            vmin = np.nanpercentile(vals, lower_pct)
            vmax = np.nanpercentile(vals, upper_pct)
            if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                vmin, vmax = np.nanmin(vals), np.nanmax(vals)

        else:
            raise ValueError("color_scale_mode doit être 'symmetric' ou 'minmax'.")

    # ---------- symétrie y pour affichage ----------
    if mirror_y:
        # y = 0 en haut, y négatif vers le bas
        extent_plot = [x_min, x_max, -y_max, -y_min]
        origin_plot = "upper"
    else:
        extent_plot = extent_um
        origin_plot = "lower"

    im = ax.imshow(
        Bz_mT,
        origin=origin_plot,
        extent=extent_plot,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # ---------- grille quiver ----------
    x = np.linspace(x_min, x_max, NX)

    if mirror_y:
        y = np.linspace(-y_min, -y_max, NY)
    else:
        y = np.linspace(y_min, y_max, NY)

    X, Y = np.meshgrid(x, y, indexing="xy")

    Xq = X[::qstep, ::qstep]
    Yq = Y[::qstep, ::qstep]

    U = Bx_rec[::qstep, ::qstep]
    V = By_rec[::qstep, ::qstep]

    # Si on fait y -> -y, la composante verticale doit aussi changer de signe
    if mirror_y:
        V = -V

    # ---------- normalisation optionnelle des flèches ----------
    if normalize_quiver:
        mag = np.sqrt(U**2 + V**2)
        valid = np.isfinite(mag) & (mag > 0)

        if np.any(valid):
            ref = np.nanpercentile(mag[valid], quiver_percentile)
        else:
            ref = 1.0

        U_plot = U / ref
        V_plot = V / ref
    else:
        U_plot = U
        V_plot = V

    q = ax.quiver(
        Xq,
        Yq,
        U_plot,
        V_plot,
        color="k",
        alpha=quiver_alpha,
        scale=quiver_scale,
        width=quiver_width,
        headwidth=quiver_headwidth,
    )

    if title is None:
        title = "" #rf"Champ reconstruit, NV $(\theta={theta_deg:.2f}^\circ, \phi={phi_deg:.1f}^\circ)$"

    ax.set_title(title)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")

    if add_colorbar:
        fig.colorbar(
            im,
            ax=ax,
            label=colorbar_label,
            fraction=0.046,
            pad=0.04,
        )

    if pdf_path is not None:
        from pathlib import Path
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")

    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")

    if show and created_fig:
        plt.show()

    return fig, ax, im, q, (Bx_rec, By_rec, Bz_rec)
