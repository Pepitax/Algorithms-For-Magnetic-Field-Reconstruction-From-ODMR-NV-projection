import numpy as np
import pandas as pd
import os
import json

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as patches
import matplotlib.ticker as mticker

import matplotlib.cm as cm
import matplotlib.colors as mcolors

from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

from matplotlib.patches import Rectangle
from contextlib import contextmanager

import re




#from odmr_map import fit_map_two_lorentz
#from fits import _init_guesses, _model_two_lorentz

#***** CONVERT DATAS TO TXT **********************
import h5py

def convert_to_txt(h5_name, txt_name=None, dataset="data",
                   sep="\t", float_fmt="%.17g", overwrite=False, chunk=10, header=True):
    """
    Convertit un dataset HDF5 en texte "long" sans perte :
    colonnes = i, j[, k], <champ1>[, <champ2>...]

    - h5_name: chemin du .h5
    - txt_name: chemin du .txt (déduit si None)
    - dataset: chemin du dataset (par défaut 'data')
    - sep: séparateur
    - float_fmt: format pour flottants (timetag)
    - overwrite: autoriser l'écrasement du .txt existant
    - chunk: nb de tranches sur l'axe 0 lues en une fois
    - header: écrire une ligne d'en-tête
    """
    if txt_name is None:
        base = os.path.splitext(h5_name)[0]
        txt_name = f"{base}.txt"

    if os.path.isfile(txt_name):
        assert overwrite, f"Won't overwrite {txt_name} (pass overwrite=True)"
        os.remove(txt_name)

    with h5py.File(h5_name, "r") as f, open(txt_name, "w", encoding="utf-8") as fh:
        d = f[dataset]
        ndim = d.ndim
        if ndim not in (1, 2, 3):
            raise ValueError(f"Dataset ndim={ndim} non supporté (attendu 1, 2 ou 3).")

        # Colonnes d'indices
        idx_cols = ["i", "j", "k"][:ndim]

        # Colonnes de données selon dtype (composé ou non)
        if d.dtype.names:  # dtype composé
            field_names = list(d.dtype.names)
            field_dtypes = [d.dtype.fields[name][0] for name in field_names]
        else:
            field_names = ["value"]
            field_dtypes = [d.dtype]

        # Formats par colonne
        def _fmt_for_dt(dt):
            if np.issubdtype(dt, np.integer):
                return "%d"
            elif np.issubdtype(dt, np.floating):
                return float_fmt
            else:
                return "%s"

        fmts = ["%d"] * ndim + [_fmt_for_dt(dt) for dt in field_dtypes] # prepare les labels des colonnes, ici ->  i j k + counts, timetag

        # En-tête
        if header:
            fh.write(sep.join(idx_cols + field_names) + "\n")

        # Dimensions
        dims = d.shape  # ex (110, 110, 100)
        n0 = dims[0]
        n1 = dims[1] if ndim >= 2 else 1
        n2 = dims[2] if ndim == 3 else 1

        # Pré-calcul indices j, k (vecteurs)
        j_vect = np.arange(n1)
        k_vect = np.arange(n2)

        for start in range(0, n0, chunk):
            stop = min(start + chunk, n0)
            B = stop - start

            block = d[start:stop]  # shape: (B, n1[, n2]) ou (B,)
            # Construire les colonnes d'indices aux bonnes tailles
            if ndim == 1:
                I = np.arange(start, stop)  # (B,)
                I = I.reshape(-1, 1)        # (B,1)
                # Valeurs
                if d.dtype.names:
                    vals = [block[name].reshape(B, 1) for name in field_names]
                else:
                    vals = [np.asarray(block).reshape(B, 1)]
                rows = np.column_stack([I] + vals)  # (B, 1 + n_fields)

            elif ndim == 2:
                # I et J de longueur B*n1
                I = np.broadcast_to(np.arange(start, stop)[:, None], (B, n1))
                J = np.broadcast_to(j_vect[None, :], (B, n1))
                # Valeurs (B, n1)
                if d.dtype.names:
                    vals = [block[name].reshape(B, n1) for name in field_names]
                else:
                    vals = [np.asarray(block).reshape(B, n1)]
                rows = np.column_stack([I.reshape(-1), J.reshape(-1),
                                        *[v.reshape(-1) for v in vals]])

            else:  # ndim == 3
                # I, J, K de longueur B*n1*n2
                I = np.broadcast_to(np.arange(start, stop)[:, None, None], (B, n1, n2))
                J = np.broadcast_to(j_vect[None, :, None], (B, n1, n2))
                K = np.broadcast_to(k_vect[None, None, :], (B, n1, n2))
                # Valeurs (B, n1, n2)
                if d.dtype.names:
                    vals = [block[name] for name in field_names]  # déjà (B, n1, n2)
                else:
                    vals = [np.asarray(block)]
                rows = np.column_stack([
                    I.reshape(-1), J.reshape(-1), K.reshape(-1),
                    *[v.reshape(-1) for v in vals]
                ])

            # Écriture du bloc
            np.savetxt(fh, rows, delimiter=sep, fmt=fmts)

    return txt_name


#BOUCLE SUR TOUS LES DOSSIERS
# Chemin racine vers le dossier Datas2 contenant les données
ROOT_DATAS2 = Path(__file__).resolve().parent / "Datas2"

def extract_measure_date(path: Path):
    """
    Cherche une date dans le chemin.
    Priorité au format complet YYYY-MM-DD-HH-MM-SS,
    sinon fallback sur YYYY-MM-DD.
    """
    for part in reversed(path.parts):
        m = re.search(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}", part)
        if m:
            return datetime.strptime(m.group(), "%Y-%m-%d-%H-%M-%S")

    for part in reversed(path.parts):
        m = re.search(r"\d{4}-\d{2}-\d{2}", part)
        if m:
            return datetime.strptime(m.group(), "%Y-%m-%d")

    return None

def convert_all_odmr_and_scalar(root=ROOT_DATAS2, show_skip=True, date_min=None, date_max=None):   #date_min="2025-06-01"
    root = Path(root)
    print(f"Parcours de {root} ...\n")

    if isinstance(date_min, str):
        date_min = datetime.strptime(date_min, "%Y-%m-%d")

    if isinstance(date_max, str):
        date_max = datetime.strptime(date_max, "%Y-%m-%d").date() 

    for dirpath, dirnames, filenames in os.walk(root):
        dirpath = Path(dirpath)
        # ancienne version du code
        # if date_min is not None:
        #     if isinstance(date_min, str):
        #         date_min = datetime.strptime(date_min, "%Y-%m-%d")
        #     try:
        #         mesure_date = datetime.strptime(dirpath.name[:19], "%Y-%m-%d-%H-%M-%S")
        #         if mesure_date < date_min:
        #             continue
        #     except ValueError:
        #         pass

        # ----- odmr.h5 -----
        if "odmr.h5" in filenames:
            h5_path = dirpath / "odmr.h5"
            txt_path = h5_path.with_suffix(".txt")
            #ajout pour labo
            if date_min is not None:
                mesure_date = extract_measure_date(h5_path)
                if mesure_date is not None and mesure_date < date_min:
                    continue
            if date_max is not None:
                mesure_date = extract_measure_date(h5_path)
                if mesure_date is not None and mesure_date.date() > date_max:
                    continue

            if txt_path.exists():
                if show_skip:
                    print(f"[SKIP] {txt_path.name} existe déjà         {filenames}")
            else:
                print(f"[CONVERT] {h5_path} ---> {txt_path.name}")
                convert_to_txt(
                    h5_name=str(h5_path),
                    dataset="data",
                    sep="\t",
                    float_fmt="%.17g",
                    overwrite=False,
                    chunk=10
                )

        # ----- scalarData.h5 -----
        if "scalarData.h5" in filenames:
            h5_path = dirpath / "scalarData.h5"
            txt_path = h5_path.with_suffix(".txt")
            #ajout labo
            if date_min is not None:
                mesure_date = extract_measure_date(h5_path)
                if mesure_date is not None and mesure_date < date_min:
                    continue
            if date_max is not None:
                mesure_date = extract_measure_date(h5_path)
                if mesure_date is not None and mesure_date.date() > date_max:
                    continue

            if txt_path.exists():
                if show_skip:
                    print(f"[SKIP] {txt_path.name} existe déjà   {filenames}")
            else:
                print(f"[CONVERT] {h5_path} ---> {txt_path.name}")
                convert_to_txt(
                    h5_name=str(h5_path),
                    dataset="data",
                    sep="\t",
                    float_fmt="%.17g",
                    overwrite=False,
                    chunk=10
                )

    print("\nTerminé (fichiers .h5 convertis en .txt)")

#Liste de dates :
def build_date_list(root=ROOT_DATAS2, suffix="-odmr_hardware", date_min=None, date_max=None):
    root = Path(root)
    dates = []
    if isinstance(date_min, str):
        date_min = datetime.strptime(date_min, "%Y-%m-%d")
    if isinstance(date_max, str):
        date_max = datetime.strptime(date_max, "%Y-%m-%d").date()

    #for d in root.iterdir():
    for d in root.rglob(f"*{suffix}"):
        if d.is_dir() and d.name.endswith(suffix):
            #enlève le suffixe pour ne garder que "2025-11-22-09-40-00"
            #date_str = d.name[:-len(suffix)]
            #dates.append(date_str)

            # code corrigé pour le labo
            date_str = d.name[:-len(suffix)]
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d-%H-%M-%S")
            except ValueError:
                continue
            if date_min is not None and date_obj < date_min:
                continue
            if date_max is not None and date_obj.date() > date_max:
                continue
            dates.append(date_str)

    # tri (les strings de ce format se trient déjà dans l'ordre chronologique)
    dates = sorted(set(dates))
    return dates






#***** LOAD DATAS ********************
def load_df1(TXT_PATH, DOWNCAST_INT, infos=None):
    df = pd.read_csv(TXT_PATH, sep="\t")

    # Types
    if DOWNCAST_INT:
        #df["counts"] = pd.to_numeric(df["counts"], downcast="integer")
        df["sig"] = pd.to_numeric(df["sig"], downcast="integer")

    # Dimensions brutes (avant rotation)
    n_i_raw = int(df["i"].max() + 1)
    n_j_raw = int(df["j"].max() + 1)

    # --- Rotation de 90° (exemple: horaire) sur les indices (i,j) --- POUR ÊTRE AFFICHE COMME SUR QZABRE
    i_old = df["i"].to_numpy()
    j_old = df["j"].to_numpy()

    i_new = j_old               # 90° horaire : i' = j
    j_new = n_i_raw - 1 - i_old #            et j' = n_i - 1 - i

    df["i"] = i_new
    df["j"] = j_new

    # Ordonner par i,j,k
    df = df.sort_values(["i","j","k"]).reset_index(drop=True)

    if infos:
        # Dimensions déduites
        n_i = int(df["i"].max() + 1)
        n_j = int(df["j"].max() + 1)
        n_k = int(df["k"].max() + 1)
        print(f"Dimensions: i={n_i}, j={n_j}, k={n_k}")
        print(df.keys())

    return df


def load_df2(SCALAR_PATH: Path, infos=None) -> pd.DataFrame:
    df2 = pd.read_csv(SCALAR_PATH, sep="\t")
    df2 = df2.sort_values(["i", "j"]).reset_index(drop=True)

    if infos:
        # Dimensions déduites
        sn_i = int(df2["i"].max() + 1)
        sn_j = int(df2["j"].max() + 1)
        print(f"Dimensions: i={sn_i}, j={sn_j}")
        print(df2.keys())
    
    return df2


def concatener_data_df1(df: pd.DataFrame, df_band: pd.DataFrame, infos = False) -> pd.DataFrame:
    """
    Concatène un scan principal 3D (i,j,k) avec une bande supplémentaire.
    ici, bande à gauche; main mesure à droite; 1 colonne blanche entre les deux.
    Recalage des indices lors de la concaténation de bandes de mesure.
    """
    # 1) Taille (en j) de la bande
    j_min_band = df_band["j"].min()
    j_band_size = df_band["j"].max() - j_min_band + 1

    # 2) Garder la partie droite du scan principal (on enlève la gauche)
    j_new_min = j_band_size           # ici 28

    df_main = df[df["j"] >= j_new_min].copy()

    # 3) Recaler la bande pour qu'elle occupe j = 0..(j_band_size-1)
    df_band = df_band.copy()
    df_band["j"] = df_band["j"] - j_min_band   # met la bande sur 0..27

    # 3b) Décaler le main d'un pixel vers la droite pour créer une colonne blanche
    df_main["j"] = df_main["j"] + 1           # <<=== AJOUT POUR LA BANDE BLANCHE

    # 4) Concat + tri
    df_final = pd.concat([df_band, df_main], ignore_index=True, axis=0)
    df_final = df_final.sort_values(["i", "j", "k"]).reset_index(drop=True)

    if infos:
        print("j_min_band =", j_min_band, "j_band_size =", j_band_size)
        print("j_new_min =", j_new_min)
        print(f"-> df tronqué. Nouveau j_min (principal): {df_main['j'].min()}, "f"j_max: {df_main['j'].max()}")
        n_i_1 = int(df_main["i"].max() + 1)
        n_j_1 = int(df_main["j"].max() + 1)
        n_k_1 = int(df_main["k"].max() + 1)
        print(f"-> Dimensions main: i={n_i_1}, j={n_j_1}, k={n_k_1}")

        n_i_2 = int(df_band["i"].max() + 1)
        n_j_2 = int(df_band["j"].max() + 1)
        n_k_2 = int(df_band["k"].max() + 1)
        print(f"-> Dimensions band: i={n_i_2}, j={n_j_2}, k={n_k_2}")

        n_i = int(df_final["i"].max() + 1)
        n_j = int(df_final["j"].max() + 1)
        n_k = int(df_final["k"].max() + 1)
        print(f"-> Dimensions combinées: i={n_i}, j={n_j}, k={n_k}")

    return df_final






def concatener_data_df2(df2, df_band, infos=False):        
    # --- Écraser la zone vide du scan principal (df2) ---> La bande vide correspond aux dernières lignes i du scan original.
    i_band_size = df_band['i'].max() + 0 # Hauteur de la bande = 28              # Dimensions de la bande : i_band_size = 28.
    i_new_max = df2['i'].max() - i_band_size
    
    # TRONQUER df2 pour supprimer la zone que la bande va recouvrir (l'espace vide)
    df2 = df2[df2['i'] <= i_new_max].copy()
    
    # --- Calcul et Application de l'Offset ---
    i_max_scan1_new = df2['i'].max()
    i_offset = i_max_scan1_new + 1 
    j_offset = 0 # Collage vertical
    
    # Application de l'Offset
    df_band['i'] = df_band['i'] + i_offset
    # Le j_offset est 0, donc pas de modification de 'j'
    
    # 4. Concaténer et mettre à jour df2 ---> Il ne devrait plus y avoir de chevauchement d'indices (i,j)
    df_final = pd.concat([df2, df_band], ignore_index=True, axis=0)
    
    if infos:
        print(f"-> df2 tronqué. Nouvel i_max (principal): {df2['i'].max()}")

        sn_i_1 = df2['i'].max() + 1
        sn_j_1 = df2['j'].max() + 1
        print(f"-> Dimensions main: i={sn_i_1}, j={sn_j_1}")

        sn_i_2 = df_band['i'].max() + 1
        sn_j_2 = df_band['j'].max() + 1
        print(f"-> Dimensions band: i={sn_i_2}, j={sn_j_2}")

        sn_i_combined = df_final['i'].max() + 1
        sn_j_combined = df_final['j'].max() + 1
        print(f"-> Dimensions combinées: i={sn_i_combined}, j={sn_j_combined}")

    return df_final




#***** Helpers pour les PLOTS FUNCTIONS *****
def _mean_plane_subtract(g):
    """Soustrait un plan global a*x + b*y + c."""
    i_vals = g.columns.values.astype(float)
    j_vals = g.index.values.astype(float)
    I, J = np.meshgrid(i_vals, j_vals)  # shape (n_j, n_i)
    Z = g.to_numpy().astype(float)

    mask = np.isfinite(Z)
    if mask.sum() < 3:
        return g  # pas assez de points pour fitter

    x = I[mask]
    y = J[mask]
    z = Z[mask]

    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    plane = coeffs[0]*I + coeffs[1]*J + coeffs[2]

    Z_corr = Z - plane
    return pd.DataFrame(Z_corr, index=g.index, columns=g.columns)

def _parabolic_background_subtract(g):
    """Soustrait une quadratique 2D a x^2 + b y^2 + c x y + d x + e y + f."""
    i_vals = g.columns.values.astype(float)
    j_vals = g.index.values.astype(float)
    I, J = np.meshgrid(i_vals, j_vals)
    Z = g.to_numpy().astype(float)

    mask = np.isfinite(Z)
    if mask.sum() < 6:
        return g  # pas assez de points

    x = I[mask]
    y = J[mask]
    z = Z[mask]

    A = np.column_stack([x**2, y**2, x*y, x, y, np.ones_like(x)])
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)

    bg = (coeffs[0]*I**2 + coeffs[1]*J**2 + coeffs[2]*I*J +
          coeffs[3]*I + coeffs[4]*J + coeffs[5])

    Z_corr = Z - bg
    return pd.DataFrame(Z_corr, index=g.index, columns=g.columns)

def _flatten_linear_x(g):
    """Pour chaque ligne j, enlève a x + b (fit linéaire en X)."""
    Z = g.to_numpy().astype(float)
    ny, nx = Z.shape
    x = g.columns.values.astype(float)

    for j in range(ny):
        z = Z[j, :]
        mask = np.isfinite(z)
        if mask.sum() >= 2:
            A = np.vstack([x[mask], np.ones(mask.sum())]).T
            coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
            fit = coeffs[0]*x + coeffs[1]
            Z[j, :] = z - fit

    return pd.DataFrame(Z, index=g.index, columns=g.columns)

def _flatten_linear_y(g):
    """Pour chaque colonne i, enlève a y + b (fit linéaire en Y)."""
    Z = g.to_numpy().astype(float)
    ny, nx = Z.shape
    y = g.index.values.astype(float)

    for i in range(nx):
        z = Z[:, i]
        mask = np.isfinite(z)
        if mask.sum() >= 2:
            A = np.vstack([y[mask], np.ones(mask.sum())]).T
            coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
            fit = coeffs[0]*y + coeffs[1]
            Z[:, i] = z - fit

    return pd.DataFrame(Z, index=g.index, columns=g.columns)

def _flatten_parabolic_x(g):
    """Pour chaque ligne j, enlève a x^2 + b x + c (parabole en X)."""
    Z = g.to_numpy().astype(float)
    ny, nx = Z.shape
    x = g.columns.values.astype(float)

    for j in range(ny):
        z = Z[j, :]
        mask = np.isfinite(z)
        if mask.sum() >= 3:
            A = np.vstack([x[mask]**2, x[mask], np.ones(mask.sum())]).T
            coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
            fit = coeffs[0]*x**2 + coeffs[1]*x + coeffs[2]
            Z[j, :] = z - fit

    return pd.DataFrame(Z, index=g.index, columns=g.columns)

def _flatten_parabolic_y(g):
    """Pour chaque colonne i, enlève a y^2 + b y + c (parabole en Y)."""
    Z = g.to_numpy().astype(float)
    ny, nx = Z.shape
    y = g.index.values.astype(float)

    for i in range(nx):
        z = Z[:, i]
        mask = np.isfinite(z)
        if mask.sum() >= 3:
            A = np.vstack([y[mask]**2, y[mask], np.ones(mask.sum())]).T
            coeffs, *_ = np.linalg.lstsq(A, z[mask], rcond=None)
            fit = coeffs[0]*y**2 + coeffs[1]*y + coeffs[2]
            Z[:, i] = z - fit

    return pd.DataFrame(Z, index=g.index, columns=g.columns)

def _apply_preprocess(g, preprocess):
    """
    Applique un prétraitement de type QZabre sur la carte 2D g.
    preprocess peut être par ex. :
      - None / "none"
      - "mean plane subtract" / "mean_plane"
      - "parabolic background subtract" / "parabolic"
      - "linear X"
      - "linear Y"
      - "parabolic X"
      - "parabolic Y"
    """
    if preprocess is None:
        return g

    name = str(preprocess).strip().lower().replace("_", " ")

    if name in ("none", ""):
        return g
    elif name in ("mean plane", "mean plane subtract", "plane"):
        return _mean_plane_subtract(g)
    elif name in ("parabolic background", "parabolic background subtract", "parabolic"):
        return _parabolic_background_subtract(g)
    elif name in ("linear x", "line x"):
        return _flatten_linear_x(g)
    elif name in ("linear y", "line y"):
        return _flatten_linear_y(g)
    elif name in ("parabolic x",):
        return _flatten_parabolic_x(g)
    elif name in ("parabolic y",):
        return _flatten_parabolic_y(g)
    else:
        raise ValueError(f"Prétraitement '{preprocess}' non reconnu")








#***** PLOTS FUNCTIONS ********************

# PLOT SIMPLE pour df2 (données qui viennent de la machine de mesure de QZabre)
def plot_QZabre(df, key, cmap="viridis",
                vmin=None, vmax=None,
                robust_pct=1.0,
                preprocess=None):
    """Affiche `key` sur la grille (i,j), avec option de vmin/vmax, et robuste_percentile pour retirer les outliers dans la colorbar.
    Et option de prétraitement de type QZabre :
      preprocess = {None, "mean plane subtract", "linear X", "linear Y",
                    "parabolic X", "parabolic Y", "parabolic background subtract", ...}
    """
    if key not in df.columns:
        raise ValueError(f"Clé inconnue: {key}")

    # j = lignes (vertical), i = colonnes (horizontal)
    g = (df.pivot(index="j", columns="i", values=key)
           .sort_index()
           .sort_index(axis=1))

    # --- prétraitement à la volée ---
    g = _apply_preprocess(g, preprocess)

    # vmin/vmax robustes si demandé (sur les données déjà prétraitées)
    if robust_pct is not None and (vmin is None or vmax is None):
        vals = g.to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.percentile(vals, [robust_pct, 100 - robust_pct])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

    # Dimensions (indices i,j supposés 0..max sans trous)
    n_i = int(df["i"].max() + 1)
    n_j = int(df["j"].max() + 1)

    fig, ax = plt.subplots()

    im = ax.imshow(g,
                   origin="lower", aspect="equal",
                   cmap=cmap, vmin=vmin, vmax=vmax)

    # inversions que tu avais déjà
    ax.invert_xaxis()
    ax.invert_yaxis()

    # --- Remap des labels sans toucher à l'image ---
    def fmt_x(x, pos):
        # x est la coordonnée "interne", on affiche i = n_i-1-x
        return str(int(round(n_i - x)))

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))

    ax.set_xlabel("j (pixels)")
    ax.set_ylabel("i (pixels)")
    title = key if preprocess is None else f"{key} ({preprocess})"
    ax.set_title(title)
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(0, n_i, 10))
    ax.set_yticks(np.arange(0, n_j, 10))

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(key)

    plt.tight_layout()
    plt.show()



#from mpl_toolkits.mplot3d import Axes3D  # nécessaire pour activer la proj 3D

def plot_QZabre_surf(df, key, cmap="viridis", vmin=None, vmax=None,
                     robust_pct=1.0, elev=30, azim=-60, preprocess=None):
    """Affiche `key` sur la grille (i,j) en surface 3D, avec prétraitement optionnel."""

    if key not in df.columns:
        raise ValueError(f"Clé inconnue: {key}")

    # j = lignes (vertical), i = colonnes (horizontal)
    g = (df.pivot(index="j", columns="i", values=key)
           .sort_index()
           .sort_index(axis=1))

    # --- prétraitement à la volée (comme pour la 2D) ---
    g = _apply_preprocess(g, preprocess)

    # vmin/vmax robustes si demandé (sur les données déjà prétraitées)
    if robust_pct is not None and (vmin is None or vmax is None):
        vals = g.to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.percentile(vals, [robust_pct, 100 - robust_pct])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

    # coordonnées i,j
    j_vals = g.index.to_numpy()      # axe "lignes"
    i_vals = g.columns.to_numpy()    # axe "colonnes"
    J, I = np.meshgrid(j_vals, i_vals, indexing="ij")  # même shape que g
    Z = g.to_numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        J, I, Z,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        linewidth=0, antialiased=True
    )

    ax.set_xlabel("j (pixels)")
    ax.set_ylabel("i (pixels)")
    ax.set_zlabel(key)

    if preprocess is None:
        title = f"{key} (surface)"
    else:
        title = f"{key} (surface, {preprocess})"
    ax.set_title(title)

    # angle de vue par défaut (modifiable à l'appel)
    ax.view_init(elev=elev, azim=azim)

    # colorbar sur les valeurs de Z
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12)
    cbar.set_label(key)

    plt.tight_layout()
    plt.show()








# PLOT POUR SUBFIGURES (pour plot des bandes de couleur qui s'ajoutent par dessus une figure, par exemple le top 25% percentile du contrast)
def overlay_max_band(df, key, ax, top_pct=5.5, alpha=0.4, red=True, yellow=False, blue=False):
    """
    Superpose sur `ax` une bande rouge semi-transparente correspondant
    aux valeurs de `key` au-dessus du percentile (100 - top_pct).
    """
    if key not in df.columns:
        return

    # Pivot comme dans plot_on_ax
    g = (df.pivot(index="j", columns="i", values=key)
           .sort_index()
           .sort_index(axis=1))

    vals = g.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return

    # Seuil = top `top_pct` %
    thr = np.percentile(vals, 100 - top_pct)
    mask = g >= thr      # True pour les maxima

    # Image RGBA : [R, G, B, A]
    overlay = np.zeros(mask.shape + (4,), dtype=float)
    if red:
        overlay[..., 0] = 1.0            # Rouge
    if yellow:
        overlay[..., 1] = 1.0            # Rouge
    if blue:
        overlay[..., 2] = 1.0            # Rouge
    overlay[..., 3] = mask * alpha   # Alpha 0 ou alpha

    # Garder les limites actuelles de l'axe (au cas où)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Superposition sur l'axe (même convention que plot_on_ax : origin="lower")
    ax.imshow(overlay, origin="lower", aspect="equal")

    # Rétablir les limites (important si tu as zoomé, etc.)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def format_length_um(length_um):
    # length_um est en micromètres
    if length_um < 1.0:
        return length_um * 1000.0, "nm"   # 1 µm = 1000 nm
    else:
        return length_um, "µm"

#Fonction pour construire automatiquement une subfigure
def plot_on_ax(df, key, ax, cmap="viridis", vmin=None, vmax=None,
               robust_pct=1.0, preprocess=None,
               Size_um=None, scalebar=True, scalebar_frac=0.25,
               scalebar_xy=(0.05, 0.05)):
    """Affiche `key` sur l'axe `ax` donné."""
    if key not in df.columns:
        ax.text(0.5, 0.5, f"Key '{key}' not found", ha='center', va='center')
        return None

    g = (df.pivot(index="j", columns="i", values=key)
           .sort_index()
           .sort_index(axis=1))

    g = _apply_preprocess(g, preprocess)

    if robust_pct is not None and (vmin is None or vmax is None):
        vals = g.to_numpy().ravel()
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.percentile(vals, [robust_pct, 100 - robust_pct])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax

    n_i = int(df["i"].max() + 1)
    n_j = int(df["j"].max() + 1)

    im = ax.imshow(g, origin="lower", aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.invert_xaxis()
    ax.invert_yaxis()

    def fmt_x(x, pos):
        val = n_i - x
        return str(int(round(val)))

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_x))
    ax.set_xlabel("j ")
    ax.set_ylabel("i ")
    ax.set_title(key)
    ax.xaxis.tick_top()

    step_i = max(1, n_i // 5)
    step_j = max(1, n_j // 5)
    ax.set_xticks(np.arange(0, n_i, step_i))
    ax.set_yticks(np.arange(0, n_j, step_j))

    # ---- BARRE D'ÉCHELLE (noir + pointillé blanc) ----
    if scalebar and (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0

        ann1 = ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-", lw=4, color="k"),
            annotation_clip=False
        )
        ann1.set_zorder(10000)

        ann2 = ax.annotate(
            "", xy=(x1, y1), xytext=(x0, y0),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
            annotation_clip=False
        )
        ann2.set_zorder(10001)

        val, unit = format_length_um(bar_um)
        txt = ax.text(
            x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
            transform=ax.transAxes, color="w",
            ha="center", va="bottom", fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5)
        )
        txt.set_zorder(10002)
        txt.set_clip_on(False)

    return im







# CONSTRUCTION AXE & PRE TRAITEMENT (NORMALISE ? SMOOTH ?) & ROI BOX & CONCATERER DATA
def build_x_axis(sub_df, freq_mode=None):
    """Construit l'abscisse selon FREQ_MODE pour la portion sub_df (doit contenir 'k')."""
    #global F_LABEL
    if freq_mode == "k_linear":
        #x = F_START_HZ + sub_df["k"].to_numpy(dtype=float) * F_STEP_HZ
        x = 1 + sub_df["k"].to_numpy(dtype=float) * 0.1
        F_LABEL = "MW freq [Hz]"
        return x

    else : #elif freq_mode in ("frequences_odmr", "frequency") :  # alias accepté
        x = sub_df["frequency"].to_numpy(dtype=float)
        F_LABEL = "MW freq [Hz]"
        return x, F_LABEL


def maybe_normalize(y, NORMALIZE, return_median=False):
    y = np.asarray(y, float)
    if not NORMALIZE:
        return (y, 1.0) if return_median else y
    med = float(np.median(y))
    if med <= 0:
        med = 1.0
    y_norm = y / med
    return (y_norm, med) if return_median else y_norm


def maybe_smooth(y, SMOOTH_WIN):
    w = int(SMOOTH_WIN)
    if w >= 3 and w % 2 == 1:
        pad = w // 2
        yy = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(w) / w
        y_s = np.convolve(yy, kernel, mode="valid")
        return y_s
    return y


def _add_roi_box(ax, roi_box, **kwargs):
    """
    roi_box: (i0, i1, j0, j1) en indices *absolus* (même repère que extent).
    """
    i0, i1, j0, j1 = roi_box
    width  = (j1 + 1) - j0
    height = (i1 + 1) - i0
    rect = patches.Rectangle(
        (j0, i0),
        width,
        height,
        fill=False,
        linewidth=1.9,
        edgecolor="k",
        **kwargs,
    )
    ax.add_patch(rect)


def overlay_percentile_band_grid(g2d, ax, pct=25.0, alpha=0.3, *, side="top", extent=None):
    """
    g2d: DataFrame/ndarray 2D dans la même orientation que le imshow sous-jacent.
    side="top" -> met en rouge les plus grandes valeurs (top pct%)
    side="bottom" -> met en rouge les plus petites valeurs (bottom pct%)
    """
    arr = np.asarray(g2d)
    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        return

    thr = np.percentile(vals, 100 - pct) if side == "top" else np.percentile(vals, pct)
    mask = (arr >= thr) if side == "top" else (arr <= thr)

    overlay = np.zeros(mask.shape + (4,), float)
    overlay[..., 2] = 0.0
    overlay[..., 3] = mask.astype(float) * alpha

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.imshow(overlay, origin="lower", interpolation="nearest",
              extent=extent, aspect="auto")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    label = f"{side} {pct:.0f}% highest contrast from left figure"
    patch = mpatches.Patch(color=(1, 0, 0, alpha), label=label)  # rouge semi-transparent

    # ajoute / concatène à une légende existante
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [patch], labels + [label], loc="lower center", framealpha=0.8)


# PLOT ROI MAP (avec min frequence)
def plot_contrast_argmin(df, NORMALIZE, SMOOTH_WIN,
                  center_i, center_j, half=2,
                  use_processed=True,
                  grid=True,
                  cmap_contrast="viridis",
                  cmap_min="magma",
                  panel_size=(3.6, 3.0),
                  figsize=None,
                  pdf_path=None,
                  png_path=None,
                  suptitle=None,
                  pdf_metadata=None,
                  roi_box=None,
                  roi_boxes=None,
                  scalebar=True,
                  Size_um=None,
                  scalebar_frac=0.25,
                  scalebar_xy=(0.05, 0.05)
                  ):
    n_i = int(df["i"].max() + 1)
    n_j = int(df["j"].max() + 1)

    i_range = np.arange(max(0, center_i - half), min(n_i, center_i + half + 1))
    j_range = np.arange(max(0, center_j - half), min(n_j, center_j + half + 1))
    if i_range.size == 0 or j_range.size == 0:
        raise ValueError("ROI hors limites — ajuste center_i/center_j/half.")

    xk = df.groupby("k")["frequency"].median().to_numpy()

    df_roi = df[df["i"].isin(i_range) & df["j"].isin(j_range)].copy()
    df_roi = df_roi.sort_values(["i", "j", "k"])

    def series_preproc(s_counts):
        y = s_counts.to_numpy(dtype=float)
        if use_processed:
            y = maybe_normalize(y, NORMALIZE)
            y = maybe_smooth(y, SMOOTH_WIN)
        return y

    def agg_contrast(s_counts):
        y = series_preproc(s_counts)
        return float(np.nanmax(y) - np.nanmin(y))

    def agg_min_x(s_counts):
        y = series_preproc(s_counts)
        k0 = int(np.nanargmin(y))
        return xk[k0]

    contrast = (df_roi.groupby(["i", "j"])["sig"].agg(agg_contrast)
                .unstack("j").reindex(index=i_range, columns=j_range))
    min_x = (df_roi.groupby(["i", "j"])["sig"].agg(agg_min_x)
             .unstack("j").reindex(index=i_range, columns=j_range))

    nrows, ncols = 1, 3
    figsize_use = (panel_size[0]*ncols, panel_size[1]*nrows) if figsize is None else figsize
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize_use)
    ax1, ax2, ax3 = axes

    extent = [j_range.min()-0.5, j_range.max()+0.5,
              i_range.min()-0.5, i_range.max()+0.5]

    def _robust_vmin_vmax(arr2d, pct=5.0):
        v = np.asarray(arr2d, float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return None, None
        return np.percentile(v, [pct, 100 - pct])

    v1min, v1max = _robust_vmin_vmax(contrast, pct=1.5)

    im1 = ax1.imshow(contrast.to_numpy(), origin="lower", aspect="auto",
                     extent=extent, cmap=cmap_contrast, interpolation="nearest",
                     vmin=v1min, vmax=v1max)
    cbar1 = fig.colorbar(im1, ax=ax1); cbar1.set_label("contrast intensity")
    ax1.set_title("Contrast (max-min)")
    ax1.set_xlabel("j"); ax1.set_ylabel("i")
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()

    im2 = ax2.imshow(min_x.to_numpy(), origin="lower", aspect="auto",
                     extent=extent, cmap=cmap_min, interpolation="nearest")
    cbar2 = fig.colorbar(im2, ax=ax2); cbar2.set_label("position of min [Hz]")
    ax2.set_title("Contrast on top of Argmin spectrogram")
    ax2.set_xlabel("j"); ax2.set_ylabel("i")
    ax2.invert_yaxis()
    ax2.xaxis.tick_top()

    im3 = ax3.imshow(min_x.to_numpy(), origin="lower", aspect="auto",
                     extent=extent, cmap=cmap_min, interpolation="nearest")
    cbar3 = fig.colorbar(im3, ax=ax3); cbar3.set_label("position of min [Hz]")
    ax3.set_title("Argmin spectrogram (frequences of resonnance)")
    ax3.set_xlabel("j"); ax3.set_ylabel("i")
    ax3.invert_yaxis()
    ax3.xaxis.tick_top()

    overlay_percentile_band_grid(contrast, ax2, pct=25.0, alpha=0.2, side="top", extent=extent)

    for ax in (ax1, ax2, ax3):
        ax.set_aspect("equal", adjustable="box")

    if grid:
        for ax in (ax1, ax2, ax3):
            ax.set_xticks(j_range)
            ax.set_yticks(i_range)
            ax.grid(which="both", color="k", linestyle=":", linewidth=0.4, alpha=0.3)
            ax.plot(center_j, center_i, marker="s", markersize=8,
                    markerfacecolor="none", markeredgecolor="w", markeredgewidth=1.2)

    # --- ROI boxes (multi) ---
    for ax in (ax1, ax2, ax3):
        _add_roi_boxes(ax, roi_box=roi_box, roi_boxes=roi_boxes, default_lw=2.2)

    try:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    except Exception:
        plt.tight_layout()

    if suptitle is not None:
        fig.suptitle(suptitle)

    # ---- scalebar ----
    if scalebar and (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0

        for ax in (ax1, ax3):
            ann1 = ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                               xycoords="axes fraction",
                               arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                               annotation_clip=False)
            ann1.set_zorder(10000)

            ann2 = ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                               xycoords="axes fraction",
                               arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
                               annotation_clip=False)
            ann2.set_zorder(10001)

            val, unit = format_length_um(bar_um)
            txt = ax.text(x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
                          transform=ax.transAxes, color="w",
                          ha="center", va="bottom", fontsize=9,
                          bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5))
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        if suptitle is not None:
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")

    if pdf_path is not None:
        meta = dict() if pdf_metadata is None else dict(pdf_metadata)
        meta.setdefault("Title", f"ROI maps ({center_i},{center_j}) half={half}")
        with PdfPages(pdf_path, metadata=meta) as pdf:
            pdf.savefig(fig, bbox_inches="tight")

    plt.show()







#***** PLOT TRYPTICH ********************

# --- Constantes physiques (NV) ---
D_Hz = 2.87e9               # zero-field splitting du GS (~ 2.87 GHz)
GAMMA_NV_Hz_per_T = 28.024e9  # γ_NV ~ 28.024 GHz/T (μB*g_e/h)

# ---------- utilitaires pour calculer les cartes B ----------

def _infer_scale_to_Hz(xv1, xv2):
    """
    Détecte si les fréquences sont données en Hz (~1e9) ou en GHz (~1..5),
    et retourne le facteur par lequel multiplier pour obtenir des Hz.
    """
    vals = np.asarray(pd.concat([xv1, xv2]), float)
    m = np.nanmedian(vals)
    if not np.isfinite(m):
        return 1.0
    # Heuristique : si médiane < 1e7, on suppose GHz -> *1e9
    return 1e9 if m < 1e7 else 1.0

def compute_B_components(res, n_i, n_j, clip_negative=False):
    """
    À partir de res (i,j,xv1,xv2), calcule :
      - B_parallel_1 = (f_high - D)/γ_NV
      - B_parallel_2 = (f_low  - D)/γ_NV
      - B_from_split  = (f_high - f_low)/γ_NV
    Retourne (df_B, B1_map, B2_map, Bsplit_map) avec les cartes en Tesla.
    """
    scale = _infer_scale_to_Hz(res["xv1"], res["xv2"])
    D = D_Hz
    g = GAMMA_NV_Hz_per_T

    B1_map     = np.full((n_i, n_j), np.nan, float)
    B2_map     = np.full((n_i, n_j), np.nan, float)
    Bsplit_map = np.full((n_i, n_j), np.nan, float)

    rows = []
    for _, r in res.iterrows():
        i, j = int(r["i"]), int(r["j"])
        f1 = float(r["xv1"]) * scale
        f2 = float(r["xv2"]) * scale
        if not (np.isfinite(f1) and np.isfinite(f2) and f1!=0 and f2!=0):
            rows.append({**r, "B1_T": np.nan, "B2_T": np.nan, "Bsplit_T": np.nan})
            continue

        # On choisit une convention f_low / f_high
        f_low, f_high = (f1, f2)  # ou trie si besoin : (f1,f2) si f1<=f2 else (f2,f1)

        B1 = (f_high - D) / g        # T
        B2 = (f_low  - D) / g        # T
        Bsplit = (f_high - f_low) / g  # T = Δf / γ_NV

        if clip_negative:
            B1 = max(B1, 0.0)
            B2 = max(B2, 0.0)

        B1_map[i, j]     = B1
        B2_map[i, j]     = B2
        Bsplit_map[i, j] = Bsplit
        rows.append({**r, "B1_T": B1, "B2_T": B2, "Bsplit_T": Bsplit})

    df_B = pd.DataFrame(rows)
    return df_B, B1_map, B2_map, Bsplit_map




def _normalize_roi_boxes(roi_box=None, roi_boxes=None,
                         default_color="w", default_lw=2.0, default_ls="-", default_alpha=1.0, default_zorder=9999):
    """
    Retourne une liste normalisée de dicts:
    [{"roi_box":(i0,i1,j0,j1), "color":..., "lw":..., "ls":..., "alpha":..., "zorder":...}, ...]
    Accepte:
      - roi_box = (i0,i1,j0,j1)
      - roi_boxes = [
            (i0,i1,j0,j1),
            ((i0,i1,j0,j1), "red"),
            {"roi_box":(i0,i1,j0,j1), "color":"red", "lw":2, "ls":"-", "alpha":1}
        ]
    """
    out = []

    boxes_in = []
    if roi_box is not None:
        boxes_in.append(roi_box)
    if roi_boxes is not None:
        boxes_in.extend(list(roi_boxes))

    for item in boxes_in:
        color = default_color
        lw    = default_lw
        ls    = default_ls
        alpha = default_alpha
        zorder= default_zorder
        box   = None

        if isinstance(item, dict):
            box   = item.get("roi_box", item.get("box"))
            color = item.get("color", color)
            lw    = item.get("lw", lw)
            ls    = item.get("ls", ls)
            alpha = item.get("alpha", alpha)
            zorder= item.get("zorder", zorder)

        elif isinstance(item, (tuple, list)):
            if len(item) == 2 and isinstance(item[1], str):
                box, color = item
            else:
                box = tuple(item)

        else:
            box = item

        if box is None:
            continue
        if len(box) != 4:
            raise ValueError(f"roi_box doit être (i0,i1,j0,j1), reçu: {box}")

        i0, i1, j0, j1 = box
        out.append(dict(roi_box=(i0, i1, j0, j1), color=color, lw=lw, ls=ls, alpha=alpha, zorder=zorder))

    return out


def _add_roi_boxes(ax, roi_box=None, roi_boxes=None, **defaults):
    """
    Ajoute sur 'ax' les rectangles de ROI.
    Convention roi_box=(i0,i1,j0,j1) en indices inclusifs.
    Dessin en coordonnées data: x=j, y=i.
    """
    boxes = _normalize_roi_boxes(roi_box=roi_box, roi_boxes=roi_boxes, **defaults)
    for b in boxes:
        i0, i1, j0, j1 = b["roi_box"]
        # indices inclusifs -> rectangle aligné sur pixels
        x = j0 - 0.5
        y = i0 - 0.5
        w = (j1 - j0 + 1.0)
        h = (i1 - i0 + 1.0)

        rect = Rectangle(
            (x, y), w, h,
            fill=False,
            edgecolor=b["color"],
            linewidth=b["lw"],
            linestyle=b["ls"],
            alpha=b["alpha"],
            zorder=b["zorder"],
        )
        ax.add_patch(rect)


#@contextmanager
#def defer_show():
#    old_show = plt.show
#    plt.show = lambda *a, **k: None
#    try:
#        yield
#    finally:
#        plt.show = old_show


# ---------- fonctions de tracé pour le champ projeté sur l'axe NV ----------

def plot_B1_B2_DeltaB_projectionNV(B1_T, B2_T, Bsplit_T, extent,
                         roi_box=None, roi_boxes=None,
                         png_path=None, clim3_mT=None, suptitle=None,
                         Size_um=None, scalebar=True, scalebar_frac=0.25, scalebar_xy=(0.05, 0.05)):

    cmap = plt.cm.get_cmap('turbo').copy()
    cmap.set_bad(alpha=0.0)

    A1 = np.ma.masked_invalid(B1_T * 1e3)
    A2 = np.ma.masked_invalid(B2_T * 1e3)
    A3 = np.ma.masked_invalid(Bsplit_T * 1e3)

    v3min, v3max = (clim3_mT if clim3_mT is not None else (None, None))

    p_lo, p_hi = 1, 99
    stack = np.concatenate([A1.ravel(), A2.ravel()])
    vmin, vmax = np.nanpercentile(stack, [p_lo, p_hi])

    fig, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    if suptitle:
        fig.suptitle(suptitle, fontsize=12)

    im1 = axes[0].imshow(A1, origin="lower", interpolation="nearest", extent=extent, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[0].set_title(r"$B_{\parallel,1}$ (mT)  = $(\nu_{\rm high} - D)/\gamma_{\rm NV}$")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    axes[0].invert_yaxis()
    fig.colorbar(im1, ax=axes[0], label="mT")

    im2 = axes[1].imshow(A2, origin="lower", interpolation="nearest", extent=extent, cmap="plasma", vmin=vmin, vmax=vmax)
    axes[1].set_title(r"$B_{\parallel,2}$ (mT)  = $(\nu_{\rm low} - D)/\gamma_{\rm NV}$")
    axes[1].set_xlabel("j"); axes[1].set_ylabel("i")
    axes[1].invert_yaxis()
    fig.colorbar(im2, ax=axes[1], label="mT")
    im2.set_norm(im1.norm)

    im3 = axes[2].imshow(A3, origin="lower", interpolation="nearest", extent=extent, cmap=cmap, vmin=v3min, vmax=v3max)
    axes[2].set_title(r"$\Delta B_{\parallel}$ (mT) $= (\nu_{\rm high} - \nu_{\rm low})/\gamma_{\rm NV}$")
    axes[2].set_xlabel("j"); axes[2].set_ylabel("i")
    axes[2].invert_yaxis()
    fig.colorbar(im3, ax=axes[2], label="mT")

    # --- ROI boxes (multi) ---
    for ax in axes:
        _add_roi_boxes(ax, roi_box=roi_box, roi_boxes=roi_boxes, default_lw=2.2)

    # ---- scalebar ----
    if scalebar and (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0
        for ax in axes:
            ann1 = ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                               xycoords="axes fraction",
                               arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                               annotation_clip=False)
            ann1.set_zorder(10000)

            ann2 = ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                               xycoords="axes fraction",
                               arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
                               annotation_clip=False)
            ann2.set_zorder(10001)

            val, unit = format_length_um(bar_um)
            txt = ax.text(x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
                          transform=ax.transAxes, color="w",
                          ha="center", va="bottom", fontsize=9,
                          bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5))
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()




####

def plot_contrast_DeltaB_projectionNV(
    df, NORMALIZE, SMOOTH_WIN,
    roi, Bsplit_T_roi, extent_roi,
    roi_box=None,
    use_processed=True,
    cmap_contrast="viridis",
    cmap_split_name="turbo",
    max_ticks=8,      # conservé pour compat, non utilisé car on force pas=10
    grid=True,
    figsize=(12, 6),
    count_brut=False,
    vmin=None,
    vmax=None,
    png_path=None,
):
    """
    Affiche côte à côte :
      - Contraste (max - min sur k) depuis df dans la fenêtre `roi`
      - ΔB_∥ (mT) fourni (Bsplit_T_roi)
    """

    # --- Ranges issus du ROI ---
    slice_i, slice_j = roi
    i0, i1 = slice_i.start, slice_i.stop - 1
    j0, j1 = slice_j.start, slice_j.stop - 1
    i_range = np.arange(i0, i1 + 1)
    j_range = np.arange(j0, j1 + 1)

    # --- Vérifs tailles ---
    expected_shape = (len(i_range), len(j_range))
    if Bsplit_T_roi.shape != expected_shape:
        raise ValueError(f"Bsplit_T_roi.shape={Bsplit_T_roi.shape} != {expected_shape}")

    # --- Sous-DF ROI ---
    mask_roi = (df["i"].between(i0, i1)) & (df["j"].between(j0, j1))
    df_roi = df.loc[mask_roi, ["i","j","k","sig"]].copy().sort_values(["i","j","k"])
    if df_roi.empty:
        raise ValueError("Aucune donnée dans df pour la fenêtre ROI.")

    # --- Prétraitements optionnels si dispos ---
    def _maybe_normalize(y, NORMALIZE):
        if use_processed:
            try:    return maybe_normalize(y, NORMALIZE)
            except NameError: pass
        return y
    def _maybe_smooth(y, SMOOTH_WIN):
        if use_processed:
            try:    return maybe_smooth(y, SMOOTH_WIN)
            except NameError: pass
        return y
    def _series_preproc(s_counts):
        y = s_counts.to_numpy(dtype=float)
        y = _maybe_normalize(y, NORMALIZE)
        y = _maybe_smooth(y, SMOOTH_WIN)
        return y

    # --- Contraste max-min sur k ---
    def _agg_contrast(s_counts):
        y = _series_preproc(s_counts)
        return float(np.nanmax(y) - np.nanmin(y))

    contrast_df = (
        df_roi.groupby(["i","j"])["sig"].agg(_agg_contrast)
              .unstack("j")
              .reindex(index=i_range, columns=j_range)
    )
    if contrast_df.isna().all().all():
        raise ValueError("Contraste entièrement NaN sur le ROI.")
    if count_brut:
        contrast_df = (
                df_roi.groupby(["i","j"])["sig"].agg(lambda s: float(s.max() - s.min()))
                    .unstack("j")
                    .reindex(index=i_range, columns=j_range)
        )

    # --- ΔB_∥ en mT + masque NaN ---
    cmap_split = plt.cm.get_cmap(cmap_split_name).copy()
    cmap_split.set_bad(alpha=0.0)
    A_split_mT = np.ma.masked_invalid(Bsplit_T_roi * 1e3)

    # --- Figure : 2 panneaux carrés ---
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    # 1) Contraste
    im0 = axes[0].imshow(
        contrast_df.to_numpy(),
        origin="lower",
        interpolation="nearest",
        extent=extent_roi,
        cmap=cmap_contrast,
    )
    axes[0].set_title("Contraste (max - min)")
    axes[0].set_xlabel("j"); axes[0].set_ylabel("i")
    axes[0].invert_yaxis()
    fig.colorbar(im0, ax=axes[0], label="contrast intensity")
    if roi_box is not None:
        _add_roi_box(axes[0], roi_box)

    # 2) ΔB_∥
    im1 = axes[1].imshow(
        A_split_mT,
        origin="lower",
        interpolation="nearest",
        extent=extent_roi,
        cmap=cmap_split,
        vmin=vmin,
        vmax=vmax
    )
    axes[1].set_title(r"$\Delta B_{\parallel}$ (mT) $= B_{\parallel,1}-B_{\parallel,2} = \Delta f / \gamma_{\rm NV}$")
    axes[1].set_xlabel("j"); axes[1].set_ylabel("i")
    axes[1].invert_yaxis()
    fig.colorbar(im1, ax=axes[1], label="mT")
    if roi_box is not None:
        _add_roi_box(axes[1], roi_box)

    # --- Pixels & panneaux carrés ---
    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        try:
            ax.set_box_aspect(1)
        except Exception:
            pass

    # --- Ticks forcés tous les 10 sur X et Y + mineurs tous les 1 ---
    major_step = 10
    for ax in axes:
        ax.xaxis.set_major_locator(mticker.MultipleLocator(major_step))
        ax.yaxis.set_major_locator(mticker.MultipleLocator(major_step))
        # ax.xaxis.set_minor_locator(mticker.MultipleLocator(1))
        # ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))
        # ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.tick_params(axis="both", which="major", labelsize=9)

    # --- Grille optionnelle ---
    if grid:
        for ax in axes:
            ax.grid(which="minor", linestyle=":", linewidth=0.4, alpha=0.25)
            ax.grid(which="major", linestyle="-", linewidth=0.5, alpha=0.4)
            
    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()





#***** RECONSTRUCTION CHAMP MAGNETIQUE B **************************************

def nv_axis_from_angles(theta_deg, phi_deg):
    """Unit NV axis for polar theta and azimuth phi, expressed in degrees."""
    theta, phi = np.deg2rad([theta_deg, phi_deg])
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def reconstruct_B_from_Bparallel_Fourier(
    Bpar_T,
    nv_axis,
    dx,
    d_meas,
    lambda0=None,
    p=10,
    alpha=1.0,
    dc_mode="projection",
):
    """
    Reconstruit Bx, By, Bz à partir d'une carte B_par(x,y) (Tesla)
    pour un seul axe NV, via inversion en k-espace (Tikhonov spectral).

    Bpar_T : array (NY, NX), Tesla
    nv_axis : [ux, uy, uz] (pas besoin d'être normalisé)
    dx : pitch spatial (m)
    d_meas : distance capteur-échantillon (m)
    lambda0 : float ou None (None correspond à 1e-8, sans facteur médian)
    p : exposant dans lambda(k) = lambda0 (1 + (k/k0)^p)
    alpha : k0 = alpha / d_meas
    dc_mode : "zero" pour les fluctuations, "projection" pour la convention
        historique Bx0=By0=0, Bz0=mean(Bpar)/uz. Cette convention ne mesure
        pas le champ uniforme physique.
    """

    Bpar = np.asarray(Bpar_T, dtype=float)
    NY, NX = Bpar.shape

    # On met les NaN à 0 pour la FFT, mais on peut les remasquer ensuite si besoin
    mask_valid = np.isfinite(Bpar)
    Bpar = np.where(mask_valid, Bpar, 0.0)

    # --- FFT 2D ---
    Bpar_hat = np.fft.fft2(Bpar)

    kx = 2*np.pi*np.fft.fftfreq(NX, d=dx)   # [rad/m]
    ky = 2*np.pi*np.fft.fftfreq(NY, d=dx)   # [rad/m]
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    K = np.sqrt(KX**2 + KY**2)
    K_safe = K.copy()
    K_safe[K_safe == 0] = 1.0

    # --- axe NV ---
    u = np.array(nv_axis, dtype=float)
    u /= np.linalg.norm(u)
    ux, uy, uz = u

    # --- opérateur direct : B_par(k) = D(k) B_z(k)
    #     D(k) = u_z - i (u_x kx + u_y ky)/k
    D = uz - 1j * (ux * KX + uy * KY) / K_safe
    D2 = np.abs(D)**2

    # --- régularisation ---
    if lambda0 is None:
        lambda0 = 1e-8
    if lambda0 < 0 or alpha <= 0 or dx <= 0 or d_meas <= 0 or p < 0:
        raise ValueError("Fourier parameters require lambda0>=0, p>=0, alpha, dx, d_meas>0.")
    if dc_mode not in ("zero", "projection"):
        raise ValueError("dc_mode must be 'zero' or 'projection'.")

    k0 = alpha / float(d_meas)
    lam_k = lambda0 * (1.0 + (K / k0)**p)

    # --- inversion ---
    denominator = D2 + lam_k
    Bz_hat = np.divide(np.conj(D) * Bpar_hat, denominator,
                       out=np.zeros_like(Bpar_hat), where=denominator > 0)
    if dc_mode == "zero":
        Bz_hat[0, 0] = 0.0
    elif abs(uz) > 1e-10:
        Bz_hat[0, 0] = Bpar_hat[0, 0] / uz
    else:
        raise ValueError("The projection DC convention requires a nonzero NV z component.")

    # --- relations de Maxwell ---
    Bx_hat = -1j * (KX / K_safe) * Bz_hat
    By_hat = -1j * (KY / K_safe) * Bz_hat

    # --- retour espace direct ---
    Bx_rec = np.real(np.fft.ifft2(Bx_hat))
    By_rec = np.real(np.fft.ifft2(By_hat))
    Bz_rec = np.real(np.fft.ifft2(Bz_hat))

    # Option : remasquer là où Bpar était invalide
    Bx_rec = np.where(mask_valid, Bx_rec, np.nan)
    By_rec = np.where(mask_valid, By_rec, np.nan)
    Bz_rec = np.where(mask_valid, Bz_rec, np.nan)

    return Bx_rec, By_rec, Bz_rec


def nv_axis_in_image_frame(theta_deg, phi_deg, axis_signs=(-1, 1, 1)):
    """Express the instrument NV vector in the displayed scan coordinates.

    For the axis-aligned FeGa scans, displayed x increases to the right and
    displayed y downwards. Instrument screenshots indicate X decreasing to
    the right and Y increasing downwards, hence (-1,+1,+1). This mapping is
    conditional on that instrument convention; export files contain no valid
    per-pixel X/Y references. It applies to vectors, not to scalar seeds.
    """
    signs = np.asarray(axis_signs, dtype=float)
    if signs.shape != (3,) or not np.all(np.isin(signs, [-1, 1])):
        raise ValueError("axis_signs must contain three signs (+1 or -1).")
    return signs * nv_axis_from_angles(theta_deg, phi_deg)



def quiver_data_from_Bfield(
    Bx_T,
    By_T,
    Bz_T,
    extent,
    step=3,
    with_norm_2D=True,
    color_mode="Bz",  # "Bz" ou "Bmag"
):
    """
    Prépare X, Y, Ux, Uy, C pour un quiver 2D à partir d'un champ (Bx, By, Bz).

    Bx_T, By_T, Bz_T : (NY, NX), en Tesla
    extent : [x_min, x_max, y_min, y_max] (en µm, comme tes imshow)
    step : sous-échantillonnage des flèches
    with_norm_2D : True => toutes les flèches ont la même longueur
    color_mode : "Bz" (couleur = Bz) ou "Bmag" (couleur = |B|)
    """

    Bx = np.asarray(Bx_T, dtype=float)
    By = np.asarray(By_T, dtype=float)
    Bz = np.asarray(Bz_T, dtype=float)
    NY, NX = Bx.shape

    x_min, x_max, y_min, y_max = extent

    dx_phys = (x_max - x_min) / NX
    dy_phys = (y_max - y_min) / NY
    x_centers = x_min + (np.arange(NX) + 0.5) * dx_phys
    y_centers = y_min + (np.arange(NY) + 0.5) * dy_phys
    X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")

    ys = np.arange(0, NY, step)
    xs = np.arange(0, NX, step)

    Xs = X[ys][:, xs]
    Ys = Y[ys][:, xs]
    Ux = Bx[ys][:, xs]
    Uy = By[ys][:, xs]
    Bz_sub = Bz[ys][:, xs]

    if color_mode == "Bmag":
        C_mT = np.sqrt(Ux**2 + Uy**2 + Bz_sub**2) * 1e3
    else:  # "Bz"
        C_mT = Bz_sub * 1e3

    if with_norm_2D:
        den = np.hypot(Ux, Uy)
        den = np.where(den == 0, 1, den)
        Ux_plot = Ux / den
        Uy_plot = Uy / den
    else:
        Ux_plot = Ux
        Uy_plot = Uy

    mask = ~np.isfinite(C_mT)
    C_mT = np.ma.masked_array(C_mT, mask=mask)
    Ux_plot = np.ma.masked_array(Ux_plot, mask=mask)
    Uy_plot = np.ma.masked_array(Uy_plot, mask=mask)

    return Xs, Ys, Ux_plot, Uy_plot, C_mT


#Ok cette fonction n'est peut être pas clean pour tous les arguments, j'ai effectué des corrections à certains endroits et ai peut être oublié de corriger à d'autres.
def reconstruction_B(
    B1_T,
    B2_T,
    nv_axis,
    dx,
    d_meas,
    extent,
    step=3,
    with_norm_2D=True,
    color_mode="Bz",   # "Bz" ou "Bmag"
    cmap_name="turbo",
    dark_bg=False,     # fond noir ou non
    colorbar_whitebg=None,
    diff_mode="quiver",  # "quiver", "norm", ou "Bpar"
    Bext=None,
    clim3_mT=None,
    png_path=None,
    s=1,
    scalebar=True,
    Size_um=None,
    scalebar_frac=0.25,
    scalebar_xy=(0.05, 0.05)
):
    """
    Reconstruit le champ 3D à partir de deux cartes B_parallel (B1, B2)
    pour un seul axe NV, puis affiche un triptyque :

      [B reconstruit depuis B1]  [B reconstruit depuis B2]  [erreur (mode choisi)]

    diff_mode :
      - "quiver" : 3e subplot = quiver de B1 - B2
      - "norm"   : 3e subplot = carte 2D de |B1 - B2| (mT), cmap = cmap_name
      - "Bpar"   : 3e subplot = carte 2D de B_par,1 - B_par,2 (mT) avec B_par = B·u_NV
    """

    # normalise l'axe NV pour la projection B_par si besoin
    u = np.asarray(nv_axis, dtype=float)
    u /= np.linalg.norm(u)
    ux, uy, uz = u

    # --- 1) reconstruction pour B1 et B2 ---
    Bx1, By1, Bz1 = reconstruct_B_from_Bparallel_Fourier(
        B1_T, nv_axis, dx, d_meas
    )
    Bx2, By2, Bz2 = reconstruct_B_from_Bparallel_Fourier(
        B2_T, nv_axis, dx, d_meas
    )

    if Bext is not None:
        # Bext est la norme du champ de biais (en Tesla) le long de nv_axis
        u2 = np.array(nv_axis, dtype=float)
        u2 /= np.linalg.norm(u2)
        bx0, by0, bz0 = Bext * u2   # vecteur 3D du champ de biais

        Bx1 = Bx1 - bx0
        By1 = By1 - by0
        Bz1 = Bz1 - bz0

        Bx2 = Bx2 - bx0
        By2 = By2 - by0
        Bz2 = Bz2 - bz0

    # --- 2) différences ---
    BxD = Bx1 - Bx2
    ByD = By1 - By2
    BzD = Bz1 - Bz2

    # norme de la différence (pour diff_mode="norm")
    BmagD_mT = np.sqrt(BxD**2 + ByD**2 + BzD**2) * 1e3

    # projection NV (pour diff_mode="Bpar")
    Bpar1_mT = (Bx1*ux + By1*uy + Bz1*uz) * 1e3
    Bpar2_mT = (Bx2*ux + By2*uy + Bz2*uz) * 1e3
    BparD_mT = (Bpar1_mT - Bpar2_mT)

    def _vv(local):
        if local is not None:
            return local
        return clim3_mT if clim3_mT is not None else (None, None)

    v3min, v3max = _vv(clim3_mT)

    # --- 3) données pour quiver (B1, B2, et éventuellement diff) ---
    # (utilisé pour le mode dark_bg ; on garde ton code existant)
    X1, Y1, Ux1, Uy1, C1 = quiver_data_from_Bfield(
        Bx1, By1, Bz1, extent,
        step=step, with_norm_2D=with_norm_2D, color_mode=color_mode
    )
    X2, Y2, Ux2, Uy2, C2 = quiver_data_from_Bfield(
        Bx2, By2, Bz2, extent,
        step=step, with_norm_2D=with_norm_2D, color_mode=color_mode
    )

    if diff_mode == "quiver":
        XD, YD, UxD, UyD, CD = quiver_data_from_Bfield(
            BxD, ByD, BzD, extent,
            step=step, with_norm_2D=with_norm_2D, color_mode=color_mode
        )

    # --- 4a) échelle de couleurs pour B1 et B2 (mode dark_bg) ---
    vals12 = np.ma.concatenate([C1.compressed(), C2.compressed()])
    if vals12.size:
        vmin12, vmax12 = np.percentile(vals12, [5, 95])
    else:
        vmin12, vmax12 = None, None

    cmap12 = cm.get_cmap(cmap_name)
    norm12 = mcolors.Normalize(vmin=vmin12, vmax=vmax12)

    # --- 4b) échelle de couleurs / colormap pour la différence (mode dark_bg) ---
    if diff_mode == "norm":
        valsD = BmagD_mT[np.isfinite(BmagD_mT)]
        if valsD.size:
            vminD = 0.0
            vmaxD = np.percentile(valsD, 95)
        else:
            vminD = vmaxD = None
        cmapD = cm.get_cmap(cmap_name)  # même cmap que pour B1/B2
        normD = mcolors.Normalize(vmin=vminD, vmax=vmaxD)

    elif diff_mode == "Bpar":
        valsD = np.abs(BparD_mT[np.isfinite(BparD_mT)])                     # ABS ??????
        if valsD.size:
            vminD = 0.0 #np.percentile(valsD, 2) #0.0
            vmaxD = v3max #np.percentile(valsD, 98)
        else:
            vminD = vmaxD = None
        cmapD = cm.get_cmap("turbo")
        normD = mcolors.Normalize(vmin=vminD, vmax=vmaxD)

    else:  # "quiver"
        valsD = CD.compressed()
        if valsD.size:
            max_abs = np.percentile(np.abs(valsD), 95)
            vminD, vmaxD = -max_abs, max_abs
        else:
            vminD = vmaxD = None
        cmapD = cm.get_cmap("bwr")
        normD = mcolors.Normalize(vmin=vminD, vmax=vmaxD)

    # --- 5) figure / triptyque ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    # =======================
    # CAS 1 : fond noir (ton style original)
    # =======================
    if dark_bg:
        fig.patch.set_facecolor("black")
        for ax in axes:
            ax.set_facecolor("black")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.title.set_color("white")

        # 1) depuis B1
        axes[0].quiver(
            X1, Y1, Ux1, Uy1, C1,
            cmap=cmap12, norm=norm12,
            angles="xy", scale_units="xy", scale=0.5, width=0.01,
            pivot="mid"
        )
        axes[0].set_aspect("equal")
        axes[0].set_xlim(extent[0], extent[1])
        axes[0].set_ylim(extent[2], extent[3])
        axes[0].set_xlabel("i")
        axes[0].set_ylabel("j")
        axes[0].set_title(r"Reconstruction $\vec{B}$ depuis $B_{\parallel,1}$")
        axes[0].invert_yaxis()

        # 2) depuis B2
        axes[1].quiver(
            X2, Y2, Ux2, Uy2, C2,
            cmap=cmap12, norm=norm12,
            angles="xy", scale_units="xy", scale=0.5, width=0.01,
            pivot="mid"
        )
        axes[1].set_aspect("equal")
        axes[1].set_xlim(extent[0], extent[1])
        axes[1].set_ylim(extent[2], extent[3])
        axes[1].set_xlabel("i")
        axes[1].set_ylabel("j")
        axes[1].set_title(r"Reconstruction $\vec{B}$ depuis $B_{\parallel,2}$")
        axes[1].invert_yaxis()

        # 3) différence
        if diff_mode == "quiver":
            axes[2].quiver(
                XD, YD, UxD, UyD, CD,
                cmap=cmapD, norm=normD,
                angles="xy", scale_units="xy", scale=0.5, width=0.01,
                pivot="mid"
            )
            axes[2].set_aspect("equal")
            axes[2].set_xlim(extent[0], extent[1])
            axes[2].set_ylim(extent[2], extent[3])
            axes[2].set_xlabel("i")
            axes[2].set_ylabel("j")
            axes[2].set_title(r"Différence $\vec{B_1}-\vec{B_2}$")
            axes[2].invert_yaxis()

        elif diff_mode == "norm":
            imD = axes[2].imshow(
                BmagD_mT,
                origin="lower",
                extent=extent,
                cmap=cmapD,
                vmin=vminD, vmax=vmaxD,
                interpolation="nearest"
            )
            axes[2].set_aspect("equal")
            axes[2].set_xlim(extent[0], extent[1])
            axes[2].set_ylim(extent[2], extent[3])
            axes[2].set_xlabel("i")
            axes[2].set_ylabel("j")
            axes[2].set_title(r"$|\vec{B_1}-\vec{B_2}|$ (mT)")
            axes[2].invert_yaxis()

        else:  # "Bpar"
            imD = axes[2].imshow(
                (BparD_mT),               # ABS ??
                origin="lower",
                extent=extent,
                cmap=cmapD,
                vmin=vminD, vmax=vmaxD,
                interpolation="nearest"
            )
            axes[2].set_aspect("equal")
            axes[2].set_xlim(extent[0], extent[1])
            axes[2].set_ylim(extent[2], extent[3])
            axes[2].set_xlabel("i")
            axes[2].set_ylabel("j")
            axes[2].set_title(r"$\Delta B_{\parallel}^{\rm (reconstruit)}$ (mT)")
            axes[2].invert_yaxis()

        # Colorbar pour B1/B2
        if color_mode == "Bmag":
            label12 = r"$|\mathbf{B}|$ (mT)"
        else:
            label12 = r"$B_z$ (mT)"

        cbar12 = fig.colorbar(
            cm.ScalarMappable(norm=norm12, cmap=cmap12),
            ax=axes[:2],
            shrink=0.85
        )
        cbar12.set_label(label12)
        cbar12.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar12.ax.get_yticklabels(), color="white")

        # Colorbar pour la différence
        if diff_mode == "quiver":
            if color_mode == "Bmag":
                labelD = r"$\Delta|\mathbf{B}|$ (mT)"
            else:
                labelD = r"$\Delta B_z$ (mT)"
            mappableD = cm.ScalarMappable(norm=normD, cmap=cmapD)

        elif diff_mode == "norm":
            labelD = r"$|\vec{B_1}-\vec{B_2}|$ (mT)"
            mappableD = cm.ScalarMappable(norm=normD, cmap=cmapD)

        else:  # "Bpar"
            labelD = r"$\Delta B_{\parallel}$ (mT)"
            mappableD = cm.ScalarMappable(norm=normD, cmap=cmapD)

        cbarD = fig.colorbar(
            mappableD,
            ax=axes[2],
            shrink=0.85
        )
        cbarD.set_label(labelD)
        cbarD.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbarD.ax.get_yticklabels(), color="white")

    # =======================
    # CAS 2 : fond blanc (bwr + flèches noires)
    # =======================
    else:
        # Bz en mT pour les fonds
        Bz1_mT = Bz1 * 1e3
        Bz2_mT = Bz2 * 1e3
        BzD_mT = BzD * 1e3

        # échelle commune symétrique pour B1/B2
        vals_z12 = np.concatenate([
            Bz1_mT[np.isfinite(Bz1_mT)].ravel(),
            Bz2_mT[np.isfinite(Bz2_mT)].ravel(),
        ])
        if vals_z12.size:
            #max_abs12 = np.max(np.abs(vals_z12))
            clip = 99.4 #98  # ou 98 si tu veux couper plus fort
            max_abs12 = np.nanpercentile(np.abs(vals_z12), clip)
        else:
            max_abs12 = 1.0
        vmin_z12, vmax_z12 = -max_abs12, max_abs12
        cmap_z = cm.get_cmap("bwr") if colorbar_whitebg is None else cm.get_cmap(colorbar_whitebg)
        if colorbar_whitebg is not None:
            vmin_z12, vmax_z12 = np.nanpercentile(vals_z12, 100-clip), np.nanpercentile(vals_z12, clip) #-max_abs12, max_abs12
        norm_z = mcolors.Normalize(vmin=vmin_z12, vmax=vmax_z12)

        x_min, x_max, y_min, y_max = extent
        NY, NX = Bz1.shape
        dx_phys = (x_max - x_min) / NX
        dy_phys = (y_max - y_min) / NY
        x_centers = x_min + (np.arange(NX) + 0.5) * dx_phys
        y_centers = y_min + (np.arange(NY) + 0.5) * dy_phys
        Xfull, Yfull = np.meshgrid(x_centers, y_centers, indexing="xy")

        # --- Panel 1 : B1 ---
        im1 = axes[0].imshow(
            Bz1_mT,
            origin="lower",
            extent=extent,
            cmap=cmap_z,
            vmin=vmin_z12,
            vmax=vmax_z12,
            interpolation="nearest",
        )
 
        # sous-échantillonnage pour les flèches (comme l'autre code)
        Xq = Xfull[::step, ::step]
        Yq = Yfull[::step, ::step]
        U1 = Bx1[::step, ::step]
        V1 = By1[::step, ::step]
        axes[0].quiver(Xq, Yq, U1, V1, color="k", scale=s)
        axes[0].set_aspect("equal")
        axes[0].set_xlim(extent[0], extent[1])
        axes[0].set_ylim(extent[2], extent[3])
        axes[0].set_xlabel("i")
        axes[0].set_ylabel("j")
        axes[0].set_title(r"Reconstruction $\vec{B_1}$ depuis $B_{\parallel,1}$")
        axes[0].invert_yaxis()

        # --- Panel 2 : B2 ---
        axes[1].imshow(
            Bz2_mT,
            origin="lower",
            extent=extent,
            cmap=cmap_z,
            vmin=vmin_z12,
            vmax=vmax_z12,
            interpolation="nearest",
        )
    
        U2 = Bx2[::step, ::step]
        V2 = By2[::step, ::step]
        axes[1].quiver(Xq, Yq, U2, V2, color="k", scale=s)
        axes[1].set_aspect("equal")
        axes[1].set_xlim(extent[0], extent[1])
        axes[1].set_ylim(extent[2], extent[3])
        axes[1].set_xlabel("i")
        axes[1].set_ylabel("j")
        axes[1].set_title(r"Reconstruction $\vec{B_2}$ depuis $B_{\parallel,2}$")
        axes[1].invert_yaxis()

        # colorbar commune pour Bz1/Bz2
        mappable12 = cm.ScalarMappable(norm=norm_z, cmap=cmap_z)
        mappable12.set_array([])
        cbar12 = fig.colorbar(mappable12, ax=axes[:2], shrink=0.85)
        cbar12.set_label("Bz (mT)")

        # --- Panel 3 : différence ---
        if diff_mode == "quiver":
            # échelle symétrique pour BzD
            vals_zD = BzD_mT[np.isfinite(BzD_mT)]
            if vals_zD.size:
                #max_absD = np.max(np.abs(vals_zD))
                clip = 98
                max_absD = np.nanpercentile(np.abs(vals_zD), clip)    #ABS
            else:
                max_absD = 1.0
            vmin_zD, vmax_zD = -max_absD, max_absD
            norm_zD = mcolors.Normalize(vmin=vmin_zD, vmax=vmax_zD)

            imD = axes[2].imshow(
                BzD_mT,
                origin="lower",
                extent=extent,
                cmap=cmap_z,
                vmin=vmin_zD,
                vmax=vmax_zD,
                interpolation="nearest",
            )
 
            UD = BxD[::step, ::step]
            VD = ByD[::step, ::step]
            axes[2].quiver(Xq, Yq, UD, VD, color="k", scale=s)
            axes[2].set_aspect("equal")
            axes[2].set_xlim(extent[0], extent[1])
            axes[2].set_ylim(extent[2], extent[3])
            axes[2].set_xlabel("i")
            axes[2].set_ylabel("j")
            axes[2].set_title(r"Différence $\vec{B_1}-\vec{B_2}$")
            axes[2].invert_yaxis()

            mappableD = cm.ScalarMappable(norm=norm_zD, cmap=cmap_z)
            mappableD.set_array([])
            cbarD = fig.colorbar(mappableD, ax=axes[2], shrink=0.85)
            cbarD.set_label("Bz diff (mT)")

        elif diff_mode == "norm":
            imD = axes[2].imshow(
                BmagD_mT,
                origin="lower",
                extent=extent,
                cmap=cmapD,
                vmin=vminD, vmax=vmaxD,
                interpolation="nearest"
            )
            axes[2].set_aspect("equal")
            axes[2].set_xlim(extent[0], extent[1])
            axes[2].set_ylim(extent[2], extent[3])
            axes[2].set_xlabel("i")
            axes[2].set_ylabel("j")
            axes[2].set_title(r"$|\vec{B_1}-\vec{B_2}|$ (mT)")
            axes[2].invert_yaxis()

            mappableD = cm.ScalarMappable(norm=normD, cmap=cmapD)
            mappableD.set_array([])
            cbarD = fig.colorbar(mappableD, ax=axes[2], shrink=0.85)
            cbarD.set_label(r"$|\vec{B_1}-\vec{B_2}|$ (mT)")

        else:  # "Bpar"
            imD = axes[2].imshow(
                (BparD_mT),           #ABS
                origin="lower",
                extent=extent,
                cmap=cmapD,
                vmin=vminD, vmax=v3max,# vmaxD,
                interpolation="nearest"
            )
            axes[2].set_aspect("equal")
            axes[2].set_xlim(extent[0], extent[1])
            axes[2].set_ylim(extent[2], extent[3])
            axes[2].set_xlabel("i")
            axes[2].set_ylabel("j")
            axes[2].set_title(r"$\Delta B_{\parallel}^{\rm (reconstruit)}=(\vec{B_1}-\vec{B_2})\cdot\vec{u}_{NV}$ (mT)")
            axes[2].invert_yaxis()

            mappableD = cm.ScalarMappable(norm=normD, cmap=cmapD)
            mappableD.set_array([])
            cbarD = fig.colorbar(mappableD, ax=axes[2], shrink=0.85)
            cbarD.set_label(r"$\Delta B_{\parallel}=(\vec{B_1}-\vec{B_2})\cdot\vec{u}_{NV}$ (mT)")

            # ---- BARRE D'ÉCHELLE (noir + pointillé blanc) ----
    if scalebar and (Size_um is not None):
        bar_um  = scalebar_frac * Size_um
        x0, y0  = scalebar_xy
        x1, y1  = x0 + scalebar_frac, y0
        for ax in axes:
            ann1 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=2, color="w", linestyle=(0, (4, 4))),
                annotation_clip=False
            )
            ann2.set_zorder(10001)

            val, unit = format_length_um(bar_um)
            txt = ax.text(
                x0 + scalebar_frac/2, y0 + 0.03, f"{val:.0f} {unit}",
                transform=ax.transAxes, color="w",
                ha="center", va="bottom", fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5)
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)
    if png_path is not None:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.show()

    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

    return (Bx1, By1, Bz1), (Bx2, By2, Bz2), (BxD, ByD, BzD)


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm


def plot_reconstructed_B_grid(
    B_list_T,
    nv_axis,
    dx,
    d_meas,
    extent,
    nrows,
    ncols,
    *,
    step=4,
    quiver_scale=0.10,
    headwidth=2,
    width=0.004,
    Bext_remove=None,
    Bext_list=None,
    Bext_unit="mT",
    titles=None,
    cmap_name="bwr",
    n_color_levels=30,
    clip_percentile=99.4,
    color_scale_mode="minmax",   # "minmax" ou "symmetric"
    figsize_per_panel=(3.0, 3.0),
    fig_dpi=120,
    save_dpi=300,
    tick_step=20,
    origin="lower",
    interpolation="nearest",
    invert_y=True,
    scalebar=False,
    Size_um=None,
    scalebar_frac=0.25,
    scalebar_xy=(0.05, 0.05),
    scalebar_indices=(0,),
    png_path=None,
    pdf_path=None,
):
    """
    Affiche une grille de reconstructions B à partir d'une liste de cartes B_parallel.

    Chaque panneau montre :
      - fond : Bz (mT)
      - quiver : (Bx, By)

    Paramètres importants
    ---------------------
    B_list_T : list[np.ndarray]
        Liste des cartes B_parallel en Tesla.
    nv_axis : array-like (3,)
        Axe NV.
    dx : float
        Pas pixel pour la reconstruction.
    d_meas : float
        Distance de mesure.
    extent : [x_min, x_max, y_min, y_max]
        Extent matplotlib.
    nrows, ncols : int
        Dimensions de la grille.

    Options utiles
    --------------
    color_scale_mode :
        - "minmax"    : vmin/vmax pris sur les percentiles globaux des valeurs de Bz
        - "symmetric" : échelle symétrique autour de 0 avec max_abs global
    Bext_remove : float or None
        Champ externe (Tesla) à soustraire après reconstruction le long de nv_axis.
    Bext_list : list or None
        Valeurs affichées dans les titres sous la forme Bext=...
    titles : list[str] or None
        Si fourni, remplace les titres automatiques.
    fig_dpi : int
        DPI de la figure affichée à l'écran.
    save_dpi : int
        DPI utilisé pour les sauvegardes.
    """

    def _format_length_um(length_um):
        if length_um >= 1000:
            return length_um / 1000, "mm"
        elif length_um >= 1:
            return length_um, "µm"
        else:
            return length_um * 1000, "nm"

    n_maps = len(B_list_T)
    n_slots = nrows * ncols

    if n_maps == 0:
        raise ValueError("B_list_T est vide.")
    if n_maps > n_slots:
        raise ValueError(
            f"{n_maps} cartes pour seulement {n_slots} sous-figures ({nrows}x{ncols})."
        )

    u = np.asarray(nv_axis, dtype=float)
    u /= np.linalg.norm(u)

    # ---------- reconstruction ----------
    results = []
    for Bpar_T in B_list_T:
        Bx, By, Bz = reconstruct_B_from_Bparallel_Fourier(Bpar_T, u, dx, d_meas)

        if Bext_remove is not None:
            bx0, by0, bz0 = Bext_remove * u
            Bx = Bx - bx0
            By = By - by0
            Bz = Bz - bz0

        results.append((Bx, By, Bz))

    # ---------- échelle commune Bz ----------
    """
    all_Bz_mT = np.concatenate([
        (Bz * 1e3)[np.isfinite(Bz)].ravel()
        for (_, _, Bz) in results
    ])

    if color_scale_mode == "minmax":
        if all_Bz_mT.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            lower_pct = 100 - clip_percentile
            upper_pct = clip_percentile
            vmin = np.nanpercentile(all_Bz_mT, lower_pct)
            vmax = np.nanpercentile(all_Bz_mT, upper_pct)

            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = -1.0, 1.0

    elif color_scale_mode == "symmetric":
        if all_Bz_mT.size == 0:
            max_abs = 1.0
        else:
            max_abs = np.nanpercentile(np.abs(all_Bz_mT), clip_percentile)
            if (not np.isfinite(max_abs)) or (max_abs == 0):
                max_abs = 1.0
        vmin, vmax = -max_abs, max_abs

    else:
        raise ValueError("color_scale_mode doit être 'minmax' ou 'symmetric'.")

    # --- ancienne version symétrique si tu veux revenir en arrière ---
    # if all_Bz_mT.size == 0:
    #     max_abs = 1.0
    # else:
    #     max_abs = np.nanpercentile(np.abs(all_Bz_mT), clip_percentile)
    #     if (not np.isfinite(max_abs)) or (max_abs == 0):
    #         max_abs = 1.0
    # vmin, vmax = -max_abs, max_abs

    cmap = cm.get_cmap(cmap_name, n_color_levels)
    boundaries = np.linspace(vmin, vmax, n_color_levels + 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    """
    cmap = cm.get_cmap(cmap_name, n_color_levels)

    # ---------- grille spatiale ----------
    x_min, x_max, y_min, y_max = extent
    _, _, Bz0 = results[0]
    ny, nx = Bz0.shape

    dx_phys = (x_max - x_min) / nx
    dy_phys = (y_max - y_min) / ny

    x_centers = x_min + (np.arange(nx) + 0.5) * dx_phys
    y_centers = y_min + (np.arange(ny) + 0.5) * dy_phys
    Xfull, Yfull = np.meshgrid(x_centers, y_centers, indexing="xy")

    Xq = Xfull[::step, ::step]
    Yq = Yfull[::step, ::step]

    xticks = np.arange(
        tick_step * np.ceil(x_min / tick_step),
        x_max + 1e-9,
        tick_step
    )
    yticks = np.arange(
        tick_step * np.ceil(y_min / tick_step),
        y_max + 1e-9,
        tick_step
    )

    # ---------- figure ----------
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=fig_dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for idx, ax in enumerate(flat_axes):
        if idx >= n_maps:
            ax.axis("off")
            continue

        Bx, By, Bz = results[idx]
        Bz_mT = Bz * 1e3

        Bz_valid_mT = Bz_mT[np.isfinite(Bz_mT)]

        if color_scale_mode == "minmax":
            if Bz_valid_mT.size == 0:
                vmin_i, vmax_i = -1.0, 1.0
            else:
                lower_pct = 100 - clip_percentile
                upper_pct = clip_percentile
                vmin_i = np.nanpercentile(Bz_valid_mT, lower_pct)
                vmax_i = np.nanpercentile(Bz_valid_mT, upper_pct)

                if (not np.isfinite(vmin_i)) or (not np.isfinite(vmax_i)) or (vmin_i == vmax_i):
                    vmin_i, vmax_i = -1.0, 1.0

        elif color_scale_mode == "symmetric":
            if Bz_valid_mT.size == 0:
                max_abs_i = 1.0
            else:
                max_abs_i = np.nanpercentile(np.abs(Bz_valid_mT), clip_percentile)
                if (not np.isfinite(max_abs_i)) or (max_abs_i == 0):
                    max_abs_i = 1.0
            vmin_i, vmax_i = -max_abs_i, max_abs_i

        else:
            raise ValueError("color_scale_mode doit être 'minmax' ou 'symmetric'.")

        boundaries_i = np.linspace(vmin_i, vmax_i, n_color_levels + 1)
        norm_i = mcolors.BoundaryNorm(boundaries_i, cmap.N, clip=True)

        #ax.imshow(
        im = ax.imshow(
            Bz_mT,
            origin=origin,
            extent=extent,
            cmap=cmap,
            norm=norm_i, #norm,
            interpolation=interpolation,
        )

        ax.quiver(
            Xq,
            Yq,
            Bx[::step, ::step],
            By[::step, ::step],
            color="k",
            scale=quiver_scale,
            headwidth=headwidth,
            width=width,
        )

        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if invert_y:
            ax.invert_yaxis()

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        row = idx // ncols
        col = idx % ncols

        ax.tick_params(
            labelbottom=(row == nrows - 1),
            labelleft=(col == 0),
            labeltop=False,
            labelright=False,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        elif Bext_list is not None and idx < len(Bext_list):
            ax.set_title(rf"$B_{{ext}}$={Bext_list[idx]:.1f} {Bext_unit}", fontsize=10)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, #)
                            boundaries=boundaries_i, spacing="proportional",)
        cbar.set_label("Bz (mT)")

    # ---------- colorbar ----------
    fig.subplots_adjust(
        left=0.06,
        right=0.95,
        bottom=0.07,
        top=0.95,
        wspace=0.18,
        hspace=0.10,
    )
    """
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig.subplots_adjust(
        left=0.06,
        right=0.85,
        bottom=0.07,
        top=0.95,
        wspace=0.08,
        hspace=0.10,
    )

    cax = fig.add_axes([0.88, 0.12, 0.018, 0.76])
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        boundaries=boundaries,
        spacing="proportional",
    )
    cbar.set_label("Bz (mT)")
    """

    # ---------- scalebar ----------
    if scalebar and (Size_um is not None):
        bar_um = scalebar_frac * Size_um
        x0, y0 = scalebar_xy
        x1, y1 = x0 + scalebar_frac, y0

        if scalebar_indices is None:
            scalebar_indices = (0,)

        for idx in scalebar_indices:
            if not (0 <= idx < n_maps):
                continue

            ax = flat_axes[idx]

            ann1 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False,
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=2,
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
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5),
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    # ---------- sauvegarde ----------
    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    if pdf_path is not None:
        from pathlib import Path
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            pdf_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return results, fig, axes


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm


def plot_reconstructed_B_grid2(
    B_list_T,
    nv_axis,
    dx,
    d_meas,
    extent,
    nrows,
    ncols,
    *,
    step=4,
    quiver_scale=0.10,
    Bext_remove=None,
    Bext_list=None,
    Bext_unit="mT",
    titles=None,
    cmap_name="bwr",
    n_color_levels=30,
    clip_percentile=99.4,
    color_scale_mode="minmax",   # "minmax" ou "symmetric"
    figsize_per_panel=(3.0, 3.0),
    fig_dpi=120,
    save_dpi=300,
    tick_step=20,
    origin="lower",
    interpolation="nearest",
    invert_y=True,
    scalebar=False,
    Size_um=None,
    scalebar_frac=0.25,
    scalebar_xy=(0.05, 0.05),
    scalebar_indices=(0,),
    png_path=None,
    pdf_path=None,
):
    """
    Affiche une grille de reconstructions B à partir d'une liste de cartes B_parallel.

    Chaque panneau montre :
      - fond : Bz (mT)
      - quiver : (Bx, By)

    Paramètres importants
    ---------------------
    B_list_T : list[np.ndarray]
        Liste des cartes B_parallel en Tesla.
    nv_axis : array-like (3,)
        Axe NV.
    dx : float
        Pas pixel pour la reconstruction.
    d_meas : float
        Distance de mesure.
    extent : [x_min, x_max, y_min, y_max]
        Extent matplotlib.
    nrows, ncols : int
        Dimensions de la grille.

    Options utiles
    --------------
    color_scale_mode :
        - "minmax"    : vmin/vmax pris sur les percentiles globaux des valeurs de Bz
        - "symmetric" : échelle symétrique autour de 0 avec max_abs global
    Bext_remove : float, list, np.ndarray or None
        Champ externe (Tesla) à soustraire après reconstruction le long de nv_axis.
        Peut être :
          - un scalaire commun à toutes les mesures
          - une liste/array de longueur len(B_list_T), une valeur par mesure
          - None
    Bext_list : list or None
        Valeurs affichées dans les titres sous la forme Bext=...
    titles : list[str] or None
        Si fourni, remplace les titres automatiques.
    fig_dpi : int
        DPI de la figure affichée à l'écran.
    save_dpi : int
        DPI utilisé pour les sauvegardes.
    """

    def _format_length_um(length_um):
        if length_um >= 1000:
            return length_um / 1000, "mm"
        elif length_um >= 1:
            return length_um, "µm"
        else:
            return length_um * 1000, "nm"

    n_maps = len(B_list_T)
    n_slots = nrows * ncols

    if n_maps == 0:
        raise ValueError("B_list_T est vide.")
    if n_maps > n_slots:
        raise ValueError(
            f"{n_maps} cartes pour seulement {n_slots} sous-figures ({nrows}x{ncols})."
        )

    u = np.asarray(nv_axis, dtype=float)
    u /= np.linalg.norm(u)

    # ---------- gestion de Bext_remove ----------
    if Bext_remove is None:
        Bext_remove_list = [None] * n_maps
    elif np.isscalar(Bext_remove):
        Bext_remove_list = [float(Bext_remove)] * n_maps
    else:
        Bext_remove_list = list(Bext_remove)
        if len(Bext_remove_list) != n_maps:
            raise ValueError(
                "Si Bext_remove est une liste/array, sa longueur doit être égale à len(B_list_T)."
            )

    # ---------- reconstruction ----------
    results = []
    for Bpar_T, Bext_i in zip(B_list_T, Bext_remove_list):
        Bx, By, Bz = reconstruct_B_from_Bparallel_Fourier(Bpar_T, u, dx, d_meas)

        if Bext_i is not None:
            bx0, by0, bz0 = Bext_i * u
            Bx = Bx - bx0
            By = By - by0
            Bz = Bz - bz0

        results.append((Bx, By, Bz))

    # ---------- échelle commune Bz ----------
    """
    all_Bz_mT = np.concatenate([
        (Bz * 1e3)[np.isfinite(Bz)].ravel()
        for (_, _, Bz) in results
    ])

    if color_scale_mode == "minmax":
        if all_Bz_mT.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            lower_pct = 100 - clip_percentile
            upper_pct = clip_percentile
            vmin = np.nanpercentile(all_Bz_mT, lower_pct)
            vmax = np.nanpercentile(all_Bz_mT, upper_pct)

            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                vmin, vmax = -1.0, 1.0

    elif color_scale_mode == "symmetric":
        if all_Bz_mT.size == 0:
            max_abs = 1.0
        else:
            max_abs = np.nanpercentile(np.abs(all_Bz_mT), clip_percentile)
            if (not np.isfinite(max_abs)) or (max_abs == 0):
                max_abs = 1.0
        vmin, vmax = -max_abs, max_abs

    else:
        raise ValueError("color_scale_mode doit être 'minmax' ou 'symmetric'.")

    # --- ancienne version symétrique si tu veux revenir en arrière ---
    # if all_Bz_mT.size == 0:
    #     max_abs = 1.0
    # else:
    #     max_abs = np.nanpercentile(np.abs(all_Bz_mT), clip_percentile)
    #     if (not np.isfinite(max_abs)) or (max_abs == 0):
    #         max_abs = 1.0
    # vmin, vmax = -max_abs, max_abs

    cmap = cm.get_cmap(cmap_name, n_color_levels)
    boundaries = np.linspace(vmin, vmax, n_color_levels + 1)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
    """
    cmap = cm.get_cmap(cmap_name, n_color_levels)

    # ---------- grille spatiale ----------
    x_min, x_max, y_min, y_max = extent
    _, _, Bz0 = results[0]
    ny, nx = Bz0.shape

    dx_phys = (x_max - x_min) / nx
    dy_phys = (y_max - y_min) / ny

    x_centers = x_min + (np.arange(nx) + 0.5) * dx_phys
    y_centers = y_min + (np.arange(ny) + 0.5) * dy_phys
    Xfull, Yfull = np.meshgrid(x_centers, y_centers, indexing="xy")

    Xq = Xfull[::step, ::step]
    Yq = Yfull[::step, ::step]

    xticks = np.arange(
        tick_step * np.ceil(x_min / tick_step),
        x_max + 1e-9,
        tick_step
    )
    yticks = np.arange(
        tick_step * np.ceil(y_min / tick_step),
        y_max + 1e-9,
        tick_step
    )

    # ---------- figure ----------
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=fig_dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    flat_axes = axes.ravel()

    for idx, ax in enumerate(flat_axes):
        if idx >= n_maps:
            ax.axis("off")
            continue

        Bx, By, Bz = results[idx]
        Bz_mT = Bz * 1e3

        ax.imshow(
            Bz_mT,
            origin=origin,
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
        )

        ax.quiver(
            Xq,
            Yq,
            Bx[::step, ::step],
            By[::step, ::step],
            color="k",
            scale=quiver_scale,
        )

        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if invert_y:
            ax.invert_yaxis()

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        row = idx // ncols
        col = idx % ncols

        ax.tick_params(
            labelbottom=(row == nrows - 1),
            labelleft=(col == 0),
            labeltop=False,
            labelright=False,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        elif Bext_list is not None and idx < len(Bext_list):
            ax.set_title(rf"$B_{{ext}}$={Bext_list[idx]:.1f} {Bext_unit}", fontsize=10)

    # ---------- colorbar ----------
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig.subplots_adjust(
        left=0.06,
        right=0.85,
        bottom=0.07,
        top=0.95,
        wspace=0.08,
        hspace=0.10,
    )

    cax = fig.add_axes([0.88, 0.12, 0.018, 0.76])
    cbar = fig.colorbar(
        mappable,
        cax=cax,
        boundaries=boundaries,
        spacing="proportional",
    )
    cbar.set_label("Bz (mT)")

    # ---------- scalebar ----------
    if scalebar and (Size_um is not None):
        bar_um = scalebar_frac * Size_um
        x0, y0 = scalebar_xy
        x1, y1 = x0 + scalebar_frac, y0

        if scalebar_indices is None:
            scalebar_indices = (0,)

        for idx in scalebar_indices:
            if not (0 <= idx < n_maps):
                continue

            ax = flat_axes[idx]

            ann1 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False,
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=2,
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
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5),
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    # ---------- sauvegarde ----------
    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    if pdf_path is not None:
        from pathlib import Path
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            pdf_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return results, fig, axes



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm


def plot_B_projection_grid(
    B_list_T,
    extent,
    nrows,
    ncols,
    *,
    Bext_list=None,
    Bext_unit="mT",
    titles=None,
    cmap_name="bwr",
    n_color_levels=30,
    clip_percentile=99.4,
    color_scale_mode="minmax",   # "minmax" ou "symmetric"
    colormap_individuelle=False, # False = colorbar commune, True = une par sous-figure
    figsize_per_panel=(3.0, 3.0),
    fig_dpi=120,
    save_dpi=300,
    tick_step=20,
    origin="lower",
    interpolation="nearest",
    invert_y=True,
    colorbar_label=r"$B_{\parallel}$ (mT)",
    scalebar=False,
    Size_um=None,
    scalebar_frac=0.25,
    scalebar_xy=(0.05, 0.05),
    scalebar_indices=(0,),
    png_path=None,
    pdf_path=None,
):
    """
    Affiche une grille de cartes de projection B_parallel.

    colormap_individuelle :
      - False : une colorbar commune à toutes les sous-figures
      - True  : chaque sous-figure a sa propre colorbar
    """

    def _format_length_um(length_um):
        if length_um >= 1000:
            return length_um / 1000, "mm"
        elif length_um >= 1:
            return length_um, "µm"
        else:
            return length_um * 1000, "nm"

    def _compute_limits(values_mT):
        if color_scale_mode == "minmax":
            if values_mT.size == 0:
                return -1.0, 1.0
            lower_pct = 100 - clip_percentile
            upper_pct = clip_percentile
            vmin = np.nanpercentile(values_mT, lower_pct)
            vmax = np.nanpercentile(values_mT, upper_pct)
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                return -1.0, 1.0
            return vmin, vmax

        elif color_scale_mode == "symmetric":
            if values_mT.size == 0:
                max_abs = 1.0
            else:
                max_abs = np.nanpercentile(np.abs(values_mT), clip_percentile)
                if (not np.isfinite(max_abs)) or (max_abs == 0):
                    max_abs = 1.0
            return -max_abs, max_abs

        else:
            raise ValueError("color_scale_mode doit être 'minmax' ou 'symmetric'.")

    n_maps = len(B_list_T)
    n_slots = nrows * ncols

    if n_maps == 0:
        raise ValueError("B_list_T est vide.")
    if n_maps > n_slots:
        raise ValueError(
            f"{n_maps} cartes pour seulement {n_slots} sous-figures ({nrows}x{ncols})."
        )

    # ---------- échelle commune si demandée ----------
    if not colormap_individuelle:
        all_B_mT = np.concatenate([
            (B * 1e3)[np.isfinite(B)].ravel()
            for B in B_list_T
        ])
        vmin_global, vmax_global = _compute_limits(all_B_mT)
        cmap_global = cm.get_cmap(cmap_name, n_color_levels)
        boundaries_global = np.linspace(vmin_global, vmax_global, n_color_levels + 1)
        norm_global = mcolors.BoundaryNorm(boundaries_global, cmap_global.N, clip=True)

    # ---------- géométrie ----------
    x_min, x_max, y_min, y_max = extent

    xticks = np.arange(
        tick_step * np.ceil(x_min / tick_step),
        x_max + 1e-9,
        tick_step
    )
    yticks = np.arange(
        tick_step * np.ceil(y_min / tick_step),
        y_max + 1e-9,
        tick_step
    )

    # ---------- figure ----------
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        dpi=fig_dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    flat_axes = axes.ravel()

    # réglage des marges
    if colormap_individuelle:
        fig.subplots_adjust(
            left=0.06,
            right=0.95,
            bottom=0.07,
            top=0.95,
            wspace=0.35,
            hspace=0.18,
        )
    else:
        fig.subplots_adjust(
            left=0.06,
            right=0.85,
            bottom=0.07,
            top=0.95,
            wspace=0.08,
            hspace=0.10,
        )

    for idx, ax in enumerate(flat_axes):
        if idx >= n_maps:
            ax.axis("off")
            continue

        B_mT = B_list_T[idx] * 1e3
        vals = B_mT[np.isfinite(B_mT)]

        if colormap_individuelle:
            vmin, vmax = _compute_limits(vals)
            cmap = cm.get_cmap(cmap_name, n_color_levels)
            boundaries = np.linspace(vmin, vmax, n_color_levels + 1)
            norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)
        else:
            vmin, vmax = vmin_global, vmax_global
            cmap = cmap_global
            boundaries = boundaries_global
            norm = norm_global

        im = ax.imshow(
            B_mT,
            origin=origin,
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation=interpolation,
        )

        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if invert_y:
            ax.invert_yaxis()

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        row = idx // ncols
        col = idx % ncols

        ax.tick_params(
            labelbottom=(row == nrows - 1),
            labelleft=(col == 0),
            labeltop=False,
            labelright=False,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        if titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
        elif Bext_list is not None and idx < len(Bext_list):
            ax.set_title(rf"$B_{{ext}}$={Bext_list[idx]:.1f} {Bext_unit}", fontsize=10)

        if colormap_individuelle:
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                boundaries=boundaries,
                spacing="proportional",
            )
            cbar.set_label(colorbar_label)

    # ---------- colorbar commune ----------
    if not colormap_individuelle:
        mappable = cm.ScalarMappable(norm=norm_global, cmap=cmap_global)
        mappable.set_array([])

        cax = fig.add_axes([0.88, 0.12, 0.018, 0.76])
        cbar = fig.colorbar(
            mappable,
            cax=cax,
            boundaries=boundaries_global,
            spacing="proportional",
        )
        cbar.set_label(colorbar_label)

    # ---------- scalebar ----------
    if scalebar and (Size_um is not None):
        bar_um = scalebar_frac * Size_um
        x0, y0 = scalebar_xy
        x1, y1 = x0 + scalebar_frac, y0

        if scalebar_indices is None:
            scalebar_indices = (0,)

        for idx in scalebar_indices:
            if not (0 <= idx < n_maps):
                continue

            ax = flat_axes[idx]

            ann1 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False,
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=2,
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
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5),
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    # ---------- sauvegarde ----------
    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    if pdf_path is not None:
        from pathlib import Path
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            pdf_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return fig, axes



import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.patches import Rectangle

def plot_reconstructed_B_pair(
    B_list_T,
    nv_axis,
    dx,
    d_meas,
    extent,
    *,
    step=4,
    quiver_scale=0.10,
    Bext_remove=None,
    Bext_list=None,
    Bext_unit="mT",
    Bext_title_scale=1e3,
    titles=None,
    cmap_name="bwr",
    n_color_levels=30,
    clip_percentile=99.4,
    color_scale_mode="minmax",   # "minmax" ou "symmetric"
    colorbar_mode="shared",      # "shared" ou "separate"
    figsize=(8.0, 3.8),
    fig_dpi=120,
    save_dpi=300,
    tick_step=20,
    origin="lower",
    interpolation="nearest",
    invert_y=True,
    scalebar=False,
    Size_um=None,
    scalebar_frac=0.25,
    scalebar_xy=(0.05, 0.05),
    scalebar_indices=(0,),
    png_path=None,
    pdf_path=None,

    show_roi_box=False,
    roi_center=None,
    roi_width=None,
    roi_box_color="red",
    roi_box_lw=1.8,
    plot_zoom=False,
    zoom_figsize=None,
    zoom_step=None,
    zoom_tick_step=None,
    zoom_png_path=None,
    zoom_pdf_path=None,
):
    """
    Compare deux reconstructions B (2 sous-figures uniquement).

    Paramètres importants
    ---------------------
    B_list_T : list[np.ndarray]
        Liste de 2 cartes B_parallel en Tesla.
    colorbar_mode : str
        - "shared"   : une colorbar commune aux deux subfigures
        - "separate" : une colorbar par subfigure, avec échelle locale
    """

    def _format_length_um(length_um):
        if length_um >= 1000:
            return length_um / 1000, "mm"
        elif length_um >= 1:
            return length_um, "µm"
        else:
            return length_um * 1000, "nm"

    def _compute_limits(Bz_mT):
        Bz_valid_mT = Bz_mT[np.isfinite(Bz_mT)]

        if color_scale_mode == "minmax":
            if Bz_valid_mT.size == 0:
                vmin, vmax = -1.0, 1.0
            else:
                lower_pct = 100 - clip_percentile
                upper_pct = clip_percentile
                vmin = np.nanpercentile(Bz_valid_mT, lower_pct)
                vmax = np.nanpercentile(Bz_valid_mT, upper_pct)

                if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmin == vmax):
                    vmin, vmax = -1.0, 1.0

        elif color_scale_mode == "symmetric":
            if Bz_valid_mT.size == 0:
                max_abs = 1.0
            else:
                max_abs = np.nanpercentile(np.abs(Bz_valid_mT), clip_percentile)
                if (not np.isfinite(max_abs)) or (max_abs == 0):
                    max_abs = 1.0
            vmin, vmax = -max_abs, max_abs

        else:
            raise ValueError("color_scale_mode doit être 'minmax' ou 'symmetric'.")

        boundaries = np.linspace(vmin, vmax, n_color_levels + 1)
        norm = mcolors.BoundaryNorm(boundaries, cm.get_cmap(cmap_name, n_color_levels).N, clip=True)
        return vmin, vmax, boundaries, norm
    
    def _compute_zoom_slices(shape, extent, roi_center, roi_width):
        if roi_center is None or roi_width is None:
            raise ValueError(
                "Pour show_roi_box/plot_zoom, il faut fournir roi_center=(x, y) et roi_width."
            )

        ny, nx = shape
        x_min, x_max, y_min, y_max = extent
        cx, cy = roi_center
        half = roi_width / 2

        zx0 = max(x_min, cx - half)
        zx1 = min(x_max, cx + half)
        zy0 = max(y_min, cy - half)
        zy1 = min(y_max, cy + half)

        if (zx1 <= zx0) or (zy1 <= zy0):
            raise ValueError("La zone ROI est vide ou hors du champ.")

        ix0 = int(np.floor((zx0 - x_min) / (x_max - x_min) * nx))
        ix1 = int(np.ceil((zx1 - x_min) / (x_max - x_min) * nx))
        iy0 = int(np.floor((zy0 - y_min) / (y_max - y_min) * ny))
        iy1 = int(np.ceil((zy1 - y_min) / (y_max - y_min) * ny))

        ix0 = max(0, min(ix0, nx - 1))
        ix1 = max(ix0 + 1, min(ix1, nx))
        iy0 = max(0, min(iy0, ny - 1))
        iy1 = max(iy0 + 1, min(iy1, ny))

        dx_plot = (x_max - x_min) / nx
        dy_plot = (y_max - y_min) / ny

        extent_zoom = [
            x_min + ix0 * dx_plot,
            x_min + ix1 * dx_plot,
            y_min + iy0 * dy_plot,
            y_min + iy1 * dy_plot,
        ]

        return slice(iy0, iy1), slice(ix0, ix1), extent_zoom
    


    if len(B_list_T) != 2:
        raise ValueError("B_list_T doit contenir exactement 2 cartes.")

    u = np.asarray(nv_axis, dtype=float)
    u /= np.linalg.norm(u)

    # ---------- gestion de Bext_remove ----------
    if Bext_remove is None:
        Bext_remove_list = [None, None]
    elif np.isscalar(Bext_remove):
        Bext_remove_list = [float(Bext_remove), float(Bext_remove)]
    else:
        Bext_remove_list = list(Bext_remove)
        if len(Bext_remove_list) != 2:
            raise ValueError("Si Bext_remove est une liste/array, elle doit être de longueur 2.")

    # ---------- reconstruction ----------
    results = []
    for Bpar_T, Bext_i in zip(B_list_T, Bext_remove_list):
        Bx, By, Bz = reconstruct_B_from_Bparallel_Fourier(Bpar_T, u, dx, d_meas)

        if Bext_i is not None:
            bx0, by0, bz0 = Bext_i * u
            Bx = Bx - bx0
            By = By - by0
            Bz = Bz - bz0

        results.append((Bx, By, Bz))

    cmap = cm.get_cmap(cmap_name, n_color_levels)

    # ---------- grille spatiale ----------
    x_min, x_max, y_min, y_max = extent
    _, _, Bz0 = results[0]
    ny, nx = Bz0.shape

    dx_phys = (x_max - x_min) / nx
    dy_phys = (y_max - y_min) / ny

    x_centers = x_min + (np.arange(nx) + 0.5) * dx_phys
    y_centers = y_min + (np.arange(ny) + 0.5) * dy_phys
    Xfull, Yfull = np.meshgrid(x_centers, y_centers, indexing="xy")

    Xq = Xfull[::step, ::step]
    Yq = Yfull[::step, ::step]

    xticks = np.arange(
        tick_step * np.ceil(x_min / tick_step),
        x_max + 1e-9,
        tick_step
    )
    yticks = np.arange(
        tick_step * np.ceil(y_min / tick_step),
        y_max + 1e-9,
        tick_step
    )

    zoom_slices = None
    extent_zoom = None

    if show_roi_box or plot_zoom:
        iy_zoom, ix_zoom, extent_zoom = _compute_zoom_slices(
            Bz0.shape, extent, roi_center, roi_width
        )
        zoom_slices = (iy_zoom, ix_zoom)

    # ---------- préparation échelles ----------
    Bz_mT_list = [(Bz * 1e3) for (_, _, Bz) in results]

    if colorbar_mode == "shared":
        all_Bz_mT = np.concatenate([
            Bz_mT[np.isfinite(Bz_mT)].ravel()
            for Bz_mT in Bz_mT_list
        ])

        vmin, vmax, boundaries_shared, norm_shared = _compute_limits(all_Bz_mT)
        norms = [norm_shared, norm_shared]
        boundaries_list = [boundaries_shared, boundaries_shared]

    elif colorbar_mode == "separate":
        norms = []
        boundaries_list = []
        for Bz_mT in Bz_mT_list:
            _, _, boundaries_i, norm_i = _compute_limits(Bz_mT)
            norms.append(norm_i)
            boundaries_list.append(boundaries_i)

    else:
        raise ValueError("colorbar_mode doit être 'shared' ou 'separate'.")

    # ---------- figure ----------
    fig, axes = plt.subplots(
        1, 2,
        figsize=figsize,
        dpi=fig_dpi,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    axes = axes.ravel()

    ims = []

    for idx, ax in enumerate(axes):
        Bx, By, Bz = results[idx]
        Bz_mT = Bz_mT_list[idx]
        norm_i = norms[idx]
        boundaries_i = boundaries_list[idx]

        im = ax.imshow(
            Bz_mT,
            origin=origin,
            extent=extent,
            cmap=cmap,
            norm=norm_i,
            interpolation=interpolation,
        )
        ims.append(im)

        ax.quiver(
            Xq,
            Yq,
            Bx[::step, ::step],
            By[::step, ::step],
            color="k",
            scale=quiver_scale,
        )

        ax.set_aspect("equal")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        if invert_y:
            ax.invert_yaxis()

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.tick_params(
            labelbottom=True,
            labelleft=(idx == 0),
            labeltop=False,
            labelright=False,
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        if Bext_list is not None and idx < len(Bext_list):
            ax.set_title(
                rf"$B_{{ext}}$={Bext_title_scale * Bext_list[idx]:.1f} {Bext_unit}",
                fontsize=10,
            )
        elif titles is not None and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)

        if show_roi_box and extent_zoom is not None:
            x0_zoom, x1_zoom, y0_zoom, y1_zoom = extent_zoom
            rect = Rectangle(
                (x0_zoom, y0_zoom),
                x1_zoom - x0_zoom,
                y1_zoom - y0_zoom,
                fill=False,
                edgecolor=roi_box_color,
                linewidth=roi_box_lw,
            )
            ax.add_patch(rect)

        if colorbar_mode == "separate":
            cbar = fig.colorbar(
                im,
                ax=ax,
                fraction=0.046,
                pad=0.04,
                boundaries=boundaries_i,
                spacing="proportional",
            )
            cbar.set_label("Bz [mT]")

    # colorbar commune
    if colorbar_mode == "shared":
        fig.subplots_adjust(
            left=0.06,
            right=0.88,
            bottom=0.07,
            top=0.93,
            wspace=0.15,
        )
        cax = fig.add_axes([0.90, 0.16, 0.02, 0.68])
        cbar = fig.colorbar(
            ims[0],
            cax=cax,
            boundaries=boundaries_shared,
            spacing="proportional",
        )
        cbar.set_label("Bz [mT]")
    else:
        fig.subplots_adjust(
            left=0.06,
            right=0.95,
            bottom=0.07,
            top=0.93,
            wspace=0.25,
        )

    # ---------- scalebar ----------
    if scalebar and (Size_um is not None):
        bar_um = scalebar_frac * Size_um
        x0, y0 = scalebar_xy
        x1, y1 = x0 + scalebar_frac, y0

        if scalebar_indices is None:
            scalebar_indices = (0,)

        for idx in scalebar_indices:
            if not (0 <= idx < 2):
                continue

            ax = axes[idx]

            ann1 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-", lw=4, color="k"),
                annotation_clip=False,
            )
            ann1.set_zorder(10000)

            ann2 = ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                xycoords="axes fraction",
                arrowprops=dict(
                    arrowstyle="-",
                    lw=2,
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
                fontsize=9,
                bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=1.5),
            )
            txt.set_zorder(10002)
            txt.set_clip_on(False)

    # ---------- sauvegarde ----------
    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    if pdf_path is not None:
        from pathlib import Path
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            pdf_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    if plot_zoom:
        if zoom_figsize is None:
            zoom_figsize = figsize
        if zoom_step is None:
            zoom_step = step
        if zoom_tick_step is None:
            zoom_tick_step = max(1, int(roi_width / 4))

        iy_zoom, ix_zoom = zoom_slices

        B_list_zoom = [Bpar[iy_zoom, ix_zoom] for Bpar in B_list_T]

        plot_reconstructed_B_pair(
            B_list_T=B_list_zoom,
            nv_axis=nv_axis,
            dx=dx,
            d_meas=d_meas,
            extent=extent_zoom,
            step=zoom_step,
            quiver_scale=0.06, #0.10
            Bext_remove=Bext_remove,
            Bext_list=Bext_list,
            Bext_unit=Bext_unit,
            Bext_title_scale=Bext_title_scale,
            titles=titles,
            cmap_name=cmap_name,
            n_color_levels=n_color_levels,
            clip_percentile=clip_percentile,
            color_scale_mode=color_scale_mode,
            colorbar_mode=colorbar_mode,
            figsize=zoom_figsize,
            fig_dpi=fig_dpi,
            save_dpi=save_dpi,
            tick_step=zoom_tick_step,
            origin=origin,
            interpolation=interpolation,
            invert_y=invert_y,
            scalebar=scalebar,
            Size_um=Size_um,
            scalebar_frac=scalebar_frac,
            scalebar_xy=scalebar_xy,
            scalebar_indices=scalebar_indices,
            png_path=zoom_png_path,
            pdf_path=zoom_pdf_path,
            show_roi_box=False,
            roi_center=None,
            roi_width=None,
            plot_zoom=False,
        )

    return results, fig, axes


import numpy as np
import matplotlib.pyplot as plt


def plot_hysteresis_loop_from_lists(
    dates,
    B_maps,
    Bext_maps,
    *,
    ordered_dates=None,
    mean_func=np.nanmean,
    map_scale=1e3,
    map_unit="mT",
    Bext_scale=1e3,
    Bext_unit="mT",
    marker="o",
    markersize=5,
    line_alpha=0.6,
    line_width=1.2,
    annotate_points=False,
    annotate_mode="index",   # "index" ou "date"
    annotation_fontsize=8,
    figsize=(5.0, 4.0),
    fig_dpi=120,
    save_dpi=300,
    title="Boucle d'hystérèse",
    xlabel=None,
    ylabel=None,
    grid=True,
    png_path=None,
    pdf_path=None,

    show_arrows=False,
    arrow_color="0.25",
    arrow_alpha=0.7,
    arrow_lw=1.0,
    shift_arrows_by_direction=False,
    arrow_y_shift=0.0,
):
    """
    Trace une boucle d'hystérèse à partir de listes parallèles :
        dates[i]     <-> B_maps[i] <-> Bext_maps[i]

    Paramètres
    ----------
    dates : list[str]
        Liste des identifiants / dates de mesures.

    B_maps : list[np.ndarray]
        Liste des cartes 2D à moyenner spatialement.

    Bext_maps : list[float]
        Liste des champs externes associés (en T ici dans ton code).

    ordered_dates : list[str] or None
        Si fourni, impose l'ordre des points sur la boucle.
        Si None, on utilise simplement l'ordre de `dates`.

    mean_func : callable
        Fonction de moyenne spatiale, par défaut np.nanmean.

    map_scale : float
        Facteur de conversion pour l'affichage vertical.
        Par défaut 1e3 : T -> mT.

    Bext_scale : float
        Facteur de conversion pour l'affichage horizontal.
        Par défaut 1e3 : T -> mT.
    """

    if not (len(dates) == len(B_maps) == len(Bext_maps)):
        raise ValueError("dates, B_maps et Bext_maps doivent avoir la même longueur.")

    # mapping date -> indice
    date_to_idx = {date: i for i, date in enumerate(dates)}

    if len(date_to_idx) != len(dates):
        raise ValueError("Certaines dates sont dupliquées dans `dates`.")

    if ordered_dates is None:
        ordered_dates = list(dates)

    indices = []
    for date in ordered_dates:
        if date not in date_to_idx:
            raise KeyError(f"La date {date!r} n'est pas présente dans `dates`.")
        indices.append(date_to_idx[date])

    x_vals = []
    y_vals = []

    for idx in indices:
        Bmap = np.asarray(B_maps[idx], dtype=float)
        Bmean = mean_func(Bmap)

        x_vals.append(Bext_maps[idx] * Bext_scale)
        y_vals.append(Bmean * map_scale)

    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)

    fig, ax = plt.subplots(figsize=figsize, dpi=fig_dpi)

    # trait léger reliant les points dans l'ordre
    ax.plot(
        x_vals,
        y_vals,
        "-",
        lw=line_width,
        alpha=line_alpha,
        color="blue",
    )

    # points
    ax.plot(
        x_vals,
        y_vals,
        linestyle="None",
        marker=marker,
        markersize=markersize,
        markerfacecolor="red",
        markeredgecolor="red"
    )

    if show_arrows:
        for i in range(len(x_vals) - 1):
            if shift_arrows_by_direction:
                dy_seg = y_vals[i + 1] - y_vals[i]

                if dy_seg > 0:
                    y_shift_i = arrow_y_shift
                elif dy_seg < 0:
                    y_shift_i = -arrow_y_shift
                else:
                    y_shift_i = 0.0
            else:
                y_shift_i = 0.0

            ax.annotate(
                "",
                xy=(x_vals[i + 1], y_vals[i + 1] + y_shift_i),
                xytext=(x_vals[i], y_vals[i] + y_shift_i),
                arrowprops=dict(
                    arrowstyle="->",
                    color=arrow_color,
                    lw=arrow_lw,
                    alpha=arrow_alpha,
                    shrinkA=4,
                    shrinkB=4,
                ),
            )

    if annotate_points:
        for k, (x, y, date) in enumerate(zip(x_vals, y_vals, ordered_dates)):
            if annotate_mode == "date":
                txt = str(date)
            else:
                txt = str(k)

            ax.annotate(
                txt,
                (x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=annotation_fontsize,
            )

    if xlabel is None:
        xlabel = rf"$B_{{ext}}$ [{Bext_unit}]"
    if ylabel is None:
        ylabel = rf"$\langle B \rangle$ ({map_unit})"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    #ax.set_yscale("log")

    if grid:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()

    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    if pdf_path is not None:
        from pathlib import Path
        Path(pdf_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            pdf_path,
            dpi=save_dpi,
            bbox_inches="tight",
            facecolor="white",
        )

    plt.show()

    return x_vals, y_vals, fig, ax




def plot_B_difference(
    Bx1, By1, Bz1,
    Bx2, By2, Bz2,
    extent,
    title_prefix="B1 - B2",
    png_path=None,
    vlim_mT=None  # si None -> calculé à partir des données
):
    """
    Affiche les cartes d'erreur (B1 - B2) pour Bx, By, Bz
    sous forme de 3 imshow, colormap bwr centrale sur 0.

    Bx1,...,Bz2 : (NY, NX), en Tesla
    extent : [x_min, x_max, y_min, y_max] (µm)
    vlim_mT : float ou None, amplitude max (mT) pour les 3 composantes
              -> vmin = -vlim, vmax = +vlim
    """

    dBx_mT = (Bx1 - Bx2) * 1e3
    dBy_mT = (By1 - By2) * 1e3
    dBz_mT = (Bz1 - Bz2) * 1e3

    # détermine vlim commun si besoin
    if vlim_mT is None:
        vals = np.concatenate([
            dBx_mT[np.isfinite(dBx_mT)].ravel(),
            dBy_mT[np.isfinite(dBy_mT)].ravel(),
            dBz_mT[np.isfinite(dBz_mT)].ravel(),
        ])
        if vals.size:
            vlim_mT = np.percentile(np.abs(vals), 95)
        else:
            vlim_mT = 1.0  # fallback
    vmin = -vlim_mT
    vmax = vlim_mT

    cmap = "bwr"

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(dBx_mT, origin="lower", extent=extent,
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"{title_prefix} : $\\Delta B_x$ (mT)")
    axes[0].set_xlabel("j")
    axes[0].set_ylabel("i")
    axes[0].invert_yaxis()
    fig.colorbar(im0, ax=axes[0], label="mT")

    im1 = axes[1].imshow(dBy_mT, origin="lower", extent=extent,
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[1].set_title(f"{title_prefix} : $\\Delta B_y$ (mT)")
    axes[1].set_xlabel("j")
    axes[1].set_ylabel("i")
    axes[1].invert_yaxis()
    fig.colorbar(im1, ax=axes[1], label="mT")

    im2 = axes[2].imshow(dBz_mT, origin="lower", extent=extent,
                         cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_title(f"{title_prefix} : $\\Delta B_z$ (mT)")
    axes[2].set_xlabel("j")
    axes[2].set_ylabel("i")
    axes[2].invert_yaxis()
    fig.colorbar(im2, ax=axes[2], label="mT")

    plt.show()
    if png_path is not None:
        from pathlib import Path
        Path(png_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            png_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )



# fonction stack :

# fonction stack :
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors


def plot_B_stack_3D(
    B1_stack_T,
    B2_stack_T,
    z_array,
    nv_axis,
    dx,
    extent,
    step=5,
    Bext=None,
    cmap_name="bwr",
    length=1,
    clim3_mT=None,
):
    """
    Visualisation 3D (stack de plans 2D) pour un axe NV donné.

    Paramètres
    ----------
    B1_stack_T, B2_stack_T : np.ndarray
        Forme (Nz, Ny, Nx), cartes B_parallel (Tesla) pour les différentes hauteurs.
        -> typiquement B1_stack_T[k] = |B1_projectionNV[roi]| - B_ext, même chose pour B2.

    z_array : array-like
        Tableau (Nz,) des hauteurs de mesure, en mètres.
        (provenant de scanHeightControl["scanDistance"])

    nv_axis : array-like
        Vecteur NV (3,) en coordonnées (x,y,z).

    dx : float
        Pas spatial dans le plan sur lequel est défini B_parallel (en µm, cohérent avec `extent`).

    extent : [x_min, x_max, y_min, y_max]
        Extent physique en µm pour l'affichage (comme dans tes imshow habituels).

    step : int
        Sous-échantillonnage pour la grille des surfaces 3D (rstride/cstride).

    Bext : float ou None
        Si non None : norme (en Tesla) du champ de biais le long de nv_axis.
        Il est retiré de chaque champ reconstruit.

    cmap_name : str
        Colormap utilisée (par défaut "bwr" pour le signe).
    """

    Nz, Ny, Nx = B1_stack_T.shape

    # --- normalisation de l'axe NV ---
    u = np.asarray(nv_axis, dtype=float)
    u /= np.linalg.norm(u)
    ux, uy, uz = u

    # --- champ de biais éventuel ---
    if Bext is not None:
        bx0, by0, bz0 = Bext * u
    else:
        bx0 = by0 = bz0 = 0.0

    # --- tableaux pour stocker les champs reconstruits ---
    Bx1_stack = np.zeros_like(B1_stack_T)
    By1_stack = np.zeros_like(B1_stack_T)
    Bz1_stack = np.zeros_like(B1_stack_T)

    Bx2_stack = np.zeros_like(B2_stack_T)
    By2_stack = np.zeros_like(B2_stack_T)
    Bz2_stack = np.zeros_like(B2_stack_T)

    def _vv(local):
        if local is not None:
            return local
        return clim3_mT if clim3_mT is not None else (None, None)

    v3min, v3max = _vv(clim3_mT)

    # --- reconstruction à chaque hauteur ---
    for k in range(Nz):
        d_meas_k = z_array[k]   # en mètres
        B1_T = B1_stack_T[k]
        B2_T = B2_stack_T[k]

        Bx1, By1, Bz1 = reconstruct_B_from_Bparallel_Fourier(
            B1_T, nv_axis, dx, d_meas_k
        )
        Bx2, By2, Bz2 = reconstruct_B_from_Bparallel_Fourier(
            B2_T, nv_axis, dx, d_meas_k
        )

        # retrait éventuel du champ de biais
        Bx1 -= bx0
        By1 -= by0
        Bz1 -= bz0

        Bx2 -= bx0
        By2 -= by0
        Bz2 -= bz0

        Bx1_stack[k] = Bx1
        By1_stack[k] = By1
        Bz1_stack[k] = Bz1

        Bx2_stack[k] = Bx2
        By2_stack[k] = By2
        Bz2_stack[k] = Bz2

    # --- projection NV et différence ---
    Bpar1_stack_mT = (Bx1_stack * ux + By1_stack * uy + Bz1_stack * uz) * 1e3
    Bpar2_stack_mT = (Bx2_stack * ux + By2_stack * uy + Bz2_stack * uz) * 1e3
    BparD_stack_mT = (Bpar2_stack_mT - Bpar1_stack_mT)

    # --- Bz en mT ---
    Bz1_stack_mT = Bz1_stack * 1e3
    Bz2_stack_mT = Bz2_stack * 1e3

    # --- échelles de couleurs communes (robustes) ---
    # On utilise des percentiles pour ignorer les outliers extrêmes
    # Pour Bz
    vals_z = np.concatenate([
        Bz1_stack_mT[np.isfinite(Bz1_stack_mT)],
        Bz2_stack_mT[np.isfinite(Bz2_stack_mT)],
    ])
    if vals_z.size:
        # percentiles 1–99 % (à ajuster si besoin)
        vmin_z, vmax_z = np.percentile(vals_z, [3, 97])
    else:
        vmin_z, vmax_z = -1.0, 1.0

    # Pour ΔB‖ (B1 - B2)
    vals_par = BparD_stack_mT[np.isfinite(BparD_stack_mT)]
    if vals_par.size:
        # On force la base à 0, on coupe les valeurs aberrantes en haut
        vmax_par = np.percentile(vals_par, 97)
        vmin_par = 0.0
    else:
        vmin_par, vmax_par = 0.0, 1.0


    cmap = cm.get_cmap(cmap_name)
    norm_z = mcolors.Normalize(vmin=vmin_z, vmax=vmax_z)
    norm_par = mcolors.Normalize(vmin=v3min, vmax=v3max) #vmin_par, vmax=vmax_par)

    # # --- grille spatiale (x, y) ---
    # x_min, x_max, y_min, y_max = extent
    # dx_phys = (x_max - x_min) / Nx
    # dy_phys = (y_max - y_min) / Ny
    # x_centers = x_min + (np.arange(Nx) + 0.5) * dx_phys
    # y_centers = y_min + (np.arange(Ny) + 0.5) * dy_phys
    # X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")
    # --- grille spatiale (x, y) en µm ---
    if length is not None:
        Lx_um = length * 1e6
        Ly_um = length * 1e6  # si ton scan est carré
        x_min, x_max, y_min, y_max = 0.0, Lx_um, 0.0, Ly_um
    else:
        x_min, x_max, y_min, y_max = extent

    dx_phys = (x_max - x_min) / Nx
    dy_phys = (y_max - y_min) / Ny
    x_centers = x_min + (np.arange(Nx) + 0.5) * dx_phys
    y_centers = y_min + (np.arange(Ny) + 0.5) * dy_phys
    X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")


    # --- z en nm pour l'affichage ---
    z_nm = np.array(z_array) * 1e9

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    # pour les colorbars
    mappable_z = cm.ScalarMappable(norm=norm_z, cmap="plasma")
    mappable_par = cm.ScalarMappable(norm=norm_par, cmap=cmap)

    # --- stack de surfaces pour chaque z ---
    for k in range(Nz):
        Zk = np.full_like(X, z_nm[k])

        # B1 : Bz1
        C1 = cmap(norm_z(Bz1_stack_mT[k]))
        ax1.plot_surface(
            X, Y, Zk,
            facecolors=C1,
            rstride=step, cstride=step,
            linewidth=0, antialiased=False, shade=False,
            alpha=0.9,
        )

        # B2 : Bz2
        C2 = cmap(norm_z(Bz2_stack_mT[k]))
        ax2.plot_surface(
            X, Y, Zk,
            facecolors=C2,
            rstride=step, cstride=step,
            linewidth=0, antialiased=False, shade=False,
            alpha=0.9,
        )

        # différence : ΔB‖
        C3 = cmap(norm_par(BparD_stack_mT[k]))
        ax3.plot_surface(
            X, Y, Zk,
            facecolors=C3,
            rstride=step, cstride=step,
            linewidth=0, antialiased=False, shade=False,
            alpha=0.9,
        )

    # --- mise en forme des axes ---
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_zlabel("z (nm)")
        ax.view_init(elev=20, azim=-60)

    ax1.set_title(r"Stack $B_z$ reconstruit (B1)")
    ax2.set_title(r"Stack $B_z$ reconstruit (B2)")
    ax3.set_title(r"Stack $\Delta B_{\parallel}$ (B1 - B2)")

    # --- colorbars ---
    cbar1 = fig.colorbar(mappable_z, ax=ax1, shrink=0.7, pad=0.1)
    cbar1.set_label("Bz (mT)")
    cbar2 = fig.colorbar(mappable_z, ax=ax2, shrink=0.7, pad=0.1)
    cbar2.set_label("Bz (mT)")
    cbar3 = fig.colorbar(mappable_par, ax=ax3, shrink=0.7, pad=0.1)
    cbar3.set_label(r"$\Delta B_{\parallel}$ (mT)")

    plt.show()

    # Je renvoie aussi les champs 3D reconstruits si tu veux les réutiliser
    return (Bx1_stack, By1_stack, Bz1_stack), (Bx2_stack, By2_stack, Bz2_stack), BparD_stack_mT



import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

def plot_B_quiver_stack_3D_white(
    B1_stack_T,
    B2_stack_T,
    z_array,
    nv_axis,
    dx,
    extent,
    step=5,                 # stride surfaces (rstride/cstride)
    quiver_step=5,          # sous-échantillonnage des flèches
    Bext=None,
    cmap_name="turbo",      # colormap pour ΔB_parallel
    length=1,               # longueur scan (m) si tu veux ignorer extent
    clim_bz_mT=None,        # (vmin,vmax) pour Bz en mT, sinon auto robuste
    clim3_mT=None,          # (vmin,vmax) pour ΔB_parallel en mT, sinon auto robuste
    quiver_length=None,     # longueur des flèches en µm (si None -> auto)
    quiver_kwargs=None,     # dict optionnel passé à ax.quiver (couleur, lw, etc.)
):
    """
    Stack 3D fond blanc:
      - Panel 1: Bz(bwr) + quiver(Bx,By) reconstruit depuis B1
      - Panel 2: Bz(bwr) + quiver(Bx,By) reconstruit depuis B2
      - Panel 3: ΔB_parallel = (B1 - B2)·u_NV (mT) en surface colorée

    Retourne:
      (Bx1_stack, By1_stack, Bz1_stack), (Bx2_stack, By2_stack, Bz2_stack), BparD_stack_mT
    """

    Nz, Ny, Nx = B1_stack_T.shape

    # --- normalisation NV ---
    u = np.asarray(nv_axis, dtype=float)
    u /= np.linalg.norm(u)
    ux, uy, uz = u

    # --- champ de biais éventuel ---
    if Bext is not None:
        bx0, by0, bz0 = (Bext * u)
    else:
        bx0 = by0 = bz0 = 0.0

    # --- reconstructions ---
    Bx1_stack = np.zeros_like(B1_stack_T, dtype=float)
    By1_stack = np.zeros_like(B1_stack_T, dtype=float)
    Bz1_stack = np.zeros_like(B1_stack_T, dtype=float)

    Bx2_stack = np.zeros_like(B2_stack_T, dtype=float)
    By2_stack = np.zeros_like(B2_stack_T, dtype=float)
    Bz2_stack = np.zeros_like(B2_stack_T, dtype=float)

    for k in range(Nz):
        d_meas_k = z_array[k]  # mètres
        B1_T = B1_stack_T[k]
        B2_T = B2_stack_T[k]

        Bx1, By1, Bz1 = reconstruct_B_from_Bparallel_Fourier(B1_T, nv_axis, dx, d_meas_k)
        Bx2, By2, Bz2 = reconstruct_B_from_Bparallel_Fourier(B2_T, nv_axis, dx, d_meas_k)

        # retrait biais
        Bx1 -= bx0; By1 -= by0; Bz1 -= bz0
        Bx2 -= bx0; By2 -= by0; Bz2 -= bz0

        Bx1_stack[k], By1_stack[k], Bz1_stack[k] = Bx1, By1, Bz1
        Bx2_stack[k], By2_stack[k], Bz2_stack[k] = Bx2, By2, Bz2

    # --- Bz en mT pour le fond bwr ---
    Bz1_stack_mT = Bz1_stack * 1e3
    Bz2_stack_mT = Bz2_stack * 1e3

    # --- ΔB_parallel en mT (B1 - B2)·uNV ---
    Bpar1_stack_mT = (Bx1_stack * ux + By1_stack * uy + Bz1_stack * uz) * 1e3
    Bpar2_stack_mT = (Bx2_stack * ux + By2_stack * uy + Bz2_stack * uz) * 1e3
    BparD_stack_mT = (Bpar1_stack_mT - Bpar2_stack_mT)

    # --- grille XY en µm ---
    if length is not None:
        Lx_um = length * 1e6
        Ly_um = length * 1e6
        x_min, x_max, y_min, y_max = 0.0, Lx_um, 0.0, Ly_um
    else:
        x_min, x_max, y_min, y_max = extent

    dx_phys = (x_max - x_min) / Nx
    dy_phys = (y_max - y_min) / Ny
    x_centers = x_min + (np.arange(Nx) + 0.5) * dx_phys
    y_centers = y_min + (np.arange(Ny) + 0.5) * dy_phys
    X, Y = np.meshgrid(x_centers, y_centers, indexing="xy")

    # z en nm pour affichage
    z_nm = np.asarray(z_array) * 1e9

    # --- normalisations couleurs ---
    # Bz: échelle commune symétrique (robuste) sauf si clim_bz_mT fourni
    if clim_bz_mT is not None:
        vmin_bz, vmax_bz = clim_bz_mT
    else:
        vals = np.concatenate([
            Bz1_stack_mT[np.isfinite(Bz1_stack_mT)].ravel(),
            Bz2_stack_mT[np.isfinite(Bz2_stack_mT)].ravel(),
        ])
        if vals.size:
            clip = 99.4
            max_abs = np.nanpercentile(np.abs(vals), clip)
        else:
            max_abs = 1.0
        vmin_bz, vmax_bz = -max_abs, max_abs

    norm_bz = mcolors.Normalize(vmin=vmin_bz, vmax=vmax_bz)
    cmap_bz = cm.get_cmap("bwr")

    # ΔB_parallel: par défaut [0, p97(|Δ|)] si clim3_mT pas donné
    if clim3_mT is not None:
        vmin_par, vmax_par = clim3_mT
    else:
        vals_par = np.abs(BparD_stack_mT[np.isfinite(BparD_stack_mT)])
        if vals_par.size:
            vmin_par = 0.0
            vmax_par = np.nanpercentile(vals_par, 97)
        else:
            vmin_par, vmax_par = 0.0, 1.0

    norm_par = mcolors.Normalize(vmin=vmin_par, vmax=vmax_par)
    cmap_par = cm.get_cmap(cmap_name)

    # --- quiver paramètres ---
    if quiver_kwargs is None:
        quiver_kwargs = {}
    qkw = dict(color="k", linewidth=0.5)
    qkw.update(quiver_kwargs)

    Xq = X[::quiver_step, ::quiver_step]
    Yq = Y[::quiver_step, ::quiver_step]

    if quiver_length is None:
        pitch = max(dx_phys, dy_phys) * quiver_step
        quiver_length = 0.7 * pitch  # en µm (unités XY)

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2, projection="3d")
    ax3 = fig.add_subplot(1, 3, 3, projection="3d")

    # mappables colorbars
    mappable_bz = cm.ScalarMappable(norm=norm_bz, cmap=cmap_bz)
    mappable_par = cm.ScalarMappable(norm=norm_par, cmap=cmap_par)

    for k in range(Nz):
        Zk = np.full_like(X, z_nm[k], dtype=float)
        Zq = np.full_like(Xq, z_nm[k], dtype=float)

        # --- Panel 1: B1 ---
        C1 = cmap_bz(norm_bz(Bz1_stack_mT[k]))
        ax1.plot_surface(
            X, Y, Zk,
            facecolors=C1,
            rstride=step, cstride=step,
            linewidth=0, antialiased=False, shade=False,
            alpha=1.0,
        )
        U1 = Bx1_stack[k][::quiver_step, ::quiver_step]
        V1 = By1_stack[k][::quiver_step, ::quiver_step]
        ax1.quiver(Xq, Yq, Zq, U1, V1, 0.0,
                   length=quiver_length, normalize=True, **qkw)

        # --- Panel 2: B2 ---
        C2 = cmap_bz(norm_bz(Bz2_stack_mT[k]))
        ax2.plot_surface(
            X, Y, Zk,
            facecolors=C2,
            rstride=step, cstride=step,
            linewidth=0, antialiased=False, shade=False,
            alpha=1.0,
        )
        U2 = Bx2_stack[k][::quiver_step, ::quiver_step]
        V2 = By2_stack[k][::quiver_step, ::quiver_step]
        ax2.quiver(Xq, Yq, Zq, U2, V2, 0.0,
                   length=quiver_length, normalize=True, **qkw)

        # --- Panel 3: ΔB_parallel ---
        C3 = cmap_par(norm_par(BparD_stack_mT[k]))
        ax3.plot_surface(
            X, Y, Zk,
            facecolors=C3,
            rstride=step, cstride=step,
            linewidth=0, antialiased=False, shade=False,
            alpha=1.0,
        )

    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("x (µm)")
        ax.set_ylabel("y (µm)")
        ax.set_zlabel("z (nm)")
        ax.view_init(elev=20, azim=-60)

    ax1.set_title(r"Stack $B_z$ (bwr) + quiver $(B_x,B_y)$ (B1)")
    ax2.set_title(r"Stack $B_z$ (bwr) + quiver $(B_x,B_y)$ (B2)")
    ax3.set_title(r"Stack $\Delta B_{\parallel} = (B_1-B_2)\cdot \vec{u}_{NV}$ (mT)")

    # colorbars
    cbar_bz = fig.colorbar(mappable_bz, ax=[ax1, ax2], shrink=0.7, pad=0.08)
    cbar_bz.set_label(r"$B_z$ (mT)")

    cbar_par = fig.colorbar(mappable_par, ax=ax3, shrink=0.7, pad=0.08)
    cbar_par.set_label(r"$\Delta B_{\parallel}$ (mT)")

    plt.show()

    return (Bx1_stack, By1_stack, Bz1_stack), (Bx2_stack, By2_stack, Bz2_stack), BparD_stack_mT
