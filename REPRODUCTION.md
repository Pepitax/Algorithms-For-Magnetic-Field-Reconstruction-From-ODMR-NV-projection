# Manuscript figures and tables

The labels below refer to the revised manuscript supplied on 15 October 2026.
Generated results are under `outputs/`; immutable manuscript exports are under
`reference_figures/`. Numerical verification uses `reference_results/`.

## Figures

| Manuscript figure / label | Notebook | Export relative to `outputs/` |
| --- | --- | --- |
| NV-axis schematic, `fig:nv_orientation` | Manually drawn illustration | Supplied as `reference_figures/U_nv2.png`; no numerical calculation |
| Sign-ambiguity schematic | Manually drawn illustration | Supplied as `reference_figures/shema.png`; no numerical calculation |
| Signed and unsigned simulated projection, `fig:simu` | Simulation | `simulation/simu_NV_projection_lattice_abs.pdf` |
| Truth / graph cut / refined / residual, `fig:gt_vs_recon2` | Simulation | `simulation/GT_graphcut.pdf` |
| Vector truth / reconstruction / error, `fig:error1` | Simulation | `simulation/reconstruction3D_GT_refined.pdf` |
| 10 mT / graph cut / 50 mT, `fig:graphcut` | Measurement | `sign_unwrap_2026-05-25-10-27-11/comparaison_B_parallel.pdf` |
| Graph cut / refined / difference, `fig:graphcut2` | Measurement | `sign_unwrap_2026-05-25-10-27-11/B_parallel_refined.pdf` |
| Experimental vector reconstruction and zoom, `fig:reconstruction_field` | Measurement | `sign_unwrap_2026-05-25-10-27-11/Brec_full_and_zoom_quiver.pdf` |
| Transverse shifts, `fig:bonusD` | Hamiltonian | `hamiltonian/shift_Db.png` and `hamiltonian/shift_D2b.png` |
| Synthetic source maps, `fig:sourcemaps` | All-cases benchmark | `benchmark_all_cases/source_maps.pdf` |
| Manual seeds, `fig:manual_seeds` | Simulation and measurement | `simulation/seeds.pdf` and `sign_unwrap_2026-05-25-10-27-11/seeds_experiment.pdf` |
| Experimental spectrum ROI, `fig:roi` | Measurement | `sign_unwrap_2026-05-25-10-27-11/scan_spectres_roi.png` |
| 7 x 7 measured spectra, `fig:measured_spectra` | Measurement | `sign_unwrap_2026-05-25-10-27-11/scan_spectres_annexe.pdf` |
| Component errors, `fig:error2` | Simulation | `simulation/error_BxByBz.pdf` |
| Four uniform-field cases, `fig:gt_vs_recon` | Four-case benchmark | `benchmark/gt_vs_reconstructed_2x4.pdf` |

The measurement notebook also exports `Bpar_roi_zoom_quiver.pdf` as an additional
view. Local spectra and ROI overlays use the same configured center and radius.
Numerical arrays are exported as `simulation/simulation_results.npz` and
`sign_unwrap_2026-05-25-10-27-11/sign_unwrap_results.npz`.

## Tables

| Manuscript table label | Export relative to `outputs/` | Columns / interpretation |
| --- | --- | --- |
| `tab:rmse3` | `simulation/projection_metrics.csv` | RMS and maximum errors of the signed projection, before and after refinement |
| `tab:rmse2` | `simulation/vector_metrics.csv` | Rows Bx, By, Bz; AC RMS, maximum AC error and maximum AC truth |
| `tab:clean_vs_noisy` | `benchmark_all_cases/oracle_clean_noisy_summary.csv` | Arithmetic means of p/alpha and errors; geometric mean of lambda0 |
| `tab:bias_vs_nobias` | `benchmark_all_cases/oracle_clean_bias_summary.csv` | Clean maps with no bias or NV bias; errors include diagnostic DC convention |
| `tab:model_error` | `hamiltonian/single_resonance_model_error.csv` | Apparent fields from each transition, for illustrative transverse fields |
| `tab:finite_fov` | `simulation/finite_fov_metrics.csv` | Cropped-window vector errors, with/without zero padding, all/interior/border regions |
| `tab:rmse` | `benchmark/fixed_protocol_metrics.csv` | Four configurations with fixed filter and diagnostic DC convention; full RMS columns |

Unrounded values are exported. Manuscript tables round these values for display.
`oracle_best_parameters.csv` contains all 56 selected configurations, including
the Gaussian noise realization fixed by `noise_seed=1234`. The search grids are
unchanged from the research notebooks.

## Changes limited to the public copy

- Updated the three older public notebooks and their helpers to the current
  research versions; added the dedicated FeGa notebook and its dependencies.
- Copied the manual simulation mask used by the manuscript and the saved
  experimental mask. No truth-corrected mask or automatic truth-based seed
  generation is included.
- Made filesystem paths relative and routed exports to per-notebook directories.
- Added CSV/NPZ exports and guards so saved seeds support unattended execution.
- Extracted the Hamiltonian equations from the research hysteresis notebook;
  made physical constants explicit parameters and exported both longitudinal
  cases without changing the Hamiltonian or the cubic-root formula.
- Removed saved notebook outputs, local path comments and two identifying
  metadata fields. Corrected stale explanatory seed counts and arrow gain text.

No fit model, graph-cut weight, refinement coefficient, Fourier regularization,
coordinate transform, manual seed value or benchmark search grid was retuned.
The private research files were not modified.
