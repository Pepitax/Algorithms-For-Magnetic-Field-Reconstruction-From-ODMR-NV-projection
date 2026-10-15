# Magnetic field reconstruction from unsigned ODMR maps

Reproduction code and experimental inputs for **Magnetic Field Reconstruction
from Unsigned ODMR Maps of a Ferromagnetic Sample**, Henri Roniger and Daniele
Mari. This snapshot follows the revised manuscript reviewed on 15 October 2026.

The notebooks cover semi-supervised sign recovery, optional continuous refinement,
Fourier vector-field reconstruction, synthetic benchmarks, and illustrative NV
Hamiltonian calculations. The experimental example uses the FeGa scans acquired
on 25 May 2026 at nominal applied fields of 10 and 50 mT.

## Setup

The numerical environment was tested with Python 3.11. Use a dedicated environment:

```sh
python -m venv .venv
```

Activate it with `.venv\Scripts\activate` on Windows or
`source .venv/bin/activate` on Linux/macOS, then install the tested dependencies:

```sh
python -m pip install -r requirements.txt
```

For mouse annotation in an existing Jupyter frontend, also install:

```sh
python -m pip install -r requirements-interactive.txt
```

Open notebooks in Jupyter or VS Code, using this environment as the kernel.
Launch from this repository's root. A notebook frontend is not required for the
batch command below.

## Reproduce all results

From the repository root:

```sh
python tools/reproduce.py
```

This runs all five notebooks in separate fresh kernels, recomputes both full
experimental fit maps from the included spectra, and checks the results against
the numerical reference files. The Python running the command is also used for
the notebook kernels; no machine-specific kernel registration is required.
The full oracle parameter search evaluates 630 configurations for each of 56
synthetic measurements. Allow several minutes for fitting, plotting and searches.

The supplied notebooks retain saved outputs so figures and results can be
inspected without rerunning the analyses. Batch execution writes fresh results
under `outputs/`, which is ignored by Git, and executed notebook copies under
`outputs/executed_notebooks/`. It does not overwrite the supplied notebooks.
The final report is `outputs/verification.json`.

Additional commands:

```sh
python tools/reproduce.py --only simulation measurement
python tools/reproduce.py --only benchmark all-cases hamiltonian
python tools/reproduce.py --check-inputs
python tools/reproduce.py --verify-only
```

`--check-inputs` checks that the five notebooks exist and have valid notebook
formats, without executing any cells. Saved outputs are accepted; file hashes
are not checked. `--verify-only` compares previously generated results against
the numerical references and expects that all five notebooks have already been
run through the batch command.

## Notebooks

| Notebook | Purpose | Output directory |
| --- | --- | --- |
| `odmr_sign_unwrap_simulation_test.ipynb` | Unsigned synthetic projection, manual seeds, graph cut, refinement, vector reconstruction and finite-window diagnostic | `outputs/simulation/` |
| `odmr_sign_unwrap_measurement_2026_05_25.ipynb` | FeGa local spectra, full-map single-resonance fits, saved/manual seeds and experimental reconstruction | `outputs/sign_unwrap_2026-05-25-10-27-11/` |
| `nv_reconstruction_benchmark.ipynb` | Four uniform-field configurations and the diagnostic zero-frequency convention | `outputs/benchmark/` |
| `nv_reconstruction_benchmark_all_cases.ipynb` | Seven source maps, four bias variants, clean/noisy projections and oracle search | `outputs/benchmark_all_cases/` |
| `nv_odmr_hamiltonian.ipynb` | Transverse-field frequency shifts, single-resonance model-error table and analytical-root verification | `outputs/hamiltonian/` |

The dedicated FeGa notebook reproduces the manuscript experiment using the same
fit and plot functions as the research QZabre notebook. The broader QZabre and
hysteresis notebooks contain other acquisitions, including WCCo two-resonance
fits, and are outside this reproduction package. The appendix Hamiltonian
calculations have been extracted from the hysteresis notebook into a standalone
notebook and `nv_hamiltonian_helpers.py`, with the same equations.

## Seed masks

- `saved_seed_masks/seed_mask_simulation.npy`: **489 manually annotated pixels**,
  the mask used for the manuscript results. No ground-truth correction is applied
  to this input. The public package contains no truth-corrected seed mask and
  no option to generate annotations from the ground truth.
- `saved_seed_masks/seeds_2026-05-25-10-27-11.npz`: experimental mask, with
  **90 annotated pixels** (50 positive, 40 negative), acquisition date and the
  associated fitted unsigned map for compatibility checking.

The experimental notebook uses `SEED_MODE='saved'` by default. The modes
`'coordinates'` and `'interactive'` retain disk-based and mouse-based annotation.
The coordinate list produces the same experimental mask as the saved input.
For interactive annotation, pause after the annotation cell and complete the
mouse selections before running validation. Generated masks are written to
`outputs/`; the supplied inputs are not overwritten.

The experimental spectrum ROI is centered at `(i, j) = (90, 82)` with
`half_small=3`, reproducing the manuscript's 7 by 7 grid. Set `half_small=2` for
a 5 by 5 grid. `plot_contrast_argmin` displays its location. This local inspection
does not restrict the full-map fits used for reconstruction.

## Inputs, helpers and numerical references

`Datas2/` contains only the two experimental acquisitions used here. Each includes
the original `odmr.h5`, its converted `odmr.txt`, and acquisition settings in
`imageMeta.json`. The notebooks read the text spectra directly. See
[Datas2/README.md](Datas2/README.md) for layout and metadata details.

The shared modules retain the research implementations:

- `helpers.py`: acquisition loading, preprocessing, NV axes and Fourier inversion;
- `fits.py` and `odmr_initMAP.py`: existing spectrum-fit and plotting utilities;
- `odmr_sign_unwrap.py`: sign annotation, graph cut and continuous refinement;
- `sim_odmr_field_helpers.py`: synthetic fields, diagnostics, metrics and plots;
- `odmr_measurement_helpers.py`: single-resonance extraction and experimental plots;
- `nv_hamiltonian_helpers.py`: the appendix spin Hamiltonian and cubic roots.

`reference_results/` contains independently calculated manual-mask metrics,
previously verified experimental arrays, and benchmark tables. These files are
used **only for verification**, never to replace fitting or reconstruction in
the notebooks. `reference_figures/` contains the revised manuscript exports for
visual comparison, including two manually drawn schematics. PDF bytes can differ
after regeneration because of timestamps, fonts and backend details; verification
compares numerical arrays rather than PDF hashes.

[REPRODUCTION.md](REPRODUCTION.md) maps every manuscript figure and numerical
table to its notebook and exported filename. [VALIDATION.md](VALIDATION.md)
records the tests performed when preparing this package.

## Interpretation

The fixed Fourier filter is `lambda(k) = lambda0 * (1 + (k/k0)**p)`, with
`lambda0=1e-8`, `p=10`, `k0=alpha/z` and `alpha=1`. The main sign-recovery
pipelines remove the uniform mode; the four-case benchmark retains its separately
documented diagnostic DC convention. Its full-field errors therefore differ from
the AC errors of the main pipeline. Oracle search results use synthetic truth and
are not an experimental hyperparameter calibration.

Simulation global-sign alignment uses truth only to resolve the overall sign
for scoring; both reported alignment factors are +1 for the supplied manual mask.
This scoring convention does not select, correct or replace any seed annotation.
The experimental signs were informed by the higher-bias image, which is not an
independent ground truth. Single-resonance conversion and the image/NV frame
relation remain physical assumptions, as stated in the manuscript. Continuous
refinement is optional and does not guarantee better accuracy.

## Moving this package

Copy the notebooks, modules, documentation, `tools/`, `Datas2/`,
`saved_seed_masks/`, reference directories, requirements and
`.gitignore` to the destination repository. Keep these relative paths. Generated
`outputs/`, environments and caches are unnecessary. Preserve the destination's
own `.git` directory rather than copying this staging repository's Git metadata.

No code or data license has been assigned by this packaging step; the authors
should specify their chosen license before public release.
