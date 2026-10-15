# Public reproduction package

## Goal
Prepare a portable, self-contained local repository reproducing the numerical
figures and tables of the revised ODMR manuscript from simulations and measured
FeGa spectra, with the original manual masks used for the manuscript.

## Non-goals
No changes to the research sources outside this directory. No push, commit,
publication, new scientific methods, or unrelated WCCo/hysteresis acquisition data.

## Constraints
- Preserve scientific parameters, fitting algorithms, manual annotations and
  coordinate conventions of the verified research notebooks.
- Keep input datasets and masks immutable during default notebook execution.
- All input and output paths must be relative to this repository.
- Preserve the existing nested Git repository and user changes.
- Keep optional interactive and coordinate-based seed workflows available.

## Acceptance criteria and verification
1. The four current reconstruction notebooks, required helper modules, two FeGa
   datasets, manual masks and standalone appendix Hamiltonian calculations exist.
2. Notebooks execute from a fresh kernel with saved seeds by default; full-map
   fits are recomputed from the included spectra, without a private fit cache.
3. Recomputed simulation metrics, experimental fits, sign maps and vector fields
   agree with independently verified manuscript arrays within stated tolerances.
4. All numerical manuscript figures and tables have a documented reproduction
   path; manually drawn schematics are supplied and identified separately.
5. A relative-path manifest verifies input integrity; relocation execution does
   not import modules or read data from the private PDM directory.
6. Notebook outputs, local absolute paths and unrelated acquisitions are excluded
   from the distributable sources. Requirements and instructions are complete.
7. Hashes of the private source notebooks/modules/masks/datasets remain unchanged.

## Risks and assumptions
The image/NV frame relation remains a stated physical assumption. The high-bias
scan guides annotations and is not independent ground truth. Numerical optimizer
roundoff can differ across platforms. Interactive mouse operation needs a Jupyter
frontend; headless execution verifies the default saved-mask path.

## Progress log
- Inspected existing public repository, current private notebooks, latest
  manuscript figure inventory and the installed scientific environment.
- Baseline: public files date from June; the old public simulation mask has 493
  annotated pixels. The manuscript manual mask has 489 pixels, SHA-256
  `897dfe01851ebc88c133b08a4f711b7500373eda6894fc1456d23e99eb785f7e`.
- Replaced the stale public files, added the dedicated FeGa notebook and
  standalone Hamiltonian notebook, included the two measured spectra datasets
  and supplied manual masks, and documented every manuscript figure/table.
- All five notebooks executed twice in fresh kernels, including one complete
  execution in a relocated copy without prior outputs. All numerical reference
  checks passed, including full-map experimental fits and complete oracle search.
- Confirmed scientific helper syntax-tree equivalence, exact disk/saved mask
  equality, annotation callbacks, source hygiene and unchanged private inputs.
- All acceptance criteria met for the tested environment. Jupyter mouse frontend
  interaction and other operating systems remain outside the executable audit.
- User requested a manual-seed-only public package: removed the truth-corrected
  mask, its notebook option and unused automatic truth-based annotation helper.
  Preserved the exact manual input and user-executed notebook copies under outputs.
