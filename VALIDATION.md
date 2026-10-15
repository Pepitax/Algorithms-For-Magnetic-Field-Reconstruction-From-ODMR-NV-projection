# Validation of the public package

Verified on 15 October 2026, Windows, Python 3.11.9, using the numerical package
versions recorded in `requirements.txt`. The machine-readable summary is
`verification_summary.json`.

## Executed checks

**PASS:** all cells of all five notebooks executed in fresh Python kernels.
Both experimental maps were refitted from the supplied text spectra:
12,100 successful single-resonance fits per scan, 24,200 fits in total.
Convergence is a computational diagnostic, not a guarantee of physical accuracy.
The complete all-cases oracle search covered 56 measurements with 630 parameter
combinations each. All required manuscript exports were generated.

**PASS:** the same full execution and numerical checks succeeded after copying
the package to a differently named directory containing a space, excluding Git
metadata, caches and previous outputs. Every kernel confirmed that all seven
scientific modules were imported from that portable copy. The second run took
approximately 4.6 minutes on the verification machine.

**PASS:** the default simulation used the exact 489-pixel manual mask supplied
with the revised manuscript. Projection RMS errors were 0.171443142 mT for graph
cut and 0.227923478 mT after refinement. Vector AC RMS errors were
0.244320398, 0.185750963 and 0.301689005 mT for Bx, By and Bz. Numerical checks
also covered maximum errors and maximum truth amplitudes.

**PASS:** the full experimental fit maps, graph-cut and refined projections,
validity mask, manual seed mask and vector fields agreed with the previously
verified experimental arrays. The default path recomputed these arrays and did
not use `reference_results/` as a fit cache. The image transform and Fourier
parameters matched the revised manuscript; reconstructed means were zero.

**PASS:** the four-case fixed-parameter table, cropped-window diagnostic and
all 56 oracle-optimal configurations matched the numerical reference tables.
The single-resonance model-error table reproduced the rounded manuscript values.
Analytic cubic roots agreed with numerical diagonalization within 1e-12 GHz.

**PASS:** the coordinate-based experimental annotation produced the exact saved
90-pixel mask. Saved-mode guards did not invoke interactive annotation or replace
the loaded mask. Positive, negative and erase annotation callbacks were exercised,
and editing an annotator did not mutate the original input mask.

**PASS:** a syntax-tree comparison established unchanged executable scientific
code in the six copied research helpers, apart from the portable default data
directory in `helpers.py`. Hashes confirmed that all 21 private source/input files
used for this assembly remained byte-for-byte unchanged. Scientific acquisition
metadata were preserved; only `operator` and `serial_number` were omitted from
the public JSON copies.

**PASS:** root notebooks had no saved execution output, Python modules parsed,
notebook schemas validated, and no machine-specific source paths remained.
Input hashes were unchanged after execution. Side-by-side visual inspection
covered the projection comparison, vector-error plot, simulation seed map,
experimental vector reconstruction, measured spectrum grid and a Hamiltonian
shift plot. Expected panels, scales and annotations were present.

## Numerical tolerances and scope

The verifier allows 0.00002 mT absolute tolerance for projection and simulation
field metrics, and 0.00002 mT for experimental projected arrays. Experimental
vector arrays allow 2e-8 T absolute tolerance with a small relative tolerance.
Benchmark tables use stricter floating-point comparisons. These thresholds do
not alter the algorithms, annotations, exported values or manuscript rounding.

Notebook output PDFs are compared visually and numerical fields are compared
directly; regenerated PDF bytes are not expected to be identical because of
timestamps and rendering details. The appendix Hamiltonian exports preserve
the equations and physical parameters; raster resolution and layout can differ.

Verification used the existing scientific environment, not a newly installed
environment or a different operating system. Optimizer roundoff and nearly tied
oracle minima may vary across numerical platforms. Actual mouse interaction in
a Jupyter frontend was not tested; its callbacks and guarded control flow were.
The physical assumptions concerning seeds, single-resonance conversion and the
experimental NV/image frame are not independently calibrated by these software
checks.
