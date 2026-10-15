"""Execute the public notebooks in fresh kernels and verify manuscript results."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = {
    'simulation': 'odmr_sign_unwrap_simulation_test.ipynb',
    'measurement': 'odmr_sign_unwrap_measurement_2026_05_25.ipynb',
    'benchmark': 'nv_reconstruction_benchmark.ipynb',
    'all-cases': 'nv_reconstruction_benchmark_all_cases.ipynb',
    'hamiltonian': 'nv_odmr_hamiltonian.ipynb',
}


def verify_inputs():
    for name in NOTEBOOKS.values():
        path = ROOT / name
        assert path.is_file(), f"Missing notebook: {name}"
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
    print("Notebook files and formats checked.", flush=True)


def execute(label):
    path = ROOT / NOTEBOOKS[label]
    nb = nbformat.read(path, as_version=4)
    # Confirm that imported scientific modules belong to this portable copy.
    nb.cells.append(nbformat.v4.new_code_cell('''from pathlib import Path
import importlib
for module_name in ('helpers', 'fits', 'odmr_initMAP', 'odmr_sign_unwrap',
                    'sim_odmr_field_helpers', 'odmr_measurement_helpers',
                    'nv_hamiltonian_helpers'):
    module = importlib.import_module(module_name)
    assert Path(module.__file__).resolve().parent == Path.cwd().resolve(), module_name
print('All scientific modules loaded from the public repository.')
'''))
    output = ROOT / 'outputs' / 'executed_notebooks'
    output.mkdir(parents=True, exist_ok=True)
    km = KernelManager(kernel_name='python3')
    # Use the Python running this script, without a user-specific kernelspec.
    km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
    client = NotebookClient(nb, km=km, timeout=1800,
                            resources={'metadata': {'path': str(ROOT)}})
    started = time.monotonic()
    print(f'Executing {path.name}', flush=True)
    try:
        client.execute(cwd=str(ROOT))
    finally:
        nbformat.write(nb, output / path.name)
        if km.has_kernel:
            km.shutdown_kernel(now=True)
    elapsed = round(time.monotonic()-started, 2)
    print(f'PASS {label}: all cells executed ({elapsed}s).', flush=True)
    return elapsed


def verify_results():
    import numpy as np
    import pandas as pd

    out = ROOT / 'outputs'
    ref = ROOT / 'reference_results'
    expected = json.loads((ref / 'simulation_manual_metrics.json').read_text())
    projection = pd.read_csv(out / 'simulation/projection_metrics.csv').set_index('method')
    for row in expected['projection']:
        # 0.00002 mT tolerance allows platform-dependent optimizer roundoff.
        np.testing.assert_allclose(projection.loc[row['method'], ['RMS_mT', 'max_error_mT']].to_numpy(float),
                                   [row['RMS'], row['max_error']], rtol=0, atol=2e-5)
    fields = pd.read_csv(out / 'simulation/vector_metrics.csv').set_index('quantity')
    for row in expected['field']:
        for key in ('AC RMS [mT]', 'max |AC error| [mT]', 'max |AC GT| [mT]'):
            np.testing.assert_allclose(fields.loc[row['quantity'], key], row[key], rtol=0, atol=2e-5)
    with np.load(out / 'simulation/simulation_results.npz', allow_pickle=False) as sim:
        original_mask = np.load(ROOT / 'saved_seed_masks/seed_mask_simulation.npy')
        np.testing.assert_array_equal(sim['seed_mask'], original_mask)
        assert np.count_nonzero(sim['seed_mask']) == expected['seed_pixels']
        assert sim['global_sign_gc'].item() == expected['global_alignment_gc']
        assert sim['global_sign_ref'].item() == expected['global_alignment_ref']

    path = out / 'sign_unwrap_2026-05-25-10-27-11'
    with np.load(path / 'sign_unwrap_results.npz', allow_pickle=False) as actual, \
            np.load(ref / 'measurement_results.npz', allow_pickle=False) as reference:
        for key in ('seed_mask', 'valid_mask'):
            np.testing.assert_array_equal(actual[key], reference[key])
        for key in ('A_mT', 'A_compare_mT', 'B_gc_mT', 'B_ref_mT'):
            np.testing.assert_allclose(actual[key], reference[key], rtol=1e-8, atol=2e-5)
        for key in ('Bx_T', 'By_T', 'Bz_T'):
            np.testing.assert_allclose(actual[key], reference[key], rtol=1e-7, atol=2e-8)
        provenance = json.loads(actual['provenance_json'].item())
        assert provenance['n_peaks'] == 1
        assert provenance['image_axis_signs'] == [-1, 1, 1]
        assert provenance['fourier_kwargs'] == dict(lambda0=1e-8, p=10, alpha=1.0, dc_mode='zero')
        for key in ('Bx_T', 'By_T', 'Bz_T'):
            assert abs(float(np.nanmean(actual[key]))) < 1e-12
    for date in ('2026-05-25-10-27-11', '2026-05-25-23-05-22'):
        fits = pd.read_csv(path / f'fits_{date}.csv')
        assert len(fits) == 12100 and fits.success.all(), date

    pd.testing.assert_frame_equal(
        pd.read_csv(out / 'benchmark/fixed_protocol_metrics.csv'),
        pd.read_csv(ref / 'benchmark_fixed_metrics.csv'),
        check_exact=False, rtol=1e-9, atol=1e-9)
    pd.testing.assert_frame_equal(
        pd.read_csv(out / 'simulation/finite_fov_metrics.csv'),
        pd.read_csv(ref / 'finite_fov_metrics.csv'),
        check_exact=False, rtol=1e-9, atol=1e-9)
    actual = pd.read_csv(out / 'benchmark_all_cases/oracle_best_parameters.csv')
    reference = pd.read_csv(ref / 'all_cases_oracle_metrics.csv')
    assert len(actual) == 56
    pd.testing.assert_frame_equal(actual, reference, check_exact=False, rtol=1e-8, atol=1e-10)
    # Check the displayed averages against the rounded manuscript tables.
    avg = pd.read_csv(out / 'benchmark_all_cases/oracle_clean_noisy_summary.csv').set_index('measurement')
    for label, values in [('clean', [10.75, 2.13, 1.296, 1.035, 1.464]),
                          ('noisy', [11.93, .68, 1.613, 1.283, 1.892])]:
        np.testing.assert_allclose(
            avg.loc[label, ['p_mean','alpha_k0_mean','RMSE_Bx_mean_mT',
                            'RMSE_By_mean_mT','RMSE_Bz_mean_mT']].to_numpy(float),
            values, rtol=0, atol=.0051)
    model = pd.read_csv(out / 'hamiltonian/single_resonance_model_error.csv')
    np.testing.assert_allclose(model[['estimated_lower_mT','estimated_upper_mT']].to_numpy(),
                               [[5.003,5.003],[4.638,5.369],[3.569,6.472]], rtol=0, atol=.00051)
    assert float((out / 'hamiltonian/analytic_roots_check.txt').read_text()) < 1e-12

    required = {
        'simulation': ['simu_NV_projection_lattice_abs.pdf','GT_graphcut.pdf','seeds.pdf',
                       'reconstruction3D_GT_refined.pdf','error_BxByBz.pdf'],
        'sign_unwrap_2026-05-25-10-27-11': ['comparaison_B_parallel.pdf','B_parallel_refined.pdf',
                       'Brec_full_and_zoom_quiver.pdf','Bpar_roi_zoom_quiver.pdf',
                       'scan_spectres_annexe.pdf','scan_spectres_roi.png','seeds_experiment.pdf'],
        'benchmark': ['gt_vs_reconstructed_2x4.pdf'],
        'benchmark_all_cases': ['source_maps.pdf'],
        'hamiltonian': ['shift_Db.png','shift_D2b.png'],
    }
    for directory, files in required.items():
        for name in files:
            assert (out / directory / name).stat().st_size > 1000, name
    verify_inputs()  # Default execution must not overwrite inputs or source notebooks.
    print('PASS: numerical reference checks and manuscript exports.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', choices=NOTEBOOKS, nargs='+', help='Run selected notebooks.')
    parser.add_argument('--check-inputs', action='store_true', help='Check hashes without executing.')
    parser.add_argument('--verify-only', action='store_true', help='Check already generated results.')
    args = parser.parse_args()
    verify_inputs()
    if args.check_inputs:
        return
    if args.verify_only:
        verify_results()
        return
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['MPLCONFIGDIR'] = str(ROOT / 'outputs' / '.matplotlib')
    os.environ['IPYTHONDIR'] = str(ROOT / 'outputs' / '.ipython')
    times = {label: execute(label) for label in (args.only or NOTEBOOKS)}
    if args.only is None:
        verify_results()
    report = dict(notebooks=times, numerical_reference_checks=args.only is None,
                  python=sys.version.split()[0], status='PASS')
    (ROOT / 'outputs' / 'verification.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
