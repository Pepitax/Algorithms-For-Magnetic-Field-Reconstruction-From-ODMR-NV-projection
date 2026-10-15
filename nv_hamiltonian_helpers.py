"""NV spin-1 Hamiltonian and analytic roots, in GHz and Tesla.

Extracted from the appendix calculations of the research workflow.
"""

import numpy as np
Sx = 1 / np.sqrt(2) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
Sy = 1 / np.sqrt(2) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)

def H_over_h(Bx, By, Bz, D=2.87, E=0.005, gamma_e=28.0):
    """
    Full NV ground-state spin Hamiltonian divided by h.
    
    Bx, By, Bz are in Tesla and are expressed in the local NV frame.
    The result is in GHz.
    """
    return D * (Sz @ Sz) + E * (Sx @ Sx - Sy @ Sy) + gamma_e * (Bx * Sx + By * Sy + Bz * Sz)

def odmr_frequencies(Bx, By, Bz, D=2.87, E=0.005, gamma_e=28.0):
    """
    Returns the two transition frequencies nu_minus, nu_plus.
    
    We identify the 'bright' state as the eigenstate with the largest
    |ms=0> character, then compute the two energy gaps from this state.
    
    This matches the usual ODMR picture at low transverse field.
    At large transverse field, the ODMR contrast may become weak, but
    the energy gaps are still useful to visualize the Hamiltonian.
    """
    H = H_over_h(Bx, By, Bz, D=D, E=E, gamma_e=gamma_e)
    evals, evecs = np.linalg.eigh(H)
    ms0_weights = np.abs(evecs[1, :]) ** 2
    idx_0like = np.argmax(ms0_weights)
    E0like = evals[idx_0like]
    gaps = []
    for i in range(3):
        if i != idx_0like:
            gaps.append(abs(evals[i] - E0like))
    gaps = np.sort(gaps)
    return (gaps[0], gaps[1])

def analytic_eigenvalues(Bx, By, Bz, D=2.87, E=0.005, gamma_e=28.0):
    """
    Analytical eigenvalues of the full 3x3 Hamiltonian H/h.

    Bx, By, Bz are in Tesla and expressed in the local NV frame.
    Returns the three eigenvalues in GHz, sorted increasingly.
    """
    X = gamma_e * Bx
    Y = gamma_e * By
    Z = gamma_e * Bz
    rho = 1.0 / 3.0 * np.sqrt(D ** 2 + 3.0 * (E ** 2 + X ** 2 + Y ** 2 + Z ** 2))
    q = 2.0 * D ** 3 / 27.0 - 2.0 * D * E ** 2 / 3.0 + D * (X ** 2 + Y ** 2 - 2.0 * Z ** 2) / 3.0 - E * (X ** 2 - Y ** 2)
    arg = -q / (2.0 * rho ** 3)
    arg = np.clip(arg, -1.0, 1.0)
    lambdas = np.array([2.0 * D / 3.0 + 2.0 * rho * np.cos(1.0 / 3.0 * np.arccos(arg) - 2.0 * np.pi * k / 3.0) for k in range(3)])
    return np.sort(lambdas)

def odmr_frequencies_analytic(Bx, By, Bz, D=2.87, E=0.005, gamma_e=28.0):
    """
    ODMR frequencies computed from the analytical roots.

    For the field range used here, the |0>-like state is the lowest
    energy eigenstate, so we compute the two gaps from lambda[0].
    """
    lambdas = analytic_eigenvalues(Bx, By, Bz, D=D, E=E, gamma_e=gamma_e)
    nu_minus = lambdas[1] - lambdas[0]
    nu_plus = lambdas[2] - lambdas[0]
    return (nu_minus, nu_plus)
