"""Snapshot the current RTE3 physics numerics. This test locks in the
Alekseev conventions (h(k), χ tensor, GL coefficients) so the upcoming
tem_cdw/ refactor is provably numerics-preserving.

Baseline values recorded from the current main via Task 1 Step 4.
"""
import numpy as np
import pytest

# ---- Baseline values recorded in Task 1 Step 4 ----
EXPECTED_LOWER_BAND_EPS = [3.1833988038937733, 2.8126786411969174, 1.6082193268636156]
EXPECTED_DIAG_AA = 16.554309760388346
EXPECTED_A_MINUS = np.float64(-14.689106902970071)


# Temporary shim for Task 1 → Task 2 window. After Task 2 renames rte3_cdw
# to tem_cdw and deletes the old package, delete the except-branch below.
def _get_module():
    """Import whichever package name is currently in use."""
    try:
        import tem_cdw as m  # after Task 2
        return m
    except ImportError:
        import rte3_cdw as m  # before Task 2
        return m


def test_hamiltonian_eigenvalues():
    m = _get_module()
    p = m.hamiltonian.TBParams()
    kx = np.array([0.1, 0.5, 1.0])
    ky = np.array([0.2, 0.4, 0.9])
    eps, _ = m.hamiltonian.bands_and_projectors(kx, ky, p)
    np.testing.assert_allclose(eps[..., 0], np.asarray(EXPECTED_LOWER_BAND_EPS), rtol=1e-10)


def test_susceptibility_at_Q0():
    m = _get_module()
    p = m.hamiltonian.TBParams()
    Qx, Qy = p.Q0()
    chi = m.susceptibility.susceptibility_tensor(Qx, Qy, T=0.026, p=p, nk=100)
    diag_aa = m.susceptibility.chi_combinations(chi)["diag_aa"]
    assert diag_aa == pytest.approx(EXPECTED_DIAG_AA, rel=1e-8)


def test_gl_quadratic_a_minus_at_Q0():
    m = _get_module()
    p = m.hamiltonian.TBParams()
    Qx, Qy = p.Q0()
    qc = m.gl_theory.quadratic_coeffs(Qx, Qy, T=0.026, p=p, g=1.0, nk=100)
    a_minus, _, _, _ = qc.diagonalize()
    assert a_minus == pytest.approx(EXPECTED_A_MINUS, rel=1e-8)
