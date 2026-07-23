import numpy as np
from tem_cdw.hamiltonian import TBParams
from scripts.fit_late2 import kF_axial, chi_peaks_along_a, objective


def test_kF_axial_finite_for_default():
    p = TBParams()
    kf = kF_axial(p, nk=200)
    assert 0.0 < kf < np.pi


def test_chi_peaks_returns_arrays_and_peaks():
    p = TBParams()
    peaks_q, q_grid, chi_grid = chi_peaks_along_a(p, T=0.026, nk_bz=60, nk_q=25)
    assert q_grid.shape == chi_grid.shape == (25,)
    assert isinstance(peaks_q, list)


def test_objective_finite_at_default():
    x = np.array([2.0, 0.37, 0.16, -1.53])
    val = objective(x, w_FS=1.0, w_q=1.0, nk_bz=60, nk_q=25)
    assert np.isfinite(val)


def test_objective_returns_large_for_bad_params():
    # μ way outside the band → no FS → sentinel value
    x = np.array([2.0, 0.37, 0.16, -10.0])
    val = objective(x, w_FS=1.0, w_q=1.0, nk_bz=60, nk_q=25)
    assert val >= 1e5
