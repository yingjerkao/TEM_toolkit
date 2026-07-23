"""Tests for the RTE2 numerical Q0 finder.

We do NOT test that the fit reproduces LaTe2 — that's Task 8's job.
Here we only test that the χ-scan mechanism itself is correct, using
the RTE3 defaults as a stand-in TB model with known χ structure.
"""
import numpy as np
import pytest
from tem_cdw.hamiltonian import TBParams
from tem_cdw.models.rte2 import RTE2Model
from tem_cdw.models.base import Q0Result


def test_find_Q0_returns_Q0Result_with_diagnostic_arrays():
    m = RTE2Model(params=TBParams())  # borrow RTE3 defaults for the mechanism test
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    assert isinstance(result, Q0Result)
    assert result.q_grid.size == 25
    assert result.chi_axial.size == 25
    assert result.chi_diagonal.size == 25


def test_find_Q0_returns_at_least_one_peak():
    m = RTE2Model(params=TBParams())
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    assert len(result.peaks) >= 1


def test_find_Q0_peaks_are_local_maxima():
    m = RTE2Model(params=TBParams())
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    for qx, qy in result.peaks:
        assert qy == 0.0                        # axial
        assert 0.0 < qx < 2 * np.pi             # inside grid range


def test_find_Q0_peaks_sorted_by_q():
    m = RTE2Model(params=TBParams())
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    qs = [q[0] for q in result.peaks]
    assert qs == sorted(qs)
