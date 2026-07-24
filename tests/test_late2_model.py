import numpy as np
import pytest
from tem_cdw.models import LaTe2Model
from tem_cdw.units import astar_from_q


def test_late2_factory_loads_fitted_params():
    m = LaTe2Model()
    # Params should differ from RTE3 defaults; sanity check.
    assert not (m.params.t_sigma == 2.0 and m.params.t_pi == 0.37
                and m.params.t_d == 0.16 and m.params.mu == -1.53), (
        "LaTe2 fit landed exactly on RTE3 defaults — suspicious."
    )


def test_late2_find_Q0_reproduces_qlow():
    """The 2-orbital+t_2 model reliably produces one axial χ peak at
    q_low ≈ 0.15 a*, matching the TEM incommensurate CDW wavevector.

    A "second peak" at higher q is grid-resolution-dependent: at moderate
    nk_bz values, `find_peaks` picks up small shoulder ripples that meet the
    2% prominence floor; at nk_bz ≥ 200 only the q_low peak survives with
    real prominence. So we do NOT assert a q_high value here — that would
    lock in a numerical artifact, not physics.

    The physics finding: the 2-orbital+t_2 form captures q_low but has no
    real structure at the TEM q_high ≈ 0.25 a*. Documented as a Phase A
    conclusion in `docs/superpowers/specs/2026-07-22-rte2-cdw-phase-a-design.md`
    § "Failure mode we surface explicitly" and in the walkthrough summary.
    """
    m = LaTe2Model()
    result = m.find_Q0(T=0.026, nk_bz=200, nk_q=80)
    peaks_astar = sorted(astar_from_q(q[0]) for q in result.peaks)
    assert len(peaks_astar) >= 1, f"Expected at least one axial peak, got 0"
    q_low = peaks_astar[0]
    assert q_low == pytest.approx(0.15, abs=0.03), f"q_low={q_low}"


def test_late2_diagonal_below_axial():
    m = LaTe2Model()
    result = m.find_Q0(T=0.026, nk_bz=80, nk_q=40)
    # Diagonal control should be smaller than axial peak at their maxima.
    assert result.chi_diagonal.max() < result.chi_axial.max(), (
        "Diagonal χ exceeds axial χ — CDW is not axial after all."
    )
