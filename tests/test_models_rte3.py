import numpy as np
from tem_cdw.hamiltonian import TBParams
from tem_cdw.models.rte3 import RTE3Model


def test_default_params_match_alekseev():
    m = RTE3Model()
    assert m.params.t_sigma == 2.0
    assert m.params.t_pi == 0.37
    assert m.params.t_d == 0.16
    assert m.params.mu == -1.53
    assert m.name == "RTE3"


def test_find_Q0_returns_diagonal_2kF():
    m = RTE3Model()
    result = m.find_Q0(T=0.026)
    assert len(result.peaks) == 1
    kF = m.params.kF()
    qx, qy = result.peaks[0]
    assert np.isclose(qx, 2 * kF)
    assert np.isclose(qy, 2 * kF)
    assert result.q_grid.size == 0
