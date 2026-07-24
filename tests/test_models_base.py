import numpy as np
import pytest
from tem_cdw.hamiltonian import TBParams
from tem_cdw.models.base import MaterialModel, Q0Result


def test_q0result_construction():
    r = Q0Result(
        peaks=[(1.0, 0.0), (2.0, 0.0)],
        q_grid=np.array([0.0, 1.0, 2.0]),
        chi_axial=np.array([0.1, 0.9, 0.3]),
        chi_diagonal=np.array([0.05, 0.4, 0.2]),
    )
    assert r.peaks == [(1.0, 0.0), (2.0, 0.0)]
    assert r.q_grid.shape == (3,)


def test_material_model_is_abstract():
    with pytest.raises(TypeError):
        MaterialModel()


def test_subclass_must_implement_find_Q0():
    class Incomplete(MaterialModel):
        name = "x"
        params = TBParams()
    with pytest.raises(TypeError):
        Incomplete()
