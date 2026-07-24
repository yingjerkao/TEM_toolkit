"""RTE2 (LaTe₂) material model — numerical χ(Q) scan along a* and diagonal.

The fitted TBParams for LaTe₂ are set by Task 8; this task uses a
placeholder (RTE3 defaults) so the χ-scan mechanism is testable in
isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import warnings

import numpy as np
from scipy.signal import find_peaks

from ..hamiltonian import TBParams
from ..susceptibility import susceptibility_tensor, chi_combinations
from .base import MaterialModel, Q0Result


# Placeholder defaults; Task 8 replaces these with the LaTe₂ fit.
LATE2_PLACEHOLDER = TBParams(t_sigma=2.0, t_pi=0.37, t_d=0.16, mu=-1.53)


@dataclass
class RTE2Model(MaterialModel):
    """Numerical axial-nesting model for LaTe₂."""
    params: TBParams = field(default_factory=lambda: LATE2_PLACEHOLDER)
    name: ClassVar[str] = "RTE2"

    def find_Q0(self, T: float = 0.026, nk_bz: int = 200, nk_q: int = 50) -> Q0Result:
        """Scan χ(Q) along Q = (q, 0) (axial) and Q = (q, q) (diagonal control).

        Returns up to the top-2 local maxima of χ_axial, sorted by q.
        Also returns the full q_grid and both χ curves for plotting.
        """
        two_pi = 2.0 * np.pi
        q_astar = np.linspace(0.02, 0.5, nk_q)
        q_grid = q_astar * two_pi

        chi_axial = np.empty(nk_q)
        chi_diagonal = np.empty(nk_q)
        for i, q in enumerate(q_grid):
            chi_a = susceptibility_tensor(q, 0.0, T, self.params, nk=nk_bz)
            chi_d = susceptibility_tensor(q, q,   T, self.params, nk=nk_bz)
            chi_axial[i]    = chi_combinations(chi_a)["diag_aa"]
            chi_diagonal[i] = chi_combinations(chi_d)["diag_aa"]

        # Local-maxima detection on the axial scan.
        prominence_floor = 0.02 * (chi_axial.max() - chi_axial.min())
        peaks_idx, _ = find_peaks(chi_axial, prominence=prominence_floor)

        if len(peaks_idx) == 0:
            warnings.warn(
                f"find_Q0 found no χ(Q) peaks along a*. "
                f"max={chi_axial.max():.3g} min={chi_axial.min():.3g}",
                stacklevel=2,
            )
            top2_sorted = []
        else:
            # Keep the two most prominent, then sort by q.
            top_by_height = sorted(peaks_idx, key=lambda i: chi_axial[i], reverse=True)[:2]
            top2_sorted = sorted(top_by_height, key=lambda i: q_grid[i])

        peaks = [(float(q_grid[i]), 0.0) for i in top2_sorted]

        return Q0Result(
            peaks=peaks,
            q_grid=q_grid,
            chi_axial=chi_axial,
            chi_diagonal=chi_diagonal,
        )


def LaTe2Model() -> RTE2Model:
    """Factory returning RTE2Model with the LaTe₂-fitted TB parameters.

    Params are frozen at fit time in tem_cdw/models/rte2_fit_report/
    fitted_params.npz (produced by scripts/fit_late2.py). Downstream
    code never re-runs the fit.
    """
    import pathlib
    npz_path = (
        pathlib.Path(__file__).parent / "rte2_fit_report" / "fitted_params.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"LaTe₂ fit not yet run. Execute `python -m scripts.fit_late2` "
            f"before calling LaTe2Model(). Expected file: {npz_path}"
        )
    data = np.load(npz_path)
    params = TBParams(
        t_sigma=float(data["t_sigma"]),
        t_pi=float(data["t_pi"]),
        t_d=float(data["t_d"]),
        t_2=float(data["t_2"]) if "t_2" in data.files else 0.0,
        mu=float(data["mu"]),
    )
    return RTE2Model(params=params)
