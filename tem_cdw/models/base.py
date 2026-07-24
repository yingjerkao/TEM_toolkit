"""MaterialModel ABC. Each concrete subclass bundles a `TBParams`
instance with a strategy for locating the CDW wavevector Q₀."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..hamiltonian import TBParams


@dataclass
class Q0Result:
    """Bundle of Q₀ candidates plus diagnostic scan arrays.

    Attributes
    ----------
    peaks : list of (Qx, Qy) tuples, sorted by q along a*.
        For RTE3 this is a single analytic point.
        For RTE2 this is up to two axial peaks (q_low, q_high).
    q_grid : 1D array of q values (radians) at which the diagnostic
        χ-scan was evaluated. Empty for analytic (RTE3) case.
    chi_axial, chi_diagonal : 1D arrays of χ(Q,T) along (q, 0) and (q, q)
        respectively, over q_grid. Empty for analytic case.
    """
    peaks: list[tuple[float, float]]
    q_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_axial: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_diagonal: np.ndarray = field(default_factory=lambda: np.array([]))


class MaterialModel(ABC):
    """Abstract material model. Subclasses provide `params` and `find_Q0`."""

    name: ClassVar[str] = "abstract"
    params: TBParams

    @abstractmethod
    def find_Q0(self, T: float, nk_bz: int = 200, nk_q: int = 50) -> Q0Result:
        """Return the CDW wavevector candidate(s) for this material at
        temperature T (eV). See Q0Result docstring for return schema."""
        ...
