"""RTE3 (LaTe₃) material model — Alekseev PRB 110, 205103 (2024) defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..hamiltonian import TBParams
from .base import MaterialModel, Q0Result


@dataclass
class RTE3Model(MaterialModel):
    """Analytic diagonal-nesting model. Q₀ = (2kF, 2kF)."""
    params: TBParams = field(default_factory=TBParams)
    name: ClassVar[str] = "RTE3"

    def find_Q0(self, T: float = 0.026, nk_bz: int = 200, nk_q: int = 50) -> Q0Result:
        qx, qy = self.params.Q0()
        return Q0Result(peaks=[(qx, qy)])
