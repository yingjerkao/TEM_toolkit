"""
Two-orbital (px, py) tight-binding Hamiltonian on the square lattice.

Eq. (2) of Alekseev et al., PRB 110, 205103 (2024):

    h(k) = [ 2 t_sigma cos(kx) - 2 t_pi cos(ky)    2 t_d sin(kx) sin(ky)   ]
           [ 2 t_d sin(kx) sin(ky)                  2 t_sigma cos(ky) - 2 t_pi cos(kx) ]

Default parameters (eV) follow Ref. [30]:
    t_sigma = 2.0,  t_pi = 0.37,  t_d = 0.16,  mu = -1.53
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TBParams:
    """Tight-binding parameters in eV."""

    t_sigma: float = 2.0
    t_pi: float = 0.37
    t_d: float = 0.16
    mu: float = -1.53

    def kF(self) -> float:
        """Fermi wavevector from mu = 2 t_sigma cos(kF), Eq. (3).

        Valid only in the t_pi = t_d = 0 limit; used to define Q0.
        """
        return float(np.arccos(self.mu / (2.0 * self.t_sigma)))

    def Q0(self) -> tuple[float, float]:
        """Nesting vector Q0 = (2 kF, 2 kF)."""
        kf = self.kF()
        return (2.0 * kf, 2.0 * kf)


def h_k(kx: np.ndarray, ky: np.ndarray, p: TBParams) -> np.ndarray:
    """Tight-binding Hamiltonian h(k), shape (..., 2, 2).

    kx, ky may be arbitrary broadcast-compatible arrays.
    Returns array with last two axes the orbital indices (px, py).
    """
    kx = np.asarray(kx)
    ky = np.asarray(ky)
    h = np.zeros(kx.shape + (2, 2), dtype=np.float64)
    h[..., 0, 0] = 2.0 * p.t_sigma * np.cos(kx) - 2.0 * p.t_pi * np.cos(ky)
    h[..., 1, 1] = 2.0 * p.t_sigma * np.cos(ky) - 2.0 * p.t_pi * np.cos(kx)
    off = 2.0 * p.t_d * np.sin(kx) * np.sin(ky)
    h[..., 0, 1] = off
    h[..., 1, 0] = off
    return h


def diagonalize(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigvals, eigvecs) of a stack of Hermitian 2x2 matrices.

    eigvals.shape == h.shape[:-1]                  (last axis: band index)
    eigvecs.shape == h.shape                       (columns: eigenvectors)
    a^alpha_n(k) = eigvecs[..., alpha, n]          (orbital alpha, band n)

    Uses np.linalg.eigh which is well-vectorized.
    """
    w, v = np.linalg.eigh(h)
    return w, v


def bands_and_projectors(
    kx: np.ndarray, ky: np.ndarray, p: TBParams
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: return (epsilon[..., n], a[..., alpha, n])."""
    return diagonalize(h_k(kx, ky, p))


def make_bz_grid(nk: int, endpoint: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Uniform grid on [0, 2 pi)^2 of shape (nk, nk).

    Returns (KX, KY) meshgrid arrays.
    """
    k = np.linspace(0.0, 2.0 * np.pi, nk, endpoint=endpoint)
    KX, KY = np.meshgrid(k, k, indexing="ij")
    return KX, KY
