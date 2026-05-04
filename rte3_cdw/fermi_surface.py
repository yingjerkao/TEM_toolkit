"""
Fermi surfaces with and without CDW order, following Appendix B of
Alekseev et al.

In the presence of a single-Q CDW with order parameters
    Delta_Q = Delta^0 sigma^0 + Delta^z sigma^z
(plus optionally Delta^x, Delta^y), the mean-field Hamiltonian couples
states at k, k+Q, k-Q.  Eq. (B2) writes a 6x6 (3-block-x-2-orbital)
Hamiltonian:

    H_k = diag(h_{k-Q}, h_k, h_{k+Q})  +  off-diagonal blocks Delta_Q.

Diagonalizing for each k gives 6 bands; the spectral weight in the first
BZ is the projection onto the central (k) block.
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import TBParams, h_k


def cdw_order_matrix(D0: complex = 0.0, Dx: complex = 0.0,
                     Dy: complex = 0.0, Dz: complex = 0.0) -> np.ndarray:
    """Build Delta_Q = D0 sigma^0 + Dx sigma^x + Dy sigma^y + Dz sigma^z."""
    s0 = np.eye(2, dtype=np.complex128)
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    return D0 * s0 + Dx * sx + Dy * sy + Dz * sz


def H_cdw_k(
    kx: np.ndarray,
    ky: np.ndarray,
    Qx: float,
    Qy: float,
    Delta: np.ndarray,
    p: TBParams,
) -> np.ndarray:
    """Build the 6x6 (3-replica-x-2-orbital) CDW mean-field Hamiltonian
    of Eq. (B2), at each k.

    Parameters
    ----------
    kx, ky : broadcast-compatible arrays of momentum components
    Qx, Qy : CDW wavevector
    Delta  : 2x2 complex matrix Delta_Q (orbital space)
    p      : TBParams

    Returns
    -------
    H : array of shape (..., 6, 6), Hermitian.
        Block ordering: [k-Q, k, k+Q].
    """
    kx = np.asarray(kx, dtype=np.float64)
    ky = np.asarray(ky, dtype=np.float64)
    h_kmQ = h_k(kx - Qx, ky - Qy, p).astype(np.complex128)
    h_kk = h_k(kx, ky, p).astype(np.complex128)
    h_kpQ = h_k(kx + Qx, ky + Qy, p).astype(np.complex128)

    H = np.zeros(kx.shape + (6, 6), dtype=np.complex128)
    H[..., 0:2, 0:2] = h_kmQ
    H[..., 2:4, 2:4] = h_kk
    H[..., 4:6, 4:6] = h_kpQ

    Dh = Delta.conj().T   # Delta^dagger

    # Eq. (B2): off-diagonal couplings -- Delta connects k -> k+Q, Delta^dagger connects k -> k-Q.
    # The matrix structure in (B2) reads:
    #   row "k-Q" couples to "k" via Delta^dagger (i.e. <k-Q| Delta^dagger |k> term).
    #   row "k" couples to "k-Q" via Delta and to "k+Q" via Delta^dagger.
    #   row "k+Q" couples to "k" via Delta.
    H[..., 0:2, 2:4] = Dh
    H[..., 2:4, 0:2] = Delta
    H[..., 2:4, 4:6] = Dh
    H[..., 4:6, 2:4] = Delta

    return H


def fermi_surface_spectral_weight(
    kx: np.ndarray,
    ky: np.ndarray,
    Qx: float,
    Qy: float,
    Delta: np.ndarray,
    p: TBParams,
    broadening: float = 0.005,
) -> np.ndarray:
    """Compute the spectral weight on the first-BZ (k-block) at the Fermi level
    for the CDW-reconstructed Hamiltonian.

    Returns A(k, omega=0) projected onto the central |k> block:
        A(k) = (1/pi) sum_n |<k | n_k>|^2 * eta / [(0 - E_n)^2 + eta^2]
    where the sum is over the 6 CDW bands and the projector keeps only
    components 2:4 (i.e., the |k> orbital block).
    """
    H = H_cdw_k(kx, ky, Qx, Qy, Delta, p)
    E, V = np.linalg.eigh(H)                     # E: (..., 6), V: (..., 6, 6)
    # Spectral weight on central block (rows 2:4 of V)
    proj = (np.abs(V[..., 2:4, :]) ** 2).sum(axis=-2)   # (..., 6)
    # Lorentzian at omega=0 minus E
    A = (broadening / np.pi) / (E ** 2 + broadening ** 2)
    return (proj * A).sum(axis=-1)
