"""
Multi-orbital charge susceptibility tensor, Eq. (4):

    chi^{beta gamma}_{alpha delta}(Q, T, mu)
        = - sum_{k, n, m} a^alpha_n(k) a^{beta *}_n(k)
                          a^gamma_m(k+Q) a^{delta *}_m(k+Q)
          * [ f(xi_{kn}) - f(xi_{k+Q, m}) ] / (xi_{kn} - xi_{k+Q, m})

with xi_{kn} = epsilon_{kn} - mu, and f the Fermi-Dirac function.

The "0/0" limit when xi_{kn} == xi_{k+Q, m} is handled with the
analytical limit  -f'(xi) = beta * f(xi) * (1 - f(xi)).

Indexing conventions
--------------------
We return chi as a numpy array of shape (2, 2, 2, 2) indexed as
    chi[alpha, delta, beta, gamma]
i.e. the *lower* (raised) indices come first, then the *upper*.
This matches chi^{beta gamma}_{alpha delta}.

Sums used in the GL theory (Eqs. 24, 25):
    chi_diag_aa = chi[0,0,0,0] + chi[1,1,1,1]   (alpha=delta=beta=gamma=same)
    chi_diag_ab = chi[0,0,1,1] + chi[1,1,0,0]
    chi_offd_aa = chi[0,1,0,1] + chi[1,0,1,0]
    chi_offd_ab = chi[0,1,1,0] + chi[1,0,0,1]
    chi_lambda  = chi[0,0,0,1] + chi[1,1,1,0] + chi[0,0,1,0] + chi[1,1,0,1]
"""

from __future__ import annotations

import numpy as np

from .hamiltonian import TBParams, bands_and_projectors, make_bz_grid


def fermi(x: np.ndarray, T: float) -> np.ndarray:
    """Fermi-Dirac function with safe handling of large |x/T|."""
    # We assume T > 0 throughout this project (no T = 0 limit is taken
    # numerically; the GL theory is computed at finite T anyway).
    return 0.5 * (1.0 - np.tanh(0.5 * x / T))


def _lindhard_kernel(
    xi_k: np.ndarray, xi_kQ: np.ndarray, T: float, eps: float = 1e-10
) -> np.ndarray:
    """Compute  [f(xi_k) - f(xi_kQ)] / (xi_k - xi_kQ)  with the L'Hopital
    limit when xi_k ~ xi_kQ.

    All inputs broadcast.  Returns a real array.
    """
    f_k = fermi(xi_k, T)
    f_kQ = fermi(xi_kQ, T)
    dxi = xi_k - xi_kQ
    # L'Hopital limit of [f(a) - f(b)] / (a - b) as b -> a is f'(a) = -beta f (1-f).
    # Note the sign: lim = -(1/T) f(xi) (1 - f(xi)).
    deriv = -(1.0 / T) * f_k * (1.0 - f_k)
    safe = np.where(np.abs(dxi) > eps, (f_k - f_kQ) / np.where(np.abs(dxi) > eps, dxi, 1.0), deriv)
    return safe


def susceptibility_tensor(
    Qx: float,
    Qy: float,
    T: float,
    p: TBParams,
    nk: int = 200,
) -> np.ndarray:
    """Compute the full 2x2x2x2 susceptibility tensor at wavevector Q.

    Returns
    -------
    chi : np.ndarray, shape (2, 2, 2, 2)
        chi[alpha, delta, beta, gamma] = chi^{beta gamma}_{alpha delta}(Q).

    Notes
    -----
    Vectorized over the BZ grid (nk x nk).  No explicit k loop.
    The overall sign is the convention of Eq. (4): chi has a leading minus,
    which we include here.
    """
    KX, KY = make_bz_grid(nk)
    eps_k, a_k = bands_and_projectors(KX, KY, p)        # shapes (nk,nk,2), (nk,nk,2,2)
    eps_kQ, a_kQ = bands_and_projectors(KX + Qx, KY + Qy, p)

    xi_k = eps_k - p.mu                                  # (nk, nk, 2)  band axis last
    xi_kQ = eps_kQ - p.mu

    # We need pairs (n, m) of bands.  Add explicit axes so xi_k has band axis "n"
    # and xi_kQ has band axis "m".
    xi_k_n = xi_k[..., :, None]                          # (nk, nk, n=2, m=1)
    xi_kQ_m = xi_kQ[..., None, :]                        # (nk, nk, n=1, m=2)

    # Lindhard kernel L_{nm}(k) = [f(xi_kn) - f(xi_{k+Q, m})] / (xi_kn - xi_{k+Q,m})
    L = _lindhard_kernel(xi_k_n, xi_kQ_m, T)              # (nk, nk, 2, 2)

    # Eigenvector projectors:
    # a_k[..., alpha, n], conj(a_k)[..., beta, n], a_kQ[..., gamma, m], conj(a_kQ)[..., delta, m]
    # Combine into M[..., alpha, beta, n] = a_k[..., alpha, n] * conj(a_k)[..., beta, n]
    # and        N[..., gamma, delta, m] = a_kQ[..., gamma, m] * conj(a_kQ)[..., delta, m]
    M = a_k[..., :, None, :] * np.conj(a_k)[..., None, :, :]      # (nk, nk, alpha, beta, n)
    N = a_kQ[..., :, None, :] * np.conj(a_kQ)[..., None, :, :]    # (nk, nk, gamma, delta, m)

    # chi^{beta gamma}_{alpha delta} = - (1/Nk) sum_{k, n, m} M[a,b,n] * N[g,d,m] * L_{nm}(k)
    # Contract n with axis -2 of L and m with axis -1.
    # First: tensor over (n, m) using L. Build T[..., a, b, g, d] = sum_{n,m} M[a,b,n] L[n,m] N[g,d,m].
    # Do it in two einsum steps to keep memory modest.
    # Step 1: P[..., a, b, m] = sum_n M[..., a, b, n] L[..., n, m]
    P = np.einsum("xyabn,xynm->xyabm", M, L)                       # (nk, nk, 2, 2, 2)
    # Step 2: T[..., a, b, g, d] = sum_m P[..., a, b, m] N[..., g, d, m]
    Tens = np.einsum("xyabm,xygdm->xyabgd", P, N)                  # (nk, nk, 2, 2, 2, 2)

    # Convention: chi = integral dk * (...) over BZ, with NO (2 pi)^2 divisor.
    # This matches the magnitudes plotted in Figs. 2, 3 of the paper.
    # Discrete approximation: integral dk ~ ((2 pi)^2 / Nk) * sum_k.
    Nk = KX.size
    bz_measure = (2.0 * np.pi) ** 2 / Nk
    chi_abgd = -bz_measure * Tens.sum(axis=(0, 1)).real            # (a, b, g, d)

    # Reorder axes from (alpha, beta, gamma, delta) to
    # (alpha, delta, beta, gamma) so that chi[a,d,b,g] == chi^{bg}_{ad}.
    chi = np.transpose(chi_abgd, (0, 3, 1, 2))
    return chi


# ----------------------------------------------------------------------
# Convenience wrappers for the symmetric combinations used in the GL theory
# ----------------------------------------------------------------------

def chi_combinations(chi: np.ndarray) -> dict[str, float]:
    """Extract the symmetric combinations entering Eqs. (24), (25).

    Returns a dict with keys:
        'diag_aa'  = chi^{11}_{11} + chi^{22}_{22}        (peaks at Q0)
        'diag_ab'  = chi^{22}_{11} + chi^{11}_{22}        (small when t_d small)
        'offd_aa'  = chi^{12}_{12} + chi^{21}_{21}        (peaks at zero Q)
        'offd_ab'  = chi^{21}_{12} + chi^{12}_{21}        (small when t_d small)
        'lambda'   = chi^{12}_{11} + chi^{21}_{22} + chi^{12}_{12_off?}...
                     actually: chi^{12}_{11} + chi^{21}_{22} + chi^{21}_{11} + chi^{12}_{22}
                     (Eq. 25 with sign convention)
    """
    # chi[alpha, delta, beta, gamma]
    diag_aa = chi[0, 0, 0, 0] + chi[1, 1, 1, 1]
    diag_ab = chi[0, 0, 1, 1] + chi[1, 1, 0, 0]
    offd_aa = chi[0, 1, 0, 1] + chi[1, 0, 1, 0]
    offd_ab = chi[0, 1, 1, 0] + chi[1, 0, 0, 1]
    # Eq. (25):  lambda = -(chi^{12}_{11} + chi^{21}_{22} + chi^{21}_{11} + chi^{12}_{22})
    # Here we return the *positive* sum; the sign is applied in gl_theory.
    lam_sum = (
        chi[0, 0, 0, 1]   # chi^{12}_{11}: alpha=1, delta=1, beta=1, gamma=2  -> indices 0,0,0,1
        + chi[1, 1, 1, 0]   # chi^{21}_{22}
        + chi[0, 0, 1, 0]   # chi^{21}_{11}: alpha=1, delta=1, beta=2, gamma=1
        + chi[1, 1, 0, 1]   # chi^{12}_{22}
    )
    return {
        "diag_aa": float(diag_aa),
        "diag_ab": float(diag_ab),
        "offd_aa": float(offd_aa),
        "offd_ab": float(offd_ab),
        "lambda_sum": float(lam_sum),
    }


def chi_scalar_at_Q(Qx: float, Qy: float, T: float, p: TBParams, nk: int = 200) -> float:
    """Fourfold-invariant combination chi(Q, T, mu) of Sec. II,
    chi = chi^{11}_{11} + chi^{22}_{22}.  Used to plot Fig. 2 (upper panel)
    and Fig. 3.
    """
    chi = susceptibility_tensor(Qx, Qy, T, p, nk=nk)
    return chi_combinations(chi)["diag_aa"]


def chi_tilde_at_Q(Qx: float, Qy: float, T: float, p: TBParams, nk: int = 200) -> float:
    """Fourfold-invariant tilde chi = chi^{12}_{12} + chi^{21}_{21}, Sec. II
    (Fig. 2, lower panel)."""
    chi = susceptibility_tensor(Qx, Qy, T, p, nk=nk)
    return chi_combinations(chi)["offd_aa"]
