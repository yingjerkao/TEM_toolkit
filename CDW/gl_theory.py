"""
Ginzburg-Landau theory for CDW order parameters with nontrivial orbital
character, following Sec. IV and Appendix A of Alekseev et al.

Order parameters
----------------
We work in the basis  Delta^i_Q0,  i = 0, x, y, z,  defined by
    Delta_Q = sum_k <psi^dagger_{k-Q} sigma^i psi_k>     [Eq. (11)]
with sigma^0 = identity, sigma^{1,2,3} = Pauli.  Equivalently,
    Delta^{px,px}_Q = (Delta^0 + Delta^z) / 2
    Delta^{py,py}_Q = (Delta^0 - Delta^z) / 2
    Delta^{px,py}_Q = (Delta^x + i Delta^y) / 2
    Delta^{py,px}_Q = (Delta^x - i Delta^y) / 2.

Quadratic action (Eq. 23):
    S^(2) = a0 |D^0|^2 + ax |D^x|^2 + ay |D^y|^2 + az |D^z|^2
            + 2 lambda Re(bar D^0 D^x)  + 2 tilde-lambda Re(bar D^y D^z),
with lambda-tilde = 0 for our model (Eq. A9 + symmetry of a^alpha_n(k)).

Coefficients (Eqs. 24, 25):
    a0/z = 2/g - (chi^11_11 + chi^22_22)  -/+  (chi^22_11 + chi^11_22)
    ax/y = 2/g - (chi^12_12 + chi^21_21)  -/+  (chi^21_12 + chi^12_21)
    lambda = -(chi^12_11 + chi^21_22 + chi^21_11 + chi^12_22)

After diagonalizing the (D^0, D^x) and (D^y, D^z) blocks, the leading
instability is in the "minus" linear combination:
    Delta^- = cos(theta) Delta^0 - sin(theta) Delta^x  (with small mixing)
which reduces to Delta^0 in the t_d = 0 limit.

Quartic coefficients (Eq. 27, in the basis {-, +, y, z}):
    S^(4) = b_- |D^-|^4 + b_z |D^z|^4
            + c'  (bar D^- D^z + bar D^z D^-)^2
            - c'' (bar D^- D^z - bar D^z D^-)^2
which we simplify (Eq. 28) using D^- = |D^-| e^{i alpha_-},  D^z = |D^z| e^{i alpha_z}:
    cross-term coefficient = 2 [c' + c'' + (c' - c'') cos(2 dalpha)] |D^-|^2 |D^z|^2

Coexistence (double-Delta) phase emerges when (Eq. 29):
    [(c' + c'') - |c' - c''|]^2 < b_- b_z.
The relative phase dalpha = alpha_z - alpha_- is set by Eq. (30):
    dalpha = +/- pi/2  if c' > c''  (B2u double-Delta, broken inversion + diag mirror)
    dalpha = 0, pi      if c' < c''  (B1g double-Delta, broken both diag mirrors)

The c' < c'' case requires a contribution from electron-phonon coupling
(Eq. 19, Eq. 20) and is not produced from the bare electronic GL theory,
which always gives c' > c''.

The quartic coefficients are computed from traces of four Green's functions
(Eq. A10):
    A_{ijkl} = (1/2) Tr[ G_k sigma^i G_{k+Q} sigma^j G_k sigma^k G_{k+Q} sigma^l ]
with the Matsubara sum done analytically (residues at the band poles of
G_k and G_{k+Q}).  We then sum over the BZ numerically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import TBParams, bands_and_projectors, make_bz_grid
from .susceptibility import fermi, susceptibility_tensor, chi_combinations


# ----------------------------------------------------------------------
# Quadratic coefficients
# ----------------------------------------------------------------------

@dataclass
class QuadraticCoeffs:
    """Coefficients of the quadratic GL action, basis {0, x, y, z}.

    a_i (i = 0, x, y, z) are the diagonal quadratic coefficients.
    lam couples Delta^0 and Delta^x (same Ag irrep).
    lam_tilde couples Delta^y and Delta^z (same B1g irrep) -- vanishes here.

    All are functions of T and the interaction g (entered via 2/g).
    """

    a0: float
    ax: float
    ay: float
    az: float
    lam: float
    lam_tilde: float

    def diagonalize(self) -> tuple[float, float, float, float]:
        """Return (a_minus, a_plus, ay, az) after diagonalizing the
        2x2 block in (Delta^0, Delta^x).  lam_tilde is assumed zero
        so the (y, z) block is already diagonal.
        """
        s = self.a0 + self.ax
        d = self.a0 - self.ax
        disc = np.sqrt(d * d + 4.0 * self.lam ** 2)
        a_minus = 0.5 * (s - disc)
        a_plus = 0.5 * (s + disc)
        return a_minus, a_plus, self.ay, self.az


def quadratic_coeffs_from_chi(chi: np.ndarray, g: float) -> QuadraticCoeffs:
    """Build QuadraticCoeffs from the susceptibility tensor at Q0.

    Uses Eqs. (24), (25). g is the attractive interaction strength.
    """
    c = chi_combinations(chi)
    # Eqs. (24):
    a0 = 2.0 / g - (c["diag_aa"] + c["diag_ab"])
    az = 2.0 / g - (c["diag_aa"] - c["diag_ab"])
    ax = 2.0 / g - (c["offd_aa"] + c["offd_ab"])
    ay = 2.0 / g - (c["offd_aa"] - c["offd_ab"])
    # Eq. (25):
    lam = -c["lambda_sum"]
    # lam_tilde = 0 by symmetry of real eigenvectors a^alpha_n(k).
    lam_tilde = 0.0
    return QuadraticCoeffs(a0=a0, ax=ax, ay=ay, az=az, lam=lam, lam_tilde=lam_tilde)


def quadratic_coeffs(
    Qx: float, Qy: float, T: float, p: TBParams, g: float, nk: int = 200
) -> QuadraticCoeffs:
    """Convenience: compute chi(Q,T,mu) and return GL coefficients."""
    chi = susceptibility_tensor(Qx, Qy, T, p, nk=nk)
    return quadratic_coeffs_from_chi(chi, g)


# ----------------------------------------------------------------------
# Quartic coefficients
# ----------------------------------------------------------------------
#
# We need the four-Green's-function trace, Eq. (A10):
#
#   A_{ijkl}(Q)
#     = (1/2) (1/beta) sum_{i omega_n}  integral_BZ dk
#         tr[ G_k(i omega_n) sigma^i G_{k+Q}(i omega_n) sigma^j
#             G_k(i omega_n) sigma^k G_{k+Q}(i omega_n) sigma^l ]
#
# The Matsubara sum is done analytically.  Working in the *band basis*
# of h_k and h_{k+Q}, define
#     U_k = matrix of eigenvectors of h_k    (columns are bands n)
#     U_kQ = matrix of eigenvectors of h_{k+Q} (columns are bands m)
# and band energies xi_{kn} = epsilon_{kn} - mu, etc.
# The Pauli matrices in this rotated basis become
#     sigma^i_band(k, k+Q) = U_k^dagger sigma^i U_{k+Q}        (off-diag block)
# But actually each G is purely diagonal in its own band basis; the rotation
# eats the U matrices.  Concretely:
#
# Let s^i_k = U_k^dagger sigma^i U_k   (acts in band space at k)
# Let t^i_kQ = U_kQ^dagger sigma^i U_k (mixes bands n at k with bands m at k+Q)
#
# Then  G_k sigma^i G_{k+Q} sigma^j G_k sigma^k G_{k+Q} sigma^l
# evaluated in the original orbital basis traces to the same result as
# evaluating in the *appropriate* mixed basis where each G is diagonal.
#
# The cleanest way: work with the orbital-basis Green's functions directly.
# Each G^{ab}_k(i omega) = sum_n a^a_n(k) a^{b*}_n(k) / (i omega - xi_{kn}).
#
# The product G_k sigma^i G_{k+Q} sigma^j G_k sigma^k G_{k+Q} sigma^l
# is a sum of 16 terms (one per choice of band indices n, m, n', m'),
# with the Matsubara sum reducing to a 4-pole sum that can be evaluated
# analytically.
#
# To keep the implementation tractable and verifiable, we derive a
# CLOSED FORM in the band-rotated representation.  Define mixed-basis
# matrix elements (coupling band n at k to band m at k+Q via Pauli sigma^i):
#
#     M^i_{nm}(k, Q) = sum_{ab} a^{a*}_n(k) sigma^i_{ab} a^b_m(k+Q)
#                    = [U_k^dagger sigma^i U_{k+Q}]_{nm}
#
# Then the trace evaluates to (after the analytic Matsubara sum):
#
#   A_{ijkl}(Q) = (1/2) integral_BZ dk
#       sum_{n n' m m'}
#         M^i_{nm} M^j*_{n'm}  (NO -- let's redo this carefully below)
#
# Actually, let us derive it step by step in the band basis.

def _build_mixed_pauli(a_k: np.ndarray, a_kQ: np.ndarray) -> np.ndarray:
    """Build M^i_{nm}(k) = [U_k^dagger sigma^i U_{k+Q}]_{nm}, i = 0,1,2,3.

    Parameters
    ----------
    a_k  : (..., 2 orbital, 2 band) eigenvector matrix at k
    a_kQ : same shape, at k+Q

    Returns
    -------
    M : (..., 4 pauli, 2 band_n, 2 band_m)
        with sigma indexed as (sigma_0, sigma_x, sigma_y, sigma_z).
    """
    sigma = np.zeros((4, 2, 2), dtype=np.complex128)
    sigma[0] = np.eye(2)
    sigma[1] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma[2] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma[3] = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    # M^i_{nm} = sum_{a,b} conj(a_k[a, n]) sigma^i_{ab} a_kQ[b, m]
    # einsum: ...an, iab, ...bm -> ...inm
    Uk_dag = np.conj(a_k)                        # ...an  (we use conj on k slot below)
    M = np.einsum("...an,iab,...bm->...inm", Uk_dag, sigma, a_kQ)
    return M


def _matsubara_box_sum(
    xi_n: np.ndarray, xi_np: np.ndarray, xi_m: np.ndarray, xi_mp: np.ndarray, T: float,
    eta: float = 1e-4,
) -> np.ndarray:
    """Matsubara sum
        S = (1/beta) sum_{iw} 1 / [(iw - xi_n)(iw - xi_{n'})(iw - xi_m)(iw - xi_{m'})]

    Computed analytically by summing Fermi-function residues.  Each pole xi_a
    contributes
        f(xi_a) / prod_{b != a} (xi_a - xi_b).

    For numerical stability when two poles are nearly degenerate, we add an
    imaginary regulator i*eta to half the poles and -i*eta to the other half.
    Specifically, treat (xi_n, xi_m) with +i eta and (xi_n', xi_m') with -i eta.
    The exact Matsubara sum is recovered as eta -> 0; for finite eta the answer
    is smooth and well-defined.

    For our application (4-pole sum with possible band degeneracies on
    measure-zero / 1D sets), eta ~ 1e-3 eV gives accurate BZ integrals
    (the regularization affects only a thin shell near the degeneracy).

    All inputs broadcast; returns a real array.
    """
    # Add small imaginary parts to lift degeneracies.  The choice (+, -, +, -)
    # for (xi_n, xi_n', xi_m, xi_m') is arbitrary; what matters is that no two
    # shifted poles coincide.
    a = xi_n + 1j * eta
    b = xi_np - 1j * eta
    c = xi_m + 2j * eta
    d = xi_mp - 2j * eta

    # Fermi function of complex arguments: f(z) = 1 / (exp(z/T) + 1).
    # For |Im(z)/T| << 1 this is close to f(Re z).  We just use the formula.
    def fc(z):
        return 1.0 / (np.exp(z / T) + 1.0)

    R_a = fc(a) / ((a - b) * (a - c) * (a - d))
    R_b = fc(b) / ((b - a) * (b - c) * (b - d))
    R_c = fc(c) / ((c - a) * (c - b) * (c - d))
    R_d = fc(d) / ((d - a) * (d - b) * (d - c))

    return (R_a + R_b + R_c + R_d).real


def _trace_4G_band_basis(
    M: np.ndarray,
    xi_k: np.ndarray,
    xi_kQ: np.ndarray,
    indices: tuple[int, int, int, int],
    T: float,
) -> np.ndarray:
    """Compute the per-k integrand of A_{ijkl}, Eq. (A10), in the band basis.

    The trace tr[ G_k sigma^i G_{k+Q} sigma^j G_k sigma^k G_{k+Q} sigma^l ]
    rotates to the band basis as a sum over band indices n, m, n', m':

        sum_{n m n' m'}  M^i_{n m} M^j_{n' m}^*  M^k_{n' m'} M^l_{n m'}^*
            * (1/beta) sum_{i omega} 1 / [(iw - xi_kn)(iw - xi_{k+Q,m})
                                          (iw - xi_kn')(iw - xi_{k+Q,m'})]

    Wait -- need to be careful about the sigma matrix structure.  Working
    in the orbital basis,
        G_k = sum_n |n_k><n_k| / (iw - xi_kn)
    so
        G_k sigma^i G_{k+Q} = sum_{n,m} |n_k><n_k| sigma^i |m_{k+Q}><m_{k+Q}|
                              / [(iw - xi_kn)(iw - xi_{k+Q,m})]
                            = sum_{n,m} |n_k> M^i_{nm}(k) <m_{k+Q}|
                              / [(iw - xi_kn)(iw - xi_{k+Q,m})]
    Multiplying four such factors and tracing:

    tr[G_k s^i G_kQ s^j G_k s^k G_kQ s^l]
      = sum_{n m n' m'}  M^i_{nm} (M^j)^dagger_{m n'} M^k_{n'm'} (M^l)^dagger_{m' n}
        * 1 / [(iw - xi_kn)(iw - xi_{k+Q,m})(iw - xi_kn')(iw - xi_{k+Q,m'})]

    where (M^j)^dagger_{m n'} = M^j*_{n' m}.  So the band amplitude is
        Phi_{n m n' m'} = M^i_{nm} M^{j*}_{n'm} M^k_{n'm'} M^{l*}_{nm'}
    and the energy denominator is the box-sum above.
    """
    i, j, k_idx, l = indices
    # M shape: (..., 4, 2_n, 2_m)
    Mi = M[..., i, :, :]            # (..., n, m)
    Mj = M[..., j, :, :]            # (..., n', m)  -- we will use M^{j*}_{n' m}
    Mk = M[..., k_idx, :, :]        # (..., n', m')
    Ml = M[..., l, :, :]            # (..., n, m')  -- we will use M^{l*}_{n m'}

    # Build energy box-sum array: shape (..., n=2, m=2, n'=2, m'=2)
    xi_n = xi_k[..., :, None, None, None]
    xi_m = xi_kQ[..., None, :, None, None]
    xi_np = xi_k[..., None, None, :, None]
    xi_mp = xi_kQ[..., None, None, None, :]
    box = _matsubara_box_sum(xi_n, xi_np, xi_m, xi_mp, T)   # (..., 2,2,2,2)

    # Phi_{nmn'm'} = Mi[n,m] * conj(Mj)[n',m] * Mk[n',m'] * conj(Ml)[n,m']
    # Axis layout (after broadcasting): (-4, -3, -2, -1) = (n, m, n', m').
    #
    #   Mi[n, m]    -> place original (n, m) at axes (-4, -3):  [..., :, :, None, None]
    #   Mj[n', m]   -> need (n', m) at axes (-2, -3); this is the TRANSPOSE
    #                  of (n, m), so swap last two axes of Mj first, then
    #                  Mj_T[m, n'] -> [..., None, :, :, None] places m at -3, n' at -2.
    #   Mk[n', m']  -> place original (n, m) at axes (-2, -1):  [..., None, None, :, :]
    #   Ml[n, m']   -> place original (n, m) at axes (-4, -1):  [..., :, None, None, :]
    Mj_T = np.swapaxes(Mj, -1, -2)
    Phi = (
        Mi[..., :, :, None, None]
        * np.conj(Mj_T)[..., None, :, :, None]
        * Mk[..., None, None, :, :]
        * np.conj(Ml)[..., :, None, None, :]
    )
    # Sum over (n, m, n', m') with the box weight:
    integrand = (Phi * box).sum(axis=(-4, -3, -2, -1))
    return integrand


def quartic_coeffs(
    Qx: float, Qy: float, T: float, p: TBParams, nk: int = 100
) -> dict[str, float]:
    """Compute the quartic GL coefficients b_-, b_z, c', c'' (Eq. A11).

    In the t_d = 0 limit, Delta^- == Delta^0 and Delta^+ == Delta^x, so we
    simply use sigma^0 in place of sigma^- and sigma^x in place of sigma^+.

    For finite t_d, the basis change to (-, +) involves an angle theta
    determined by the diagonalization of the 2x2 (Delta^0, Delta^x) block.
    For the realistic parameters of the paper (lambda << ax - a0), this
    rotation is very small and we approximate Delta^- ~ Delta^0.  The
    correction enters at O(lambda / (ax - a0)) and is neglected here.

    Returns
    -------
    dict with keys 'b_minus', 'b_z', 'c_prime', 'c_pprime'
    """
    KX, KY = make_bz_grid(nk)
    eps_k, a_k = bands_and_projectors(KX, KY, p)
    eps_kQ, a_kQ = bands_and_projectors(KX + Qx, KY + Qy, p)
    xi_k = eps_k - p.mu
    xi_kQ = eps_kQ - p.mu

    M = _build_mixed_pauli(a_k, a_kQ)   # (nk, nk, 4, 2, 2)

    # Eq. A11 (in the {0, z} basis approximation):
    # b_-    = A_{0000}
    # b_z    = A_{zzzz} = A_{3333}
    # c'-c'' = A_{0z0z} = A_{0303}
    # c'+c'' = (1/2) (A_{00zz} + A_{0zz0} + A_{zz00} + A_{z00z})
    #        = (1/2) (A_{0033} + A_{0330} + A_{3300} + A_{3003})
    #
    # Each A_{ijkl} = (1/2) integral_BZ dk * trace_per_k.
    bz_measure = (2.0 * np.pi) ** 2 / KX.size

    def A(i, j, k_idx, l):
        per_k = _trace_4G_band_basis(M, xi_k, xi_kQ, (i, j, k_idx, l), T)
        return 0.5 * bz_measure * per_k.sum().real

    A_0000 = A(0, 0, 0, 0)
    A_3333 = A(3, 3, 3, 3)
    A_0303 = A(0, 3, 0, 3)
    A_0033 = A(0, 0, 3, 3)
    A_0330 = A(0, 3, 3, 0)
    A_3300 = A(3, 3, 0, 0)
    A_3003 = A(3, 0, 0, 3)

    b_minus = A_0000
    b_z = A_3333
    c_minus_cpp = A_0303                                # c' - c''
    c_plus_cpp = 0.5 * (A_0033 + A_0330 + A_3300 + A_3003)  # c' + c''
    c_prime = 0.5 * (c_plus_cpp + c_minus_cpp)
    c_pprime = 0.5 * (c_plus_cpp - c_minus_cpp)

    return {
        "b_minus": float(b_minus),
        "b_z": float(b_z),
        "c_prime": float(c_prime),
        "c_pprime": float(c_pprime),
        "c_diff": float(c_minus_cpp),    # c' - c''
        "c_sum": float(c_plus_cpp),      # c' + c''
    }


def coexistence_criterion(coeffs: dict[str, float]) -> tuple[bool, float]:
    """Check the double-Delta condition Eq. (29):
        [(c' + c'') - |c' - c''|]^2  <  b_- b_z

    Returns (is_double_phase, ratio) where ratio = LHS / RHS.
    """
    LHS = ((coeffs["c_sum"]) - abs(coeffs["c_diff"])) ** 2
    RHS = coeffs["b_minus"] * coeffs["b_z"]
    return (LHS < RHS, LHS / RHS if RHS != 0 else np.inf)
