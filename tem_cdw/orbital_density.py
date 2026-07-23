"""
Real-space orbital occupations rho_{px}(r), rho_{py}(r), rho(r) for the
four phases of Sec. V (Figs. 4 and 10 of the paper).

From Eq. (22):
    rho_{p_alpha}(r) = |Delta^{p_alpha p_alpha}_Q| cos(Q . r + alpha_{p_alpha}),
where Delta^{p_alpha p_alpha}_Q = |Delta^{p_alpha p_alpha}_Q| e^{i alpha_{p_alpha}}.

The four phases of Table V are parameterized in terms of Delta^0 and Delta^z
(both relative to a global phase, which we fix to 0):

    Single-Delta A_g  : Delta^0 = D, Delta^z = 0
        => Delta^{px,px}_Q = Delta^{py,py}_Q = D/2
    Single-Delta B_1g : Delta^0 = 0, Delta^z = D
        => Delta^{px,px}_Q = D/2, Delta^{py,py}_Q = -D/2
    Double-Delta B_2u : Delta^0 = D0, Delta^z = i Dz   (relative phase pi/2)
        => Delta^{px,px}_Q = (D0 + i Dz)/2, Delta^{py,py}_Q = (D0 - i Dz)/2
        (equal magnitude, opposite phase)
    Double-Delta B_1g : Delta^0 = D0, Delta^z = Dz     (relative phase 0)
        => Delta^{px,px}_Q = (D0 + Dz)/2, Delta^{py,py}_Q = (D0 - Dz)/2
        (different magnitudes, same phase)
"""

from __future__ import annotations

import numpy as np


def orbital_amplitudes(D0: complex, Dz: complex) -> tuple[complex, complex]:
    """Return (Delta^{px,px}_Q, Delta^{py,py}_Q) from (Delta^0_Q, Delta^z_Q).

        Delta^{px,px}_Q = (Delta^0 + Delta^z) / 2
        Delta^{py,py}_Q = (Delta^0 - Delta^z) / 2
    """
    Dpx = 0.5 * (D0 + Dz)
    Dpy = 0.5 * (D0 - Dz)
    return Dpx, Dpy


def real_space_density(
    Qx: float,
    Qy: float,
    D0: complex,
    Dz: complex,
    nx: int = 9,
    ny: int = 9,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute rho_{px}(r), rho_{py}(r), rho(r) on a (nx x ny) lattice.

    Returns
    -------
    X, Y : meshgrids of integer lattice positions
    rho_px, rho_py : real-space orbital occupations
    rho : total density rho_px + rho_py
    """
    Dpx, Dpy = orbital_amplitudes(D0, Dz)

    amp_px = np.abs(Dpx)
    amp_py = np.abs(Dpy)
    phase_px = np.angle(Dpx)
    phase_py = np.angle(Dpy)

    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y, indexing="ij")

    arg = Qx * X + Qy * Y
    rho_px = amp_px * np.cos(arg + phase_px)
    rho_py = amp_py * np.cos(arg + phase_py)
    rho = rho_px + rho_py
    return X, Y, rho_px, rho_py, rho


PHASES = {
    "single_Ag":     dict(D0=1.0,         Dz=0.0,         label="Single-$\\Delta$ $A_g$"),
    "single_B1g":    dict(D0=0.0,         Dz=1.0,         label="Single-$\\Delta$ $B_{1g}$"),
    "double_B2u":    dict(D0=1.0,         Dz=1.0j,        label="Double-$\\Delta$ $B_{2u}$"),
    "double_B1g":    dict(D0=1.0,         Dz=0.5,         label="Double-$\\Delta$ $B_{1g}$"),
}
