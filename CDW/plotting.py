"""Plotting utilities for the figures of the paper."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .fermi_surface import fermi_surface_spectral_weight, cdw_order_matrix
from .hamiltonian import TBParams, h_k, make_bz_grid
from .orbital_density import PHASES, real_space_density


# ----------------------------------------------------------------------
# Figure 1: bare Fermi surfaces
# ----------------------------------------------------------------------

def plot_fig1(p: TBParams = TBParams(), nk: int = 400, savepath: str | None = None):
    """Plot Fig. 1: Fermi surfaces for three parameter combinations."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    KX, KY = make_bz_grid(nk)

    # Shift to (-pi, pi) for nicer plotting
    KX_p = np.where(KX > np.pi, KX - 2 * np.pi, KX)
    KY_p = np.where(KY > np.pi, KY - 2 * np.pi, KY)

    cases = [
        (TBParams(t_sigma=p.t_sigma, t_pi=0.0, t_d=0.0, mu=p.mu), "(a) $t_\\pi=t_d=0$"),
        (TBParams(t_sigma=p.t_sigma, t_pi=0.37, t_d=0.0, mu=p.mu), "(b) $t_\\pi=0.37$, $t_d=0$"),
        (TBParams(t_sigma=p.t_sigma, t_pi=0.37, t_d=0.16, mu=p.mu), "(c) $t_\\pi=0.37$, $t_d=0.16$"),
    ]

    for ax, (pp, title) in zip(axes, cases):
        h = h_k(KX, KY, pp)
        E, _ = np.linalg.eigh(h)
        # Plot zero contour of (E - mu) for each band
        for n in range(2):
            ax.contour(KX_p, KY_p, E[..., n] - pp.mu, levels=[0.0], colors="k", linewidths=0.8)

        kF = pp.kF()
        Q0 = (2 * kF if 2 * kF < np.pi else 2 * kF - 2 * np.pi,
              2 * kF if 2 * kF < np.pi else 2 * kF - 2 * np.pi)
        ax.annotate("", xy=Q0, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.annotate("", xy=(2 * kF - 2 * np.pi, np.pi - 2 * np.pi if np.pi > np.pi else np.pi),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="blue", lw=2, alpha=0.6))

        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")
        ax.set_xlabel("$k_x$")
        ax.set_ylabel("$k_y$")
        ax.set_title(title)
        ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"])
        ax.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "$0$", r"$\pi/2$", r"$\pi$"])

    fig.suptitle("Fig. 1: Bare Fermi surfaces", y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figure 3: chi(Q, T) vs T for two Q values, t_d = 0 and 0.16
# ----------------------------------------------------------------------

def plot_fig3(savepath: str | None = None, nk: int = 200):
    from .susceptibility import chi_scalar_at_Q
    p_no = TBParams(t_sigma=2.0, t_pi=0.37, t_d=0.0, mu=-1.53)
    p_full = TBParams(t_sigma=2.0, t_pi=0.37, t_d=0.16, mu=-1.53)
    kF = p_no.kF()
    Ts = np.linspace(0.005, 0.15, 25)

    chi_Q0_no = [chi_scalar_at_Q(2*kF, 2*kF, T, p_no, nk=nk) for T in Ts]
    chi_alt_no = [chi_scalar_at_Q(2*kF, np.pi, T, p_no, nk=nk) for T in Ts]
    chi_Q0_full = [chi_scalar_at_Q(2*kF, 2*kF, T, p_full, nk=nk) for T in Ts]
    chi_alt_full = [chi_scalar_at_Q(2*kF, np.pi, T, p_full, nk=nk) for T in Ts]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, (Q0_data, alt_data, title) in zip(
        axes,
        [(chi_Q0_no, chi_alt_no, "$t_d = 0$"),
         (chi_Q0_full, chi_alt_full, "$t_d = 0.16$ eV")],
    ):
        ax.plot(Ts, Q0_data, "rs", label="$Q = (2k_F, 2k_F)$", markersize=5)
        ax.plot(Ts, alt_data, "bo", label="$Q = (2k_F, \\pi)$", markersize=5)
        ax.set_xlabel("$T$ (eV)")
        ax.set_ylabel("$\\chi(Q, T, \\mu)$")
        ax.set_title(title)
        ax.legend()
        ax.set_ylim(10, 20)
    fig.suptitle("Fig. 3: Susceptibility vs T", y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figure 4 / 10: real-space orbital occupations
# ----------------------------------------------------------------------

def plot_fig4(Qx: float = np.pi/3, Qy: float = np.pi/3, nx: int = 9, ny: int = 9,
              savepath: str | None = None):
    """Plot Fig. 4-style orbital occupations for the four phases."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (key, params) in zip(axes, PHASES.items()):
        D0 = params["D0"]
        Dz = params["Dz"]
        X, Y, rho_px, rho_py, rho = real_space_density(Qx, Qy, D0, Dz, nx=nx, ny=ny)
        # Draw two markers per site (px and py): use diamonds for px, circles for py
        # Filled = positive, empty = negative; intensity = magnitude
        for arr, marker, dx in [(rho_px, "D", -0.18), (rho_py, "o", 0.18)]:
            mag = np.abs(arr)
            mag_norm = mag / (np.max(mag) + 1e-12)
            for i in range(nx):
                for j in range(ny):
                    val = arr[i, j]
                    color = plt.cm.viridis(mag_norm[i, j])
                    if val >= 0:
                        ax.scatter(i + dx, j, c=[color], s=200, marker=marker,
                                   edgecolor="k", linewidths=0.5)
                    else:
                        ax.scatter(i + dx, j, c="white", s=200, marker=marker,
                                   edgecolor=color, linewidths=2.0)
        ax.set_aspect("equal")
        ax.set_xlim(-1, nx)
        ax.set_ylim(-1, ny)
        ax.set_title(params["label"])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Fig. 4: Real-space orbital occupations\n(diamonds: $p_x$, circles: $p_y$; "
                 "filled = +, open = -)", y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figure 5: quadratic GL coefficients vs T
# ----------------------------------------------------------------------

def plot_fig5(td: float = 0.16, g: float = 1.0, nk: int = 200,
              savepath: str | None = None):
    from .gl_theory import quadratic_coeffs
    p = TBParams(t_sigma=2.0, t_pi=0.37, t_d=td, mu=-1.53)
    kF = p.kF()
    Qx, Qy = 2 * kF, 2 * kF
    Ts = np.linspace(0.005, 0.30, 30)

    a_minus, a_plus, ay, az, lam = [], [], [], [], []
    for T in Ts:
        qc = quadratic_coeffs(Qx, Qy, T, p, g=g, nk=nk)
        am, ap, _ay, _az = qc.diagonalize()
        a_minus.append(am - 2 / g)   # plot a_i - 2/g (the susceptibility part)
        a_plus.append(ap - 2 / g)
        ay.append(_ay - 2 / g)
        az.append(_az - 2 / g)
        lam.append(qc.lam)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Ts, a_minus, "rs", label="$a_-$", markersize=4)
    ax.plot(Ts, az, "o", color="orange", label="$a_z$", markersize=4)
    ax.plot(Ts, a_plus, "bs", label="$a_+$", markersize=4)
    ax.plot(Ts, ay, "go", label="$a_y$", markersize=4)
    ax.plot(Ts, lam, "p", color="purple", label="$\\lambda$", markersize=4)
    ax.set_xlabel("$T$ (eV)")
    ax.set_ylabel("$a_i - 2/g$")
    ax.set_title(f"Fig. 5: Quadratic GL coefficients ($t_d = {td}$ eV)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figure 6: c'-c'' and c'+c'' vs T for several td
# ----------------------------------------------------------------------

def plot_fig6(savepath: str | None = None, nk: int = 200):
    from .gl_theory import quartic_coeffs
    Ts = np.linspace(0.01, 0.05, 9)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    for ax, td in zip(axes, [0.05, 0.10, 0.15]):
        p = TBParams(t_sigma=2.0, t_pi=0.37, t_d=td, mu=-1.53)
        kF = p.kF()
        c_diff, c_sum = [], []
        for T in Ts:
            q = quartic_coeffs(2*kF, 2*kF, T, p, nk=nk)
            c_diff.append(q["c_diff"])
            c_sum.append(q["c_sum"])
        ax.plot(Ts, c_diff, "bo", label="$c' - c''$", markersize=6)
        ax.plot(Ts, c_sum, "rs", label="$c' + c''$", markersize=6)
        ax.set_xlabel("$T$ (eV)")
        ax.set_title(f"$t_d = {td}$ eV")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle("Fig. 6: Quartic cross-coupling coefficients", y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figure 7: coexistence ratio  (4 c''^2 - b_- b_z) / (4 c''^2)
# ----------------------------------------------------------------------

def plot_fig7(savepath: str | None = None, nk: int = 200):
    from .gl_theory import quartic_coeffs
    Ts = np.linspace(0.01, 0.05, 9)
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0.05: "red", 0.10: "blue", 0.15: "orange"}
    for td in [0.05, 0.10, 0.15]:
        p = TBParams(t_sigma=2.0, t_pi=0.37, t_d=td, mu=-1.53)
        kF = p.kF()
        ratios = []
        for T in Ts:
            q = quartic_coeffs(2*kF, 2*kF, T, p, nk=nk)
            cpp = q["c_pprime"]
            r = (4 * cpp**2 - q["b_minus"] * q["b_z"]) / (4 * cpp**2 + 1e-30)
            ratios.append(r)
        ax.plot(Ts, ratios, "o-", color=colors[td], label=f"$t_d = {td}$ eV")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("$T$ (eV)")
    ax.set_ylabel("$(4 c''^2 - b_- b_z)/(4 c''^2)$")
    ax.set_title("Fig. 7: Coexistence criterion (negative = double-$\\Delta$ favored)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig


# ----------------------------------------------------------------------
# Figures 8, 9: CDW-reconstructed Fermi surfaces
# ----------------------------------------------------------------------

def plot_fig9(savepath: str | None = None, nk: int = 400, broadening: float = 0.02):
    """Reproduce Fig. 9: FS for four CDW configurations."""
    p = TBParams()
    kF = p.kF()
    Qx, Qy = 2 * kF, 2 * kF
    KX, KY = make_bz_grid(nk)
    KX_p = np.where(KX > np.pi, KX - 2 * np.pi, KX)
    KY_p = np.where(KY > np.pi, KY - 2 * np.pi, KY)

    # Sort the meshgrid so pcolormesh draws contiguous quads (otherwise the
    # BZ-wrap creates artifacts that span the whole plot).
    ix = np.argsort(KX_p[:, 0])
    iy = np.argsort(KY_p[0, :])
    KX_s = KX_p[np.ix_(ix, iy)]
    KY_s = KY_p[np.ix_(ix, iy)]

    cases = [
        ("(a) $\\Delta^0=0.1$, $\\Delta^z=0$",                  cdw_order_matrix(D0=0.1, Dz=0)),
        ("(b) $\\Delta^0=0$, $\\Delta^z=0.1$",                  cdw_order_matrix(D0=0,   Dz=0.1)),
        ("(c) $\\Delta^0=0.1$, $\\Delta^z=0.1i$ (B$_{2u}$)",    cdw_order_matrix(D0=0.1, Dz=0.1j)),
        ("(d) $\\Delta^0=0.1$, $\\Delta^z=0.05$ (B$_{1g}$)",    cdw_order_matrix(D0=0.1, Dz=0.05)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (title, Delta) in zip(axes.flat, cases):
        A = fermi_surface_spectral_weight(KX, KY, Qx, Qy, Delta, p,
                                          broadening=broadening)
        A_s = A[np.ix_(ix, iy)]
        # vmax tuned to 20% of peak: gives a reasonable dynamic range so the
        # main FS contours are visible without being crushed by the very-sharp
        # peaks at non-nested portions.
        ax.pcolormesh(KX_s, KY_s, A_s, shading="auto", cmap="Blues",
                      vmin=0, vmax=0.2 * A_s.max())
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xticks([-np.pi, 0, np.pi]); ax.set_xticklabels([r"$-\pi$", "$0$", r"$\pi$"])
        ax.set_yticks([-np.pi, 0, np.pi]); ax.set_yticklabels([r"$-\pi$", "$0$", r"$\pi$"])
    fig.suptitle("Fig. 9: CDW-reconstructed Fermi surfaces", y=1.0)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=130, bbox_inches="tight")
    return fig
