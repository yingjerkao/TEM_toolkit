"""Offline LaTe₂ tight-binding fit.

Usage:
    python -m scripts.fit_late2               # runs full fit
    python -m scripts.fit_late2 --quick       # smaller grid / lower nk (dev)

Outputs:
    tem_cdw/models/rte2_fit_report/fitted_params.npz
    tem_cdw/models/rte2_fit_report/fit_report.png
"""
from __future__ import annotations

import argparse
import itertools
import pathlib

import numpy as np
from scipy.optimize import minimize
from scipy.signal import find_peaks

from tem_cdw.hamiltonian import TBParams, h_k, diagonalize
from tem_cdw.susceptibility import susceptibility_tensor, chi_combinations
from tem_cdw.units import TWO_PI


T_FIT = 0.026                                  # eV, ~ 300 K
TARGET_KF_AXIAL = 0.35 * TWO_PI                # radians
TARGET_QLOW = 0.15 * TWO_PI
TARGET_QHIGH = 0.25 * TWO_PI

SENTINEL_LARGE = 1e6


def kF_axial(p: TBParams, nk: int = 400) -> float:
    """Outermost Fermi-level crossing of the lower band along ky=0, kx>0."""
    kx = np.linspace(0.0, np.pi, nk)
    ky = np.zeros_like(kx)
    h = h_k(kx, ky, p)
    eps, _ = diagonalize(h)
    lower = eps[..., 0]
    d = lower - p.mu
    idx = np.where(np.diff(np.sign(d)))[0]
    if len(idx) == 0:
        return float("nan")
    i = idx[-1]
    x0, x1 = kx[i], kx[i + 1]
    y0, y1 = d[i], d[i + 1]
    return float(x0 - y0 * (x1 - x0) / (y1 - y0))


def chi_peaks_along_a(
    p: TBParams, T: float = T_FIT, nk_bz: int = 100, nk_q: int = 40
) -> tuple[list[float], np.ndarray, np.ndarray]:
    """Return sorted peak-q list, q_grid, chi_grid along Q=(q, 0)."""
    q_astar = np.linspace(0.02, 0.5, nk_q)
    q_grid = q_astar * TWO_PI
    chi_grid = np.empty(nk_q)
    for i, q in enumerate(q_grid):
        chi = susceptibility_tensor(q, 0.0, T, p, nk=nk_bz)
        chi_grid[i] = chi_combinations(chi)["diag_aa"]

    prom = 0.02 * (chi_grid.max() - chi_grid.min())
    idx, _ = find_peaks(chi_grid, prominence=prom)
    if len(idx) == 0:
        return [], q_grid, chi_grid
    top = sorted(idx, key=lambda i: chi_grid[i], reverse=True)[:2]
    top = sorted(top, key=lambda i: q_grid[i])
    return [float(q_grid[i]) for i in top], q_grid, chi_grid


def objective(
    x: np.ndarray, w_FS: float = 1.0, w_q: float = 10.0,
    nk_bz: int = 100, nk_q: int = 40,
) -> float:
    ts, tp, td, mu = x
    if not (0.1 < ts < 5.0 and 0.0 <= tp < 2.0 and 0.0 <= td < 1.5 and -5.0 < mu < 2.0):
        return SENTINEL_LARGE
    p = TBParams(t_sigma=float(ts), t_pi=float(tp), t_d=float(td), mu=float(mu))
    kf = kF_axial(p)
    if not np.isfinite(kf):
        return SENTINEL_LARGE
    qs, _, _ = chi_peaks_along_a(p, nk_bz=nk_bz, nk_q=nk_q)
    if len(qs) < 2:
        return SENTINEL_LARGE
    q_lo, q_hi = qs[0], qs[1]
    return (
        w_FS * (kf - TARGET_KF_AXIAL) ** 2
        + w_q * ((q_lo - TARGET_QLOW) ** 2 + (q_hi - TARGET_QHIGH) ** 2)
    )


def coarse_grid(quick: bool):
    if quick:
        ts_g = np.linspace(1.8, 2.4, 3)
        tp_g = np.linspace(0.2, 0.6, 3)
        td_g = np.linspace(0.1, 0.3, 3)
        mu_g = np.linspace(-2.0, -1.0, 3)
    else:
        ts_g = np.linspace(1.5, 3.0, 5)
        tp_g = np.linspace(0.15, 0.75, 5)
        td_g = np.linspace(0.05, 0.35, 5)
        mu_g = np.linspace(-2.5, -0.8, 5)
    return list(itertools.product(ts_g, tp_g, td_g, mu_g))


def run_fit(quick: bool = False) -> dict:
    nk_bz = 60 if quick else 100
    nk_q = 25 if quick else 40

    print("Coarse grid search...")
    combos = coarse_grid(quick)
    scores = np.array([
        objective(np.asarray(c), nk_bz=nk_bz, nk_q=nk_q) for c in combos
    ])
    order = np.argsort(scores)
    top5 = [combos[i] for i in order[:5]]
    print(f"  best coarse score: {scores[order[0]]:.4f}")
    for i in order[:5]:
        print(f"    x={combos[i]} score={scores[i]:.4f}")

    print("Nelder-Mead refine...")
    best_x = None
    best_score = np.inf
    for start in top5:
        res = minimize(
            objective, np.asarray(start, dtype=float),
            method="Nelder-Mead",
            options={"xatol": 5e-3, "fatol": 1e-4, "maxiter": 200},
        )
        if res.fun < best_score:
            best_score = res.fun
            best_x = res.x

    ts, tp, td, mu = best_x
    p_best = TBParams(t_sigma=ts, t_pi=tp, t_d=td, mu=mu)
    peaks_q, q_grid, chi_grid = chi_peaks_along_a(p_best, nk_bz=nk_bz, nk_q=nk_q)
    kf = kF_axial(p_best)

    return {
        "params": p_best,
        "score": float(best_score),
        "peaks_astar": [q / TWO_PI for q in peaks_q],
        "kF_astar": kf / TWO_PI,
        "q_grid": q_grid,
        "chi_grid": chi_grid,
    }


def save_report(result: dict) -> pathlib.Path:
    import matplotlib.pyplot as plt

    outdir = pathlib.Path("tem_cdw/models/rte2_fit_report")
    outdir.mkdir(parents=True, exist_ok=True)

    p = result["params"]
    np.savez(
        outdir / "fitted_params.npz",
        t_sigma=p.t_sigma, t_pi=p.t_pi, t_d=p.t_d, mu=p.mu,
        score=result["score"], peaks_astar=np.asarray(result["peaks_astar"]),
        kF_astar=result["kF_astar"],
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    q_astar = result["q_grid"] / TWO_PI
    ax.plot(q_astar, result["chi_grid"], "b-", label=r"$\chi(q, 0)$")
    for q in result["peaks_astar"]:
        ax.axvline(q, color="k", ls=":", alpha=0.5)
        ax.text(q, ax.get_ylim()[1] * 0.9, f"{q:.3f}", ha="center")
    ax.axvline(0.15, color="r", ls="--", alpha=0.5, label="target 0.15")
    ax.axvline(0.25, color="g", ls="--", alpha=0.5, label="target 0.25")
    ax.set_xlabel(r"$q$ (units of $a^*$)")
    ax.set_ylabel(r"$\chi_{\rm diag\_aa}(q, 0)$")
    ax.set_title(
        f"LaTe2 fit: t_sigma={p.t_sigma:.3f}, t_pi={p.t_pi:.3f}, "
        f"t_d={p.t_d:.3f}, mu={p.mu:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "fit_report.png", dpi=120)
    plt.close(fig)
    return outdir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid for dev")
    args = ap.parse_args()

    result = run_fit(quick=args.quick)
    outdir = save_report(result)

    p = result["params"]
    print("Fit complete.")
    print(f"  t_sigma = {p.t_sigma:.4f}")
    print(f"  t_pi    = {p.t_pi:.4f}")
    print(f"  t_d     = {p.t_d:.4f}")
    print(f"  mu      = {p.mu:.4f}")
    print(f"  score   = {result['score']:.4f}")
    print(f"  peaks   = {result['peaks_astar']}  (target 0.15, 0.25 a*)")
    print(f"  kF      = {result['kF_astar']:.4f} a*  (target 0.35 a*)")
    print(f"  Report saved to {outdir}/")


if __name__ == "__main__":
    main()
