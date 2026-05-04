"""
Demo: reproduce the main figures of Alekseev et al., PRB 110, 205103 (2024),
using only Python / NumPy.

Run with:
    python demo.py [--quick]

`--quick` uses smaller grids for faster rendering; default uses publication-grade
grids (slower but matches the paper closely).

Outputs are saved to ./figures/ as PNG files.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, '/home/claude')

from rte3_cdw.plotting import (
    plot_fig1, plot_fig3, plot_fig4, plot_fig5, plot_fig6, plot_fig7, plot_fig9,
)

OUTDIR = "/mnt/user-data/outputs/figures"
os.makedirs(OUTDIR, exist_ok=True)


def banner(s):
    print("\n" + "=" * 60)
    print("  " + s)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use smaller grids (faster but less smooth)")
    parser.add_argument("--only", default=None, type=str,
                        help="Comma-separated list of figs to plot (e.g. '1,3,4')")
    args = parser.parse_args()

    nk_band = 200 if args.quick else 400      # for FS plots
    nk_chi = 100 if args.quick else 200       # for susceptibility
    nk_gl = 100 if args.quick else 200        # for GL coefficients

    only = set(args.only.split(",")) if args.only else None

    def should(name):
        return only is None or name in only

    if should("1"):
        banner("Fig. 1: Bare Fermi surfaces")
        t0 = time.time()
        plot_fig1(nk=nk_band, savepath=f"{OUTDIR}/fig1_bare_FS.png")
        print(f"  done in {time.time() - t0:.1f}s")

    if should("3"):
        banner("Fig. 3: Susceptibility vs T")
        t0 = time.time()
        plot_fig3(nk=nk_chi, savepath=f"{OUTDIR}/fig3_chi_vs_T.png")
        print(f"  done in {time.time() - t0:.1f}s")

    if should("4"):
        banner("Fig. 4: Real-space orbital occupations")
        t0 = time.time()
        plot_fig4(savepath=f"{OUTDIR}/fig4_orbital_textures.png")
        print(f"  done in {time.time() - t0:.1f}s")

    if should("5"):
        banner("Fig. 5: Quadratic GL coefficients")
        t0 = time.time()
        plot_fig5(td=0.16, nk=nk_gl, savepath=f"{OUTDIR}/fig5_quadratic_td016.png")
        plot_fig5(td=0.4, nk=nk_gl, savepath=f"{OUTDIR}/fig5_quadratic_td040.png")
        print(f"  done in {time.time() - t0:.1f}s")

    if should("6"):
        banner("Fig. 6: Quartic c'-c'' and c'+c''")
        t0 = time.time()
        plot_fig6(nk=nk_gl, savepath=f"{OUTDIR}/fig6_quartic.png")
        print(f"  done in {time.time() - t0:.1f}s")

    if should("7"):
        banner("Fig. 7: Coexistence criterion")
        t0 = time.time()
        plot_fig7(nk=nk_gl, savepath=f"{OUTDIR}/fig7_coexistence.png")
        print(f"  done in {time.time() - t0:.1f}s")

    if should("9"):
        banner("Fig. 9: CDW-reconstructed Fermi surfaces")
        t0 = time.time()
        plot_fig9(nk=nk_band, savepath=f"{OUTDIR}/fig9_reconstructed_FS.png")
        print(f"  done in {time.time() - t0:.1f}s")

    print(f"\nAll done. Figures saved in {OUTDIR}/")


if __name__ == "__main__":
    main()
