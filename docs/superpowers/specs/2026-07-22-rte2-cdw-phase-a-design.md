# RTE2 CDW — Phase A design

**Date:** 2026-07-22
**Author:** Ying-Jer Kao (with Claude)
**Status:** Draft for review

## Purpose

Adapt the existing `rte3_cdw` (Alekseev PRB 110, 205103 (2024)) machinery to
study the CDW of rare-earth ditellurides (RTE2), specifically LaTe₂. Phase A
reproduces the Alekseev workflow — susceptibility tensor, Ginzburg–Landau
quadratic + quartic coefficients, coexistence check, orbital-density textures
— at RTE2 band filling and axial nesting geometry, matching the experimental
TEM observations of two CDW states in LaTe₂:

- **q ≈ 0.15 a\*** — incommensurate
- **q ≈ 0.25 a\*** — discommensurate (near ¼ commensurate lock-in)

## Motivation

Two experimental / theoretical anchors:

1. *Band Engineering of Dirac Semimetals Using Charge Density Waves*,
   Adv. Mater. **33**, 2101591 (2021), doi: 10.1002/adma.202101591 —
   frames RTE2 CDW as a route to CDW-protected Dirac semimetal states.
2. *Stability of charge-density waves under continuous variation of band
   filling in LaTe₂₋ₓSbₓ* (PRB) — Sb substitution tunes band filling (μ)
   continuously and probes CDW stability.

Together with in-house TEM data (two q values above), these motivate a
LaTe₂-focused adaptation of the Alekseev 2-orbital tight-binding + GL
framework. This design covers Phase A only; Phases B (Dirac cones from
CDW-reconstructed bands) and C (μ-sweep for Sb doping) are sketched in
`docs/superpowers/roadmap-rte2.md` and each will get its own spec.

## Non-goals for Phase A

- CDW-reconstructed band structure along Γ–X–M–Y (Phase B).
- Filling scan for LaTe₂₋ₓSbₓ (Phase C).
- Electron–phonon coupling; soliton-lattice / discommensurate lock-in
  physics; the "why q_high sits near ¼" story (deferred beyond A/B/C).
- Multi-Q / checkerboard CDW mean-field (deferred; single-Q per instability
  is sufficient to establish the two-peak χ structure).
- Any change to the Alekseev physics conventions: `h(k)`, χ sign/indexing,
  GL basis `{0, x, y, z}`, quartic formulas. The RTE3 notebook must
  reproduce bit-for-bit after the refactor.

## Architecture — shared core + material adapters

The current `rte3_cdw/` module is generalized into a shared physics core
plus material-specific adapters. The 2-orbital `(t_σ, t_π, t_d, μ)`
tight-binding form is the same for both materials; what differs is the
parameter values and how Q₀ is derived.

### Directory layout

```
TEM_toolkit/
├── tem_cdw/                    # shared core (renamed from rte3_cdw)
│   ├── __init__.py
│   ├── hamiltonian.py          # generic 2-orbital TB h(k), diagonalize, BZ grid
│   ├── susceptibility.py       # χ tensor + combinations — unchanged
│   ├── gl_theory.py            # quadratic + quartic GL — unchanged
│   ├── fermi_surface.py        # 6×6 CDW block — parameterized by (Qx, Qy)
│   ├── orbital_density.py      # ρ_px(r), ρ_py(r) — parameterized by Q
│   ├── plotting.py             # low-level plot helpers
│   └── models/
│       ├── __init__.py         # exports TBParams, MaterialModel
│       ├── base.py             # MaterialModel ABC
│       ├── rte3.py             # Alekseev defaults, analytic (2kF, 2kF) Q₀
│       └── rte2.py             # LaTe₂ fitted params, numerical axial Q₀
├── rte3_cdw_walkthrough.ipynb  # updated imports only
├── rte2_cdw_walkthrough.ipynb  # NEW — Phase A walkthrough
└── docs/superpowers/
    ├── roadmap-rte2.md
    └── specs/2026-07-22-rte2-cdw-phase-a-design.md
```

Old `rte3_cdw/` is removed (its contents move to `tem_cdw/` with
`models/rte3.py` holding the RTE3 parameter defaults and Q₀ recipe).

### Key abstractions

**`TBParams`** (unchanged): frozen dataclass of `(t_σ, t_π, t_d, μ)`.

**`MaterialModel`** (new ABC in `models/base.py`):

- `.params: TBParams`
- `.find_Q0(T: float, nk_bz: int, nk_q: int) -> Q0Result`
  where `Q0Result` bundles a list of candidate `(Qx, Qy)` peaks plus the
  diagnostic scan arrays (`q_grid`, `chi_axial`, `chi_diagonal`) for
  plotting.
- `.name: str` for labels and cache keys.

**`RTE3Model`** implements `find_Q0` analytically: returns
`[(2kF, 2kF)]` and empty diagnostic arrays. Matches current behavior.

**`RTE2Model`** implements `find_Q0` numerically (see § χ-scan below).

Everything else in the core (`susceptibility.py`, `gl_theory.py`,
`fermi_surface.py`, `orbital_density.py`) is already `TBParams`-driven;
imports change but logic does not.

### Refactor verification

After the refactor, re-running `rte3_cdw_walkthrough.ipynb` must produce
identical numeric outputs (χ values, GL coefficients, phase diagram)
compared to the current main branch. This is the invariant that
protects Alekseev's physics conventions during the reorganization.

## RTE2 tight-binding fit

`models/rte2.py::LaTe2Model()` is a factory that returns a `RTE2Model`
with fitted `TBParams`. The fit is a small offline optimization run
once; results are frozen into the factory as defaults.

### Fit targets (weighted objective)

Weights `w_FS`, `w_q` chosen so the two terms are of comparable
magnitude after normalization; specific values tuned in implementation.

1. **FS diamond corner** — outer FS crosses a* axis at ≈ 0.35 a*
   (from the reference figure). Pins a combination of (t_σ, t_π, μ).
2. **Nesting sweet spots** — `χ(Q = (q, 0))` at T = 300 K has local peaks
   near **q_min ≈ 0.15 a\*** and **q_max ≈ 0.25 a\***. Pins FS
   *curvature*, which requires nonzero t_π and t_d.
3. **Orbital texture** — pₓ sheet aligned along b*, p_y along a*.
   Automatic in the 2-orbital form; verified, not fitted.

Objective:

```
L(t_σ, t_π, t_d, μ) = w_FS * (kF_axial - 0.35 · 2π)²
                    + w_q  * [(q_min_pred - 0.15 · 2π)²
                             + (q_max_pred - 0.25 · 2π)²]
```

`kF_axial` is defined as the outermost FS crossing on the a* axis
(numerical root of ε_lower(k, 0) = μ).

### Fit procedure

1. Coarse grid, 5⁴ = 625 combos centered on the RTE3 defaults
   `(t_σ, t_π, t_d, μ) = (2.0, 0.37, 0.16, -1.53)`, ranges wide enough
   to cover a factor of ~2 in each parameter.
2. Score every combo by `L`; keep the top-5.
3. Refine each with `scipy.optimize.minimize(method="Nelder-Mead")`;
   pick the global best.
4. Save the fitted params, a "fit report" figure (FS overlay + χ(q)
   showing both peaks), and the raw objective value into
   `tem_cdw/models/rte2_fit_report/`. The factory reads these frozen
   values on import.

### Failure mode we surface explicitly

If the 2-orbital form cannot reproduce both q_min and q_max
simultaneously (best-fit `L` above a threshold, or χ has only one peak
along a*), the fit report says so and Phase A stops here. That is a
physics finding worth reporting — RTE2 would need extra hoppings (e.g.
second-neighbor) or a bilayer/multi-orbital extension of the model.

### Units housekeeping

The Alekseev code uses radians for k, Q throughout. TEM q values and
figure axes are in units of a\* = 2π/a. A small helper module
`tem_cdw/units.py` exposes `q_from_astar(q_astar) = 2π · q_astar` and
its inverse, used at every model/plot boundary to avoid stray 2π
factors.

## Numerical χ(Q) scan and Q₀ finder

`RTE2Model.find_Q0(T, nk_bz=200, nk_q=50)`:

```
1. Build 1D grid along a*:
     q_grid = linspace(0.02, 0.5, nk_q) * 2π
     Q_axial   = [(q, 0.0) for q in q_grid]
     Q_diag    = [(q, q)   for q in q_grid]     # control
2. For each Q, compute χ scalar:
     chi = susceptibility_tensor(Qx, Qy, T, params, nk=nk_bz)
     chi_val = chi_combinations(chi)["diag_aa"]
3. Detect local maxima on chi_axial vs q with
   scipy.signal.find_peaks (prominence gating).
4. Return top-2 axial peaks as
     [(q_low, 0), (q_high, 0)]
   plus diagnostic arrays (q_grid, chi_axial, chi_diag) for plotting.
   If fewer than 2 peaks, return whatever exists and log a warning.
```

**Symmetry.** C₄ symmetry of the model means (q, 0) and (0, q) are
degenerate; downstream code handles orientation partners explicitly.
Phase A does not scan (0, q).

**Diagonal (q, q) scan** is a control: we expect χ(q, q) < χ(q, 0)
across the range, confirming the CDW is axial (not diagonal like RTE3).

**Efficiency.** Naïve cost is `nk_q · nk_bz²` susceptibility
evaluations. At `(nk_q, nk_bz) = (50, 200)` this is ~2M grid points
per T. Acceptable on a laptop. An optimization noting that
`bands_and_projectors(k)` is Q-independent will halve the work; it is
worth including in the initial refactor since the T-scan reuses this
hot loop.

**Verification target.** At the fitted TB parameters and T = 300 K,
`find_Q0` should return `q_low ≈ 0.15 a*` and `q_high ≈ 0.25 a*` to
within a few percent. If not, the TB fit is not converged — closed
loop between the fit and the χ-scan.

## Ginzburg–Landau flow at Q₀ candidates

For each `Q0 ∈ {Q_low, Q_high}`, existing machinery from
`gl_theory.py` is called unchanged.

### Quadratic

- Compute χ tensor at `(Q0, T)`.
- Extract `(a₀, a_x, a_y, a_z, λ)` per Eq. 24–25.
- Diagonalize the (Δ⁰, Δˣ) block → `(a_−, a_+, a_y, a_z)`.
- **T_c**: temperature at which `a_−(T)` crosses zero. `a_−` is the
  leading CDW channel; `g` (coupling) enters via `2/g` and is treated
  as a swept parameter (e.g. `g ∈ {2.0, 2.5, 3.0, 3.5}`).

### Quartic

Evaluated at T just below T_c (e.g. 0.95 T_c):

- Compute `b_−, b_z, c′, c″` per Eq. A11 / Eq. 27.
- Coexistence: `((c′ + c″) − |c′ − c″|)² < b_− b_z` → double-Δ,
  else single-Δ A_g.
- If double-Δ and c′ > c″: B₂ᵤ. If c′ < c″: B₁_g (would require
  phonon coupling from Alekseev Eq. 19–20 — reported but flagged
  "not from bare electronic GL").

### Winner selection

Compare Q_low vs Q_high:

- Higher T_c → wins as T is lowered from above.
- At the winner's T_c, report `|Δ|² = -a_− / (2 b_−)` and
  `F_min = -a_−² / (4 b_−)` for completeness.

The expected story (to be tested by the numbers): Q_low = 0.15 a\*
wins at high T (larger nesting phase space), Q_high = 0.25 a\* would
win at low T with lattice / umklapp coupling — but Phase A only sees
bare electronic GL, so Q_high may look sub-leading here.

## Phase A deliverables — `rte2_cdw_walkthrough.ipynb`

Six figures, structured to mirror the RTE3 walkthrough:

- **Fig. 1 — Fermi surface with orbital texture.** FS in (a*, b*)
  plane, colored by `|a_px|²` (green) and `|a_py|²` (red). Nesting
  arrows for q_low and q_high overlaid. Should visually match the
  reference figure.
- **Fig. 2 — χ(q) scans.** Two curves per panel: χ(q, 0) axial (solid)
  and χ(q, q) diagonal (dashed control). Peaks at q_low ≈ 0.15 and
  q_high ≈ 0.25 marked. Diagonal below axial. Three temperatures
  (100, 300, 500 K).
- **Fig. 3 — χ(Q_peak, T) vs T** for both Q candidates. Divergence
  temperature ≈ mean-field T_c.
- **Fig. 4 — Quadratic GL `a_−(T)`** at Q_low and Q_high on same
  axes. Zero-crossings marked. Coupling `g` swept over 3–4 values.
- **Fig. 5 — Quartic phase check** at the winning Q₀: coexistence
  ratio LHS/RHS of Eq. 29 vs T (or g). Above 1 → single-Δ A_g;
  below 1 → double-Δ.
- **Fig. 6 — Real-space orbital density** in the winning phase:
  ρ_px(r), ρ_py(r), ρ(r) on a small lattice using Q₀_winner and
  the GL amplitudes.

Figures cached to `figures/rte2_phase_a/` for review outside the
notebook.

## Roadmap for Phases B, C

Detailed at `docs/superpowers/roadmap-rte2.md`. Summary:

- **Phase B — Dirac semimetal band structure.** Diagonalize the 6×6
  CDW-reconstructed `H_cdw_k` along Γ–X–M–Y at the winning
  (Q₀, Δ⁰, Δ_z). Identify gap openings and residual Dirac nodes.
  Reproduce the story of Adv. Mater. 33, 2101591 (2021).
- **Phase C — LaTe₂₋ₓSbₓ filling scan.** Sweep μ (Sb doping proxy),
  re-run `find_Q0` + quadratic GL at each μ. Map `T_c(x)`,
  `Q_peaks(x)`, orbital-texture evolution. Reproduces the CDW
  stability plot of the LaTe₂₋ₓSbₓ PRB.

Each phase gets its own spec + implementation plan before code starts.

## Success criteria for Phase A

Phase A is complete when the walkthrough notebook runs end-to-end
and:

1. `rte3_cdw_walkthrough.ipynb` still produces the same figures as
   before the refactor (regression check).
2. `LaTe2Model()` fit converges; fitted `TBParams` reproduce the
   diamond-shape FS.
3. `find_Q0` returns two axial peaks within a few percent of
   0.15 a\* and 0.25 a\*.
4. Diagonal (q, q) susceptibility sits below axial (q, 0) across
   the scanned range.
5. Quadratic + quartic GL are computed without numerical instability
   at both Q candidates; T_c and phase (single-Δ / double-Δ) are
   reported for each.
6. All six deliverable figures render and are cached.
