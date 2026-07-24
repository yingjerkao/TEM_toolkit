# RTE2 CDW — roadmap

Three phases adapting the Alekseev (RTE3) framework to LaTe₂ and related
rare-earth ditellurides. Each phase gets its own design spec and
implementation plan.

## Physical anchors

- **Alekseev et al., PRB 110, 205103 (2024)** — 2-orbital (pₓ, p_y)
  tight-binding + GL machinery for RTE3 CDW. Basis for the existing code.
- **Adv. Mater. 33, 2101591 (2021)**, doi:10.1002/adma.202101591 —
  *Band Engineering of Dirac Semimetals Using CDWs.* CDW-protected
  Dirac cones in RTE2-like systems. Phase B target.
- **LaTe₂₋ₓSbₓ PRB** — CDW stability under band-filling variation via
  Sb substitution. Phase C target.
- **In-house TEM data** — two CDW states in LaTe₂: incommensurate
  q ≈ 0.15 a\*, discommensurate q ≈ 0.25 a\*.

## Phase A — Alekseev machinery at RTE2 filling

**Goal.** Reproduce the Alekseev workflow (χ tensor, GL quadratic,
GL quartic, coexistence, orbital textures) for LaTe₂ with axial
nesting geometry. Match the two observed q values.

**Deliverables.** Shared-core refactor (`tem_cdw/`), `LaTe2Model()`
fit, numerical Q₀ finder, `rte2_cdw_walkthrough.ipynb` with six
figures.

**Spec.** `docs/superpowers/specs/2026-07-22-rte2-cdw-phase-a-design.md`

## Phase B — Dirac semimetal band structure from CDW

**Goal.** Reproduce the "band engineering" story of Adv. Mater. 33,
2101591 (2021): CDW opens gaps but leaves Dirac cones.

**Sketch.**

- Take the winning Q₀ and mean-field (Δ⁰, Δ_z) from Phase A.
- Diagonalize the 6×6 `H_cdw_k` from `fermi_surface.py` along
  Γ–X–M–Y (and any other line the Adv. Mater. paper uses).
- Identify gap openings and residual band crossings; check
  symmetry protection.
- New deliverables: band-structure plot with orbital projection,
  gap map on the FS, possibly a small tight-binding sanity check
  vs the Adv. Mater. figures.

**Prerequisites.** Phase A complete.

## Phase C — LaTe₂₋ₓSbₓ filling scan

**Goal.** Reproduce the CDW stability vs Sb doping story from the
LaTe₂₋ₓSbₓ PRB. Sb substitutes for Te and shifts μ.

**Sketch.**

- Add a μ-sweep loop wrapping the Phase A pipeline.
- For each μ (mapped to x via a linear rigid-band approximation),
  re-fit only μ (keeping hoppings), re-run `find_Q0`, re-run
  quadratic GL to get T_c(x) and Q_peaks(x).
- Deliverables: T_c(x) plot, Q(x) plot, orbital-texture evolution,
  optional phase diagram in (T, x).

**Prerequisites.** Phase A complete. Phase B independent; can run in
parallel or after.

## Explicitly deferred (beyond A/B/C)

- Electron–phonon coupling, discommensurate soliton lattice, ¼
  lock-in physics for q_high.
- Bidirectional (2Q / checkerboard) CDW mean-field.
- Bilayer / extended-hopping tight-binding models if the 2-orbital
  form is insufficient (surfaced explicitly by Phase A's fit
  report).
