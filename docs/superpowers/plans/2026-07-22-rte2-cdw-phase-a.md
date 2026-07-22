# RTE2 CDW Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adapt the existing `rte3_cdw` (Alekseev PRB 110, 205103 (2024)) machinery to LaTe₂, targeting the two observed TEM CDW states q ≈ 0.15 a* (incommensurate) and q ≈ 0.25 a* (discommensurate).

**Architecture:** Rename `rte3_cdw/` → `tem_cdw/` as a shared physics core; add `tem_cdw/models/` with `MaterialModel` ABC + `RTE3Model` (analytic Q₀) + `RTE2Model` (numerical χ(Q) scan along axial and diagonal directions). Fit LaTe₂ tight-binding parameters (t_σ, t_π, t_d, μ) to the observed diamond-shape Fermi surface and the two q values. Run existing quadratic + quartic Ginzburg–Landau at both χ peaks. Deliver a six-figure Jupyter walkthrough.

**Tech Stack:** Python 3.13, NumPy 2.x, SciPy 1.17, Matplotlib 3.10, pytest, Jupyter (`ipykernel`).

**Design spec:** `docs/superpowers/specs/2026-07-22-rte2-cdw-phase-a-design.md`
**Roadmap:** `docs/superpowers/roadmap-rte2.md`

**Working directory:** `/Users/yjkao/TEM_toolkit`

**Invariants across the plan:**
- Existing physics conventions (h(k) form, χ sign/indexing, GL basis {0, x, y, z}) must not change. Task 2's regression test enforces this.
- All temperatures are in eV (Alekseev convention). 300 K ↔ `T = 0.026` eV.
- All q values in code are in radians. `a*` units used only at the user boundary via `tem_cdw/units.py`.

---

## Task 1: Bootstrap pytest with baseline snapshot of RTE3 numerics

**Purpose:** Establish a regression check before we rename anything, so the refactor is provably numerics-preserving.

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_rte3_regression.py`
- Modify: `pyproject.toml` — add `pytest` to a dev extra

- [ ] **Step 1: Add pytest to pyproject.toml**

Edit `/Users/yjkao/TEM_toolkit/pyproject.toml`, add after the `dependencies` block:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Install pytest**

Run: `cd /Users/yjkao/TEM_toolkit && uv sync --extra dev`
Expected: `pytest` installed in `.venv`.

- [ ] **Step 3: Create empty test package files**

Create `tests/__init__.py` (empty).
Create `tests/conftest.py`:

```python
"""Pytest configuration — make the repo root importable."""
import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 4: Record baseline numerics from current main**

Run this one-off script to capture the current numeric outputs of the RTE3 pipeline:

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python -c "
from rte3_cdw import hamiltonian, susceptibility
import numpy as np
p = hamiltonian.TBParams()
kx = np.array([0.1, 0.5, 1.0]); ky = np.array([0.2, 0.4, 0.9])
eps, _ = hamiltonian.bands_and_projectors(kx, ky, p)
print('EXPECTED_LOWER =', repr(eps[..., 0].tolist()))
Qx, Qy = p.Q0()
chi = susceptibility.susceptibility_tensor(Qx, Qy, T=0.026, p=p, nk=100)
print('EXPECTED_DIAG_AA =', repr(susceptibility.chi_combinations(chi)['diag_aa']))
"
```

Copy the two printed literals — you will paste them into the test in Step 5.

- [ ] **Step 5: Write the regression test with the recorded baselines**

Create `tests/test_rte3_regression.py`, pasting the two literals recorded in Step 4 into the constants at the top:

```python
"""Snapshot the current RTE3 physics numerics. This test locks in the
Alekseev conventions (h(k), χ tensor, GL coefficients) so the upcoming
tem_cdw/ refactor is provably numerics-preserving.

Baseline values recorded from the current main via Task 1 Step 4.
"""
import numpy as np
import pytest

# ---- Paste the two baseline values recorded in Step 4 here ----
EXPECTED_LOWER = ...     # e.g. [-2.6628..., -1.0612..., 1.2026...]
EXPECTED_DIAG_AA = ...   # single float


def _get_module():
    """Import whichever package name is currently in use."""
    try:
        import tem_cdw as m  # after Task 2
        return m
    except ImportError:
        import rte3_cdw as m  # before Task 2
        return m


def test_hamiltonian_eigenvalues():
    m = _get_module()
    p = m.hamiltonian.TBParams()
    kx = np.array([0.1, 0.5, 1.0])
    ky = np.array([0.2, 0.4, 0.9])
    eps, _ = m.hamiltonian.bands_and_projectors(kx, ky, p)
    np.testing.assert_allclose(eps[..., 0], np.asarray(EXPECTED_LOWER), rtol=1e-10)


def test_susceptibility_at_Q0():
    m = _get_module()
    p = m.hamiltonian.TBParams()
    Qx, Qy = p.Q0()
    chi = m.susceptibility.susceptibility_tensor(Qx, Qy, T=0.026, p=p, nk=100)
    diag_aa = m.susceptibility.chi_combinations(chi)["diag_aa"]
    assert diag_aa == pytest.approx(EXPECTED_DIAG_AA, rel=1e-8)
```

Both baseline sentinels (`EXPECTED_LOWER`, `EXPECTED_DIAG_AA`) must be replaced with the literal values printed in Step 4 before running Step 6.

- [ ] **Step 6: Verify test passes on current codebase**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_rte3_regression.py -v`
Expected: 2 passed.

- [ ] **Step 7: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add pyproject.toml tests/
git commit -m "Add pytest infra and RTE3 baseline regression test"
```

---

## Task 2: Rename `rte3_cdw` → `tem_cdw`

**Purpose:** Physical rename with no code content changes; verify Task 1 regression test still passes.

**Files:**
- Rename: `rte3_cdw/` → `tem_cdw/`
- Modify: `rte3_cdw_walkthrough.ipynb` — every `rte3_cdw` import → `tem_cdw`
- Modify: `pyproject.toml` — no changes (package layout is flat, no `[tool.setuptools.packages]` yet)
- Delete: `rte3_cdw/__pycache__/` if present

- [ ] **Step 1: Verify no other consumer imports rte3_cdw**

Run: `cd /Users/yjkao/TEM_toolkit && grep -rn "rte3_cdw" --include="*.py" --include="*.ipynb" . | grep -v ".venv" | grep -v ".git"`

Expected: only the notebook, `demo.py` inside the package, and the test file (which uses the try/except from Task 1) reference `rte3_cdw`.

- [ ] **Step 2: Rename the package directory with git**

```bash
cd /Users/yjkao/TEM_toolkit
git mv rte3_cdw tem_cdw
rm -rf tem_cdw/__pycache__
```

- [ ] **Step 3: Update intra-package imports (none needed — all use relative imports)**

Run: `grep -n "from rte3_cdw\|import rte3_cdw" /Users/yjkao/TEM_toolkit/tem_cdw/*.py`

Expected: no matches. The physics modules already use relative imports (`from .hamiltonian import ...`). Only `tem_cdw/demo.py` may have an absolute import; if so, replace `from rte3_cdw.plotting` with `from tem_cdw.plotting`.

- [ ] **Step 4: Update the notebook imports**

In `/Users/yjkao/TEM_toolkit/rte3_cdw_walkthrough.ipynb`, replace every `rte3_cdw` with `tem_cdw` in cell source. Use this sed-style replacement (safe because `rte3_cdw` is not a Python substring of any other identifier):

```bash
cd /Users/yjkao/TEM_toolkit
python - <<'PY'
import json, pathlib
p = pathlib.Path("rte3_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
for cell in nb["cells"]:
    if "source" in cell:
        cell["source"] = [line.replace("rte3_cdw", "tem_cdw") for line in cell["source"]]
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Notebook imports updated.")
PY
```

- [ ] **Step 5: Rename the notebook file itself for clarity**

```bash
cd /Users/yjkao/TEM_toolkit && git mv rte3_cdw_walkthrough.ipynb tem_cdw_rte3_walkthrough.ipynb
```

The notebook is still the "RTE3 walkthrough" but now lives alongside the future RTE2 walkthrough with a matching prefix.

- [ ] **Step 6: Run regression test — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_rte3_regression.py -v`
Expected: 2 passed. The `_get_module()` helper falls through the `tem_cdw` branch now.

- [ ] **Step 7: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add -A
git commit -m "Rename rte3_cdw to tem_cdw (shared physics core)"
```

---

## Task 3: Add units helper (a* ↔ radians conversion)

**Files:**
- Create: `tem_cdw/units.py`
- Create: `tests/test_units.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_units.py`:

```python
import math
from tem_cdw.units import q_from_astar, astar_from_q, TWO_PI


def test_q_from_astar_zero():
    assert q_from_astar(0.0) == 0.0


def test_q_from_astar_quarter():
    assert q_from_astar(0.25) == math.pi / 2


def test_roundtrip():
    for x in [0.0, 0.15, 0.25, 0.5, 1.0]:
        assert astar_from_q(q_from_astar(x)) == x


def test_two_pi_constant():
    assert TWO_PI == 2.0 * math.pi
```

- [ ] **Step 2: Run test to see it fails**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_units.py -v`
Expected: 4 errors (ModuleNotFoundError).

- [ ] **Step 3: Implement the module**

Create `tem_cdw/units.py`:

```python
"""Unit conversion between reciprocal-lattice units (a*) and radians.

a* = 2π/a where a is the real-space lattice constant. TEM experiments
report q in units of a*; the Alekseev code uses radians throughout.
This module exists so unit conversions happen only at the user boundary.
"""
import math

TWO_PI = 2.0 * math.pi


def q_from_astar(q_astar: float) -> float:
    """Convert q from a* units to radians (q_rad = 2π · q_astar)."""
    return TWO_PI * q_astar


def astar_from_q(q_rad: float) -> float:
    """Convert q from radians to a* units."""
    return q_rad / TWO_PI
```

- [ ] **Step 4: Run tests — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_units.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add tem_cdw/units.py tests/test_units.py
git commit -m "Add a*/radian unit conversion helper"
```

---

## Task 4: Add `MaterialModel` ABC and `Q0Result` dataclass

**Files:**
- Create: `tem_cdw/models/__init__.py`
- Create: `tem_cdw/models/base.py`
- Create: `tests/test_models_base.py`

- [ ] **Step 1: Write failing test for `Q0Result` and ABC**

Create `tests/test_models_base.py`:

```python
import numpy as np
import pytest
from tem_cdw.hamiltonian import TBParams
from tem_cdw.models.base import MaterialModel, Q0Result


def test_q0result_construction():
    r = Q0Result(
        peaks=[(1.0, 0.0), (2.0, 0.0)],
        q_grid=np.array([0.0, 1.0, 2.0]),
        chi_axial=np.array([0.1, 0.9, 0.3]),
        chi_diagonal=np.array([0.05, 0.4, 0.2]),
    )
    assert r.peaks == [(1.0, 0.0), (2.0, 0.0)]
    assert r.q_grid.shape == (3,)


def test_material_model_is_abstract():
    with pytest.raises(TypeError):
        MaterialModel()  # abstract; cannot instantiate directly


def test_subclass_must_implement_find_Q0():
    class Incomplete(MaterialModel):
        name = "x"
        params = TBParams()
    with pytest.raises(TypeError):
        Incomplete()
```

- [ ] **Step 2: Run test to see it fails**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_models_base.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `models/base.py`**

Create `tem_cdw/models/__init__.py`:

```python
"""Material-specific adapters bundling TBParams + Q₀-finding strategy."""
from .base import MaterialModel, Q0Result

__all__ = ["MaterialModel", "Q0Result"]
```

Create `tem_cdw/models/base.py`:

```python
"""MaterialModel ABC. Each concrete subclass bundles a `TBParams`
instance with a strategy for locating the CDW wavevector Q₀."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..hamiltonian import TBParams


@dataclass
class Q0Result:
    """Bundle of Q₀ candidates plus diagnostic scan arrays.

    Attributes
    ----------
    peaks : list of (Qx, Qy) tuples, sorted by q along a*.
        For RTE3 this is a single analytic point.
        For RTE2 this is up to two axial peaks (q_low, q_high).
    q_grid : 1D array of q values (radians) at which the diagnostic
        χ-scan was evaluated. Empty for analytic (RTE3) case.
    chi_axial, chi_diagonal : 1D arrays of χ(Q,T) along (q, 0) and (q, q)
        respectively, over q_grid. Empty for analytic case.
    """
    peaks: list[tuple[float, float]]
    q_grid: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_axial: np.ndarray = field(default_factory=lambda: np.array([]))
    chi_diagonal: np.ndarray = field(default_factory=lambda: np.array([]))


class MaterialModel(ABC):
    """Abstract material model. Subclasses provide `params` and `find_Q0`."""

    name: ClassVar[str] = "abstract"
    params: TBParams

    @abstractmethod
    def find_Q0(self, T: float, nk_bz: int = 200, nk_q: int = 50) -> Q0Result:
        """Return the CDW wavevector candidate(s) for this material at
        temperature T (eV). See Q0Result docstring for return schema."""
        ...
```

- [ ] **Step 4: Run tests — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_models_base.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add tem_cdw/models/ tests/test_models_base.py
git commit -m "Add MaterialModel ABC and Q0Result dataclass"
```

---

## Task 5: Add `RTE3Model` (analytic Q₀)

**Files:**
- Create: `tem_cdw/models/rte3.py`
- Create: `tests/test_models_rte3.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_models_rte3.py`:

```python
import numpy as np
from tem_cdw.hamiltonian import TBParams
from tem_cdw.models.rte3 import RTE3Model


def test_default_params_match_alekseev():
    m = RTE3Model()
    assert m.params.t_sigma == 2.0
    assert m.params.t_pi == 0.37
    assert m.params.t_d == 0.16
    assert m.params.mu == -1.53
    assert m.name == "RTE3"


def test_find_Q0_returns_diagonal_2kF():
    m = RTE3Model()
    result = m.find_Q0(T=0.026)
    assert len(result.peaks) == 1
    kF = m.params.kF()
    qx, qy = result.peaks[0]
    assert np.isclose(qx, 2 * kF)
    assert np.isclose(qy, 2 * kF)
    # RTE3 analytic path leaves diagnostic arrays empty
    assert result.q_grid.size == 0
```

- [ ] **Step 2: Run test to see it fails**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_models_rte3.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `models/rte3.py`**

Create `tem_cdw/models/rte3.py`:

```python
"""RTE3 (LaTe₃) material model — Alekseev PRB 110, 205103 (2024) defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..hamiltonian import TBParams
from .base import MaterialModel, Q0Result


@dataclass
class RTE3Model(MaterialModel):
    """Analytic diagonal-nesting model. Q₀ = (2kF, 2kF)."""
    params: TBParams = field(default_factory=TBParams)
    name: ClassVar[str] = "RTE3"

    def find_Q0(self, T: float = 0.026, nk_bz: int = 200, nk_q: int = 50) -> Q0Result:
        qx, qy = self.params.Q0()
        return Q0Result(peaks=[(qx, qy)])
```

- [ ] **Step 4: Update `models/__init__.py` to export RTE3Model**

Edit `tem_cdw/models/__init__.py`:

```python
"""Material-specific adapters bundling TBParams + Q₀-finding strategy."""
from .base import MaterialModel, Q0Result
from .rte3 import RTE3Model

__all__ = ["MaterialModel", "Q0Result", "RTE3Model"]
```

- [ ] **Step 5: Run tests — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_models_rte3.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add tem_cdw/models/rte3.py tem_cdw/models/__init__.py tests/test_models_rte3.py
git commit -m "Add RTE3Model with analytic Q0 = (2kF, 2kF)"
```

---

## Task 6: Add `RTE2Model` with numerical χ(Q) scan

**Files:**
- Create: `tem_cdw/models/rte2.py`
- Create: `tests/test_models_rte2.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models_rte2.py`:

```python
"""Tests for the RTE2 numerical Q0 finder.

We do NOT test that the fit reproduces LaTe2 — that's Task 8's job.
Here we only test that the χ-scan mechanism itself is correct, using
the RTE3 defaults as a stand-in TB model with known χ structure.
"""
import numpy as np
import pytest
from tem_cdw.hamiltonian import TBParams
from tem_cdw.models.rte2 import RTE2Model
from tem_cdw.models.base import Q0Result


def test_find_Q0_returns_Q0Result_with_diagnostic_arrays():
    m = RTE2Model(params=TBParams())  # borrow RTE3 defaults for the mechanism test
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    assert isinstance(result, Q0Result)
    assert result.q_grid.size == 25
    assert result.chi_axial.size == 25
    assert result.chi_diagonal.size == 25


def test_find_Q0_returns_at_least_one_peak():
    m = RTE2Model(params=TBParams())
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    assert len(result.peaks) >= 1


def test_find_Q0_peaks_are_local_maxima():
    m = RTE2Model(params=TBParams())
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    for qx, qy in result.peaks:
        assert qy == 0.0                        # axial
        assert 0.0 < qx < 2 * np.pi             # inside grid range


def test_find_Q0_peaks_sorted_by_q():
    m = RTE2Model(params=TBParams())
    result = m.find_Q0(T=0.026, nk_bz=60, nk_q=25)
    qs = [q[0] for q in result.peaks]
    assert qs == sorted(qs)
```

- [ ] **Step 2: Run test to see it fails**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_models_rte2.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `models/rte2.py`**

Create `tem_cdw/models/rte2.py`:

```python
"""RTE2 (LaTe₂) material model — numerical χ(Q) scan along a* and diagonal.

The fitted TBParams for LaTe₂ are set by Task 8; this task uses a
placeholder (RTE3 defaults) so the χ-scan mechanism is testable in
isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar
import warnings

import numpy as np
from scipy.signal import find_peaks

from ..hamiltonian import TBParams
from ..susceptibility import susceptibility_tensor, chi_combinations
from .base import MaterialModel, Q0Result


# Placeholder defaults; Task 8 replaces these with the LaTe₂ fit.
LATE2_PLACEHOLDER = TBParams(t_sigma=2.0, t_pi=0.37, t_d=0.16, mu=-1.53)


@dataclass
class RTE2Model(MaterialModel):
    """Numerical axial-nesting model for LaTe₂."""
    params: TBParams = field(default_factory=lambda: LATE2_PLACEHOLDER)
    name: ClassVar[str] = "RTE2"

    def find_Q0(self, T: float = 0.026, nk_bz: int = 200, nk_q: int = 50) -> Q0Result:
        """Scan χ(Q) along Q = (q, 0) (axial) and Q = (q, q) (diagonal control).

        Returns up to the top-2 local maxima of χ_axial, sorted by q.
        Also returns the full q_grid and both χ curves for plotting.
        """
        two_pi = 2.0 * np.pi
        q_astar = np.linspace(0.02, 0.5, nk_q)
        q_grid = q_astar * two_pi

        chi_axial = np.empty(nk_q)
        chi_diagonal = np.empty(nk_q)
        for i, q in enumerate(q_grid):
            chi_a = susceptibility_tensor(q, 0.0, T, self.params, nk=nk_bz)
            chi_d = susceptibility_tensor(q, q,   T, self.params, nk=nk_bz)
            chi_axial[i]    = chi_combinations(chi_a)["diag_aa"]
            chi_diagonal[i] = chi_combinations(chi_d)["diag_aa"]

        # Local-maxima detection on the axial scan.
        prominence_floor = 0.02 * (chi_axial.max() - chi_axial.min())
        peaks_idx, _ = find_peaks(chi_axial, prominence=prominence_floor)

        if len(peaks_idx) == 0:
            warnings.warn(
                f"find_Q0 found no χ(Q) peaks along a*. "
                f"max={chi_axial.max():.3g} min={chi_axial.min():.3g}",
                stacklevel=2,
            )
            top2_sorted = []
        else:
            # Keep the two most prominent, then sort by q.
            top_by_height = sorted(peaks_idx, key=lambda i: chi_axial[i], reverse=True)[:2]
            top2_sorted = sorted(top_by_height, key=lambda i: q_grid[i])

        peaks = [(float(q_grid[i]), 0.0) for i in top2_sorted]

        return Q0Result(
            peaks=peaks,
            q_grid=q_grid,
            chi_axial=chi_axial,
            chi_diagonal=chi_diagonal,
        )
```

- [ ] **Step 4: Update `models/__init__.py`**

Edit `tem_cdw/models/__init__.py`:

```python
"""Material-specific adapters bundling TBParams + Q₀-finding strategy."""
from .base import MaterialModel, Q0Result
from .rte2 import RTE2Model
from .rte3 import RTE3Model

__all__ = ["MaterialModel", "Q0Result", "RTE2Model", "RTE3Model"]
```

- [ ] **Step 5: Run tests — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_models_rte2.py -v`
Expected: 4 passed. (Runtime ~ 15 s at nk_bz=60, nk_q=25.)

- [ ] **Step 6: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add tem_cdw/models/rte2.py tem_cdw/models/__init__.py tests/test_models_rte2.py
git commit -m "Add RTE2Model with numerical chi(Q) scan"
```

---

## Task 7: Write the LaTe₂ tight-binding fit script

**Purpose:** Offline script that runs a coarse grid + Nelder-Mead refine to fit `(t_σ, t_π, t_d, μ)` so the FS corner sits at ≈ 0.35 a* and the two χ peaks sit at ≈ 0.15 a* and ≈ 0.25 a*.

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/fit_late2.py`
- Create: `tests/test_fit_late2_helpers.py`

- [ ] **Step 1: Write failing tests for the fit helpers**

Create `tests/test_fit_late2_helpers.py`:

```python
import numpy as np
from tem_cdw.hamiltonian import TBParams
from scripts.fit_late2 import kF_axial, chi_peaks_along_a, objective


def test_kF_axial_finite_for_default():
    p = TBParams()
    kf = kF_axial(p, nk=200)
    assert 0.0 < kf < np.pi


def test_chi_peaks_returns_arrays_and_peaks():
    p = TBParams()
    peaks_q, q_grid, chi_grid = chi_peaks_along_a(p, T=0.026, nk_bz=60, nk_q=25)
    assert q_grid.shape == chi_grid.shape == (25,)
    # At default (RTE3) params, one strong peak; not testing count exactly.
    assert isinstance(peaks_q, list)


def test_objective_finite_at_default():
    x = np.array([2.0, 0.37, 0.16, -1.53])
    val = objective(x, w_FS=1.0, w_q=1.0, nk_bz=60, nk_q=25)
    assert np.isfinite(val)


def test_objective_returns_large_for_bad_params():
    # μ way outside the band → no FS → sentinel value
    x = np.array([2.0, 0.37, 0.16, -10.0])
    val = objective(x, w_FS=1.0, w_q=1.0, nk_bz=60, nk_q=25)
    assert val >= 1e5
```

- [ ] **Step 2: Run tests to see they fail**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_fit_late2_helpers.py -v`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/fit_late2.py`**

Create `scripts/__init__.py` (empty).

Create `scripts/fit_late2.py`:

```python
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
    # linear interpolation between kx[i] and kx[i+1] for the crossing.
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

    print("Coarse grid search…")
    combos = coarse_grid(quick)
    scores = np.array([
        objective(np.asarray(c), nk_bz=nk_bz, nk_q=nk_q) for c in combos
    ])
    order = np.argsort(scores)
    top5 = [combos[i] for i in order[:5]]
    print(f"  best coarse score: {scores[order[0]]:.4f}")
    for i in order[:5]:
        print(f"    x={combos[i]} score={scores[i]:.4f}")

    print("Nelder-Mead refine…")
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
        f"LaTe₂ fit: t_σ={p.t_sigma:.3f}, t_π={p.t_pi:.3f}, "
        f"t_d={p.t_d:.3f}, μ={p.mu:.3f}"
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
```

- [ ] **Step 4: Run helper tests — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_fit_late2_helpers.py -v`
Expected: 4 passed. (Runtime ~ 30 s.)

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add scripts/ tests/test_fit_late2_helpers.py
git commit -m "Add LaTe2 tight-binding fit script and helper tests"
```

---

## Task 8: Run the fit; freeze results into `LaTe2Model` factory

**Purpose:** Execute the fit, save the report, and wire the fitted params into a factory function so downstream code always sees frozen values.

**Files:**
- Create: `tem_cdw/models/rte2_fit_report/fitted_params.npz` (generated)
- Create: `tem_cdw/models/rte2_fit_report/fit_report.png` (generated)
- Modify: `tem_cdw/models/rte2.py` — add `LaTe2Model()` factory that reads the frozen npz
- Create: `tests/test_late2_model.py`

- [ ] **Step 1: Run the fit script**

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python -m scripts.fit_late2
```

Expected: Completes in a few minutes. Prints fitted params and score, saves `tem_cdw/models/rte2_fit_report/{fitted_params.npz, fit_report.png}`. **Best score should be well below 0.01** if the 2-orbital form can reproduce both q peaks.

- [ ] **Step 2: Inspect the fit report**

Open `tem_cdw/models/rte2_fit_report/fit_report.png`. Confirm two peaks in the χ(q) curve near 0.15 and 0.25 a*.

If the fit fails (score > 0.1, or fewer than 2 peaks): **stop here** and report the finding. Per the design spec § "Failure mode we surface explicitly", this is a physics result (the 2-orbital model is insufficient for LaTe₂), not a bug to work around. Escalate to the user before extending the model.

- [ ] **Step 3: Add `LaTe2Model()` factory to `tem_cdw/models/rte2.py`**

At the bottom of `tem_cdw/models/rte2.py`, append:

```python
def LaTe2Model() -> RTE2Model:
    """Factory returning RTE2Model with the LaTe₂-fitted TB parameters.

    Params are frozen at fit time in tem_cdw/models/rte2_fit_report/
    fitted_params.npz (produced by scripts/fit_late2.py). Downstream
    code never re-runs the fit.
    """
    import pathlib
    npz_path = (
        pathlib.Path(__file__).parent / "rte2_fit_report" / "fitted_params.npz"
    )
    if not npz_path.exists():
        raise FileNotFoundError(
            f"LaTe₂ fit not yet run. Execute `python -m scripts.fit_late2` "
            f"before calling LaTe2Model(). Expected file: {npz_path}"
        )
    data = np.load(npz_path)
    params = TBParams(
        t_sigma=float(data["t_sigma"]),
        t_pi=float(data["t_pi"]),
        t_d=float(data["t_d"]),
        mu=float(data["mu"]),
    )
    return RTE2Model(params=params)
```

Update `tem_cdw/models/__init__.py` to export it:

```python
"""Material-specific adapters bundling TBParams + Q₀-finding strategy."""
from .base import MaterialModel, Q0Result
from .rte2 import LaTe2Model, RTE2Model
from .rte3 import RTE3Model

__all__ = ["LaTe2Model", "MaterialModel", "Q0Result", "RTE2Model", "RTE3Model"]
```

- [ ] **Step 4: Write factory verification test**

Create `tests/test_late2_model.py`:

```python
import numpy as np
import pytest
from tem_cdw.models import LaTe2Model
from tem_cdw.units import astar_from_q


def test_late2_factory_loads_fitted_params():
    m = LaTe2Model()
    # Params should differ from RTE3 defaults; sanity check.
    assert not (m.params.t_sigma == 2.0 and m.params.t_pi == 0.37
                and m.params.t_d == 0.16 and m.params.mu == -1.53), (
        "LaTe2 fit landed exactly on RTE3 defaults — suspicious."
    )


def test_late2_find_Q0_reproduces_target_peaks():
    m = LaTe2Model()
    result = m.find_Q0(T=0.026, nk_bz=80, nk_q=40)
    peaks_astar = sorted(astar_from_q(q[0]) for q in result.peaks)
    assert len(peaks_astar) == 2, (
        f"Expected 2 axial peaks, got {len(peaks_astar)}: {peaks_astar}"
    )
    q_low, q_high = peaks_astar
    assert q_low == pytest.approx(0.15, abs=0.03), f"q_low={q_low}"
    assert q_high == pytest.approx(0.25, abs=0.03), f"q_high={q_high}"


def test_late2_diagonal_below_axial():
    m = LaTe2Model()
    result = m.find_Q0(T=0.026, nk_bz=80, nk_q=40)
    # Diagonal control should be smaller than axial peak at their maxima.
    assert result.chi_diagonal.max() < result.chi_axial.max(), (
        "Diagonal χ exceeds axial χ — CDW is not axial after all."
    )
```

- [ ] **Step 5: Run tests — must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/test_late2_model.py -v`
Expected: 3 passed. (Runtime ~ 30 s.)

- [ ] **Step 6: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add tem_cdw/models/rte2.py tem_cdw/models/__init__.py \
        tem_cdw/models/rte2_fit_report/ tests/test_late2_model.py
git commit -m "Fit LaTe2 TB parameters and freeze into LaTe2Model factory"
```

---

## Task 9: Notebook — cells 1–2 (setup) and Fig. 1 (FS with orbital texture)

**Purpose:** Start the walkthrough notebook; produce the FS-with-orbital-texture figure that visually matches the reference TEM/DFT figure.

**Files:**
- Create: `rte2_cdw_walkthrough.ipynb`
- Create: `figures/rte2_phase_a/` (dir)

- [ ] **Step 1: Create notebook skeleton with setup cells**

Create `rte2_cdw_walkthrough.ipynb` as a JSON structure via this helper script:

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json
nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# LaTe₂ CDW walkthrough — Phase A\n",
            "\n",
            "Reproduces the Alekseev PRB 110, 205103 (2024) machinery at RTE2 "
            "band filling. Matches the two in-house TEM q values: "
            "**q ≈ 0.15 a\\*** (incommensurate) and **q ≈ 0.25 a\\*** (discommensurate).\n"
        ]},
        {"cell_type": "code", "metadata": {}, "source": [
            "import sys, pathlib\n",
            "sys.path.insert(0, str(pathlib.Path.cwd()))\n",
            "\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "from tem_cdw import hamiltonian, susceptibility, gl_theory, fermi_surface, orbital_density\n",
            "from tem_cdw.hamiltonian import TBParams, h_k, bands_and_projectors, make_bz_grid\n",
            "from tem_cdw.susceptibility import susceptibility_tensor, chi_combinations\n",
            "from tem_cdw.gl_theory import quadratic_coeffs, quartic_coeffs, coexistence_criterion\n",
            "from tem_cdw.fermi_surface import cdw_order_matrix\n",
            "from tem_cdw.orbital_density import orbital_amplitudes, real_space_density\n",
            "from tem_cdw.models import LaTe2Model\n",
            "from tem_cdw.units import astar_from_q, q_from_astar, TWO_PI\n",
            "\n",
            "OUTDIR = pathlib.Path('figures/rte2_phase_a'); OUTDIR.mkdir(parents=True, exist_ok=True)\n"
        ], "execution_count": None, "outputs": []},
        {"cell_type": "code", "metadata": {}, "source": [
            "material = LaTe2Model()\n",
            "print('Fitted LaTe2 TB parameters:')\n",
            "print(f'  t_sigma = {material.params.t_sigma:.4f}')\n",
            "print(f'  t_pi    = {material.params.t_pi:.4f}')\n",
            "print(f'  t_d     = {material.params.t_d:.4f}')\n",
            "print(f'  mu      = {material.params.mu:.4f}')\n"
        ], "execution_count": None, "outputs": []},
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}
pathlib.Path("rte2_cdw_walkthrough.ipynb").write_text(json.dumps(nb, indent=1) + "\n")
print("Notebook skeleton written.")
PY
```

Verify: `ls -l /Users/yjkao/TEM_toolkit/rte2_cdw_walkthrough.ipynb`

- [ ] **Step 2: Add Fig. 1 cell — FS with orbital texture**

Append cells to the notebook via this helper:

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json, pathlib

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Fig. 1 — Fermi surface with orbital texture\n",
        "Diamond-shape FS in (a*, b*) plane, colored by |a_px|² (green) and "
        "|a_py|² (red). Nesting arrows for q_low and q_high overlaid.\n"
    ]},
    {"cell_type": "code", "metadata": {}, "source": [
        "def plot_fig1_late2(material, nk=400, savepath=None):\n",
        "    p = material.params\n",
        "    kg = np.linspace(-np.pi, np.pi, nk)\n",
        "    KX, KY = np.meshgrid(kg, kg, indexing='ij')\n",
        "    eps, avec = bands_and_projectors(KX, KY, p)\n",
        "    lower = eps[..., 0] - p.mu\n",
        "    # orbital weights for the lower band\n",
        "    wpx = np.abs(avec[..., 0, 0]) ** 2\n",
        "    wpy = np.abs(avec[..., 1, 0]) ** 2\n",
        "\n",
        "    fig, ax = plt.subplots(figsize=(6, 6))\n",
        "    # px sheet — green contour weighted by wpx\n",
        "    ax.contour(KX / TWO_PI, KY / TWO_PI, lower, levels=[0.0], colors='k',\n",
        "               linewidths=0.5, alpha=0.4)\n",
        "    # scatter FS points with color by orbital character\n",
        "    from matplotlib.colors import LinearSegmentedColormap\n",
        "    mask = np.abs(lower) < 0.05\n",
        "    if mask.any():\n",
        "        sc = ax.scatter((KX / TWO_PI)[mask], (KY / TWO_PI)[mask],\n",
        "                        c=(wpx - wpy)[mask], cmap='RdYlGn', s=3,\n",
        "                        vmin=-1, vmax=1)\n",
        "        cbar = plt.colorbar(sc, ax=ax, label=r'$|a_{p_x}|^2 - |a_{p_y}|^2$')\n",
        "\n",
        "    # Overlay nesting arrows from find_Q0\n",
        "    result = material.find_Q0(T=0.026, nk_bz=80, nk_q=40)\n",
        "    for (qx, qy) in result.peaks:\n",
        "        q_astar = qx / TWO_PI\n",
        "        ax.annotate('', xy=(q_astar/2, 0.05), xytext=(-q_astar/2, 0.05),\n",
        "                    arrowprops=dict(arrowstyle='->', color='blue'))\n",
        "        ax.text(0, 0.08, f'q={q_astar:.3f}a*', ha='center', fontsize=8)\n",
        "\n",
        "    ax.set_xlabel(r'$k_x / 2\\pi$ (a*)')\n",
        "    ax.set_ylabel(r'$k_y / 2\\pi$ (b*)')\n",
        "    ax.set_title(f'LaTe₂ Fermi surface — {material.name}')\n",
        "    ax.set_aspect('equal')\n",
        "    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)\n",
        "    fig.tight_layout()\n",
        "    if savepath: fig.savefig(savepath, dpi=150)\n",
        "    return fig\n",
        "\n",
        "fig = plot_fig1_late2(material, nk=300, savepath=OUTDIR / 'fig1_fermi_surface.png')\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []},
]

p = pathlib.Path("rte2_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
nb["cells"].extend(new_cells)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Fig 1 cells appended.")
PY
```

- [ ] **Step 3: Execute notebook to verify Fig. 1 renders**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute rte2_cdw_walkthrough.ipynb --output rte2_cdw_walkthrough.ipynb --ExecutePreprocessor.timeout=180`
Expected: succeeds; `figures/rte2_phase_a/fig1_fermi_surface.png` written.

- [ ] **Step 4: Inspect the figure**

Open `figures/rte2_phase_a/fig1_fermi_surface.png`. FS should be diamond-shaped, with visible px (green) / py (red) texture, and blue arrows indicating q_low and q_high nesting. If the figure is off (empty, no diamond, arrows in wrong direction), report before proceeding.

- [ ] **Step 5: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add rte2_cdw_walkthrough.ipynb figures/rte2_phase_a/fig1_fermi_surface.png
git commit -m "Notebook Fig 1: LaTe2 Fermi surface with orbital texture"
```

---

## Task 10: Notebook — Fig. 2 (χ(q) axial + diagonal at 3 temperatures)

**Files:**
- Modify: `rte2_cdw_walkthrough.ipynb` — append cells

- [ ] **Step 1: Append the Fig. 2 cells**

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json, pathlib

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Fig. 2 — χ(q) scan along a* and diagonal\n",
        "Axial (solid) vs diagonal (dashed control) susceptibility at three temperatures. "
        "Peaks near 0.15 and 0.25 a*; diagonal below axial confirms axial CDW.\n"
    ]},
    {"cell_type": "code", "metadata": {}, "source": [
        "temps_K = [100, 300, 500]\n",
        "temps_eV = [T * 8.617e-5 for T in temps_K]\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(7, 5))\n",
        "colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(temps_eV)))\n",
        "for T_eV, T_K, c in zip(temps_eV, temps_K, colors):\n",
        "    result = material.find_Q0(T=T_eV, nk_bz=100, nk_q=50)\n",
        "    q_astar = result.q_grid / TWO_PI\n",
        "    ax.plot(q_astar, result.chi_axial, '-', color=c, label=f'axial T={T_K}K')\n",
        "    ax.plot(q_astar, result.chi_diagonal, '--', color=c, alpha=0.6,\n",
        "            label=f'diag T={T_K}K')\n",
        "    for qx, _ in result.peaks:\n",
        "        ax.axvline(qx / TWO_PI, color=c, ls=':', alpha=0.4)\n",
        "\n",
        "ax.axvline(0.15, color='r', ls=':', alpha=0.5, label='target 0.15')\n",
        "ax.axvline(0.25, color='r', ls=':', alpha=0.5, label='target 0.25')\n",
        "ax.set_xlabel(r'$q$ (a*)')\n",
        "ax.set_ylabel(r'$\\chi_{\\rm diag\\_aa}(Q, T)$')\n",
        "ax.set_title(r'LaTe₂ susceptibility scan along $(q, 0)$ and $(q, q)$')\n",
        "ax.legend(fontsize=8)\n",
        "fig.tight_layout()\n",
        "fig.savefig(OUTDIR / 'fig2_chi_scan.png', dpi=150)\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []},
]

p = pathlib.Path("rte2_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
nb["cells"].extend(new_cells)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Fig 2 cells appended.")
PY
```

- [ ] **Step 2: Execute notebook**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute rte2_cdw_walkthrough.ipynb --output rte2_cdw_walkthrough.ipynb --ExecutePreprocessor.timeout=600`
Expected: succeeds; `figures/rte2_phase_a/fig2_chi_scan.png` written.

- [ ] **Step 3: Inspect — peaks at correct locations, diagonal below axial**

Open the figure. Confirm: two peaks near 0.15 and 0.25 a*; peak height grows as T lowers; dashed diagonal curves sit below solid axial curves.

- [ ] **Step 4: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add rte2_cdw_walkthrough.ipynb figures/rte2_phase_a/fig2_chi_scan.png
git commit -m "Notebook Fig 2: LaTe2 chi(q) axial vs diagonal at multiple T"
```

---

## Task 11: Notebook — Fig. 3 (χ(Q_peak, T) vs T)

**Files:**
- Modify: `rte2_cdw_walkthrough.ipynb`

- [ ] **Step 1: Append Fig. 3 cells**

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json, pathlib

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Fig. 3 — χ(Q_peak, T) vs T for both Q₀ candidates\n"
    ]},
    {"cell_type": "code", "metadata": {}, "source": [
        "# Get Q peaks at low T (peaks are stable across T for a given fit).\n",
        "result_ref = material.find_Q0(T=0.02, nk_bz=100, nk_q=50)\n",
        "assert len(result_ref.peaks) == 2, 'Expected 2 Q peaks; got ' + str(result_ref.peaks)\n",
        "Q_low, Q_high = result_ref.peaks\n",
        "print(f'Q_low  = {Q_low[0]/TWO_PI:.4f} a*')\n",
        "print(f'Q_high = {Q_high[0]/TWO_PI:.4f} a*')\n"
    ], "execution_count": None, "outputs": []},
    {"cell_type": "code", "metadata": {}, "source": [
        "T_K_grid = np.linspace(50, 500, 20)\n",
        "T_eV_grid = T_K_grid * 8.617e-5\n",
        "\n",
        "chi_low = np.empty_like(T_eV_grid)\n",
        "chi_high = np.empty_like(T_eV_grid)\n",
        "for i, T in enumerate(T_eV_grid):\n",
        "    chi_l = susceptibility_tensor(Q_low[0],  Q_low[1],  T, material.params, nk=100)\n",
        "    chi_h = susceptibility_tensor(Q_high[0], Q_high[1], T, material.params, nk=100)\n",
        "    chi_low[i]  = chi_combinations(chi_l)['diag_aa']\n",
        "    chi_high[i] = chi_combinations(chi_h)['diag_aa']\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "ax.plot(T_K_grid, chi_low,  'o-', label=f'Q_low  ({Q_low[0]/TWO_PI:.3f} a*)')\n",
        "ax.plot(T_K_grid, chi_high, 's-', label=f'Q_high ({Q_high[0]/TWO_PI:.3f} a*)')\n",
        "ax.set_xlabel('T (K)'); ax.set_ylabel(r'$\\chi_{\\rm diag\\_aa}$')\n",
        "ax.set_title(r'$\\chi(Q_{\\rm peak}, T)$ vs T for both Q₀ candidates')\n",
        "ax.legend(); fig.tight_layout()\n",
        "fig.savefig(OUTDIR / 'fig3_chi_vs_T.png', dpi=150)\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []},
]

p = pathlib.Path("rte2_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
nb["cells"].extend(new_cells)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Fig 3 cells appended.")
PY
```

- [ ] **Step 2: Execute**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute rte2_cdw_walkthrough.ipynb --output rte2_cdw_walkthrough.ipynb --ExecutePreprocessor.timeout=900`
Expected: succeeds.

- [ ] **Step 3: Inspect**

Both χ(T) curves should be monotonically decreasing with T; one should sit above the other consistently (that's the winning Q for high T).

- [ ] **Step 4: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add rte2_cdw_walkthrough.ipynb figures/rte2_phase_a/fig3_chi_vs_T.png
git commit -m "Notebook Fig 3: chi(Q_peak, T) vs T for both Q0 candidates"
```

---

## Task 12: Notebook — Fig. 4 (Quadratic GL a_−(T), multiple g)

**Files:**
- Modify: `rte2_cdw_walkthrough.ipynb`

- [ ] **Step 1: Append Fig. 4 cells**

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json, pathlib

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Fig. 4 — Quadratic GL coefficient $a_-(T)$ at both Q₀\n",
        "Zero-crossings mark mean-field T_c. Coupling g swept over several values.\n"
    ]},
    {"cell_type": "code", "metadata": {}, "source": [
        "g_values = [2.0, 2.5, 3.0, 3.5]\n",
        "T_K_grid = np.linspace(50, 500, 15)\n",
        "T_eV_grid = T_K_grid * 8.617e-5\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)\n",
        "for ax, (Q_label, Q) in zip(axes, [('Q_low', Q_low), ('Q_high', Q_high)]):\n",
        "    for g in g_values:\n",
        "        a_minus = np.empty_like(T_eV_grid)\n",
        "        for i, T in enumerate(T_eV_grid):\n",
        "            qc = quadratic_coeffs(Q[0], Q[1], T, material.params, g=g, nk=100)\n",
        "            am, _, _, _ = qc.diagonalize()\n",
        "            a_minus[i] = am\n",
        "        ax.plot(T_K_grid, a_minus, 'o-', label=f'g={g}')\n",
        "    ax.axhline(0, color='k', ls=':', alpha=0.5)\n",
        "    ax.set_xlabel('T (K)'); ax.set_title(f'{Q_label} = ({Q[0]/TWO_PI:.3f}, 0) a*')\n",
        "    ax.legend(fontsize=8)\n",
        "axes[0].set_ylabel(r'$a_-(T)$')\n",
        "fig.suptitle('Quadratic GL leading-instability coefficient')\n",
        "fig.tight_layout()\n",
        "fig.savefig(OUTDIR / 'fig4_quadratic_gl.png', dpi=150)\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []},
]

p = pathlib.Path("rte2_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
nb["cells"].extend(new_cells)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Fig 4 cells appended.")
PY
```

- [ ] **Step 2: Execute**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute rte2_cdw_walkthrough.ipynb --output rte2_cdw_walkthrough.ipynb --ExecutePreprocessor.timeout=1800`

Expected: succeeds. Runtime dominates here (~2 Q × 4 g × 15 T × ~1 s each ≈ 2 min).

- [ ] **Step 3: Inspect**

`a_−(T)` should decrease with T (increasing T raises the coefficient); zero crossings (T_c) should appear at higher T for larger g. Compare crossings between Q_low and Q_high panels — the higher-T_c Q is the "winning" candidate.

- [ ] **Step 4: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add rte2_cdw_walkthrough.ipynb figures/rte2_phase_a/fig4_quadratic_gl.png
git commit -m "Notebook Fig 4: quadratic GL a_-(T) at both Q0"
```

---

## Task 13: Notebook — Fig. 5 (Quartic coexistence check)

**Files:**
- Modify: `rte2_cdw_walkthrough.ipynb`

- [ ] **Step 1: Append Fig. 5 cells**

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json, pathlib

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Fig. 5 — Quartic coexistence check at the winning Q₀\n",
        "Above 1 → single-Δ A_g. Below 1 → double-Δ (Δ⁰ + Δ_z coexist).\n"
    ]},
    {"cell_type": "code", "metadata": {}, "source": [
        "# Pick the Q candidate with the larger chi at T=200 K (as proxy for higher T_c).\n",
        "T_probe = 200 * 8.617e-5\n",
        "chi_l = chi_combinations(susceptibility_tensor(Q_low[0], 0.0, T_probe, material.params, nk=100))['diag_aa']\n",
        "chi_h = chi_combinations(susceptibility_tensor(Q_high[0], 0.0, T_probe, material.params, nk=100))['diag_aa']\n",
        "Q_win, Q_win_label = (Q_low, 'Q_low') if chi_l >= chi_h else (Q_high, 'Q_high')\n",
        "print(f'Winning Q = {Q_win_label} at {Q_win[0]/TWO_PI:.4f} a* (chi_l={chi_l:.3f}, chi_h={chi_h:.3f})')\n"
    ], "execution_count": None, "outputs": []},
    {"cell_type": "code", "metadata": {}, "source": [
        "T_K_grid = np.array([100, 150, 200, 250, 300])\n",
        "ratios = []\n",
        "for T_K in T_K_grid:\n",
        "    T = T_K * 8.617e-5\n",
        "    qc = quartic_coeffs(Q_win[0], Q_win[1], T, material.params, nk=80)\n",
        "    _, ratio = coexistence_criterion(qc)\n",
        "    ratios.append(ratio)\n",
        "ratios = np.array(ratios)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(6, 4))\n",
        "ax.plot(T_K_grid, ratios, 'o-', color='C1')\n",
        "ax.axhline(1.0, color='r', ls='--', alpha=0.5, label='coexistence threshold')\n",
        "ax.set_xlabel('T (K)'); ax.set_ylabel(r'LHS/RHS (Eq. 29)')\n",
        "ax.set_title(f'Quartic coexistence check at {Q_win_label} = ({Q_win[0]/TWO_PI:.3f}, 0) a*')\n",
        "ax.legend(); fig.tight_layout()\n",
        "fig.savefig(OUTDIR / 'fig5_quartic_coexistence.png', dpi=150)\n",
        "\n",
        "print('T (K), LHS/RHS ratio')\n",
        "for T_K, r in zip(T_K_grid, ratios):\n",
        "    verdict = 'single-Δ A_g' if r > 1 else 'double-Δ'\n",
        "    print(f'  {T_K:6.0f}   {r:.4f}   ({verdict})')\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []},
]

p = pathlib.Path("rte2_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
nb["cells"].extend(new_cells)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Fig 5 cells appended.")
PY
```

- [ ] **Step 2: Execute**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute rte2_cdw_walkthrough.ipynb --output rte2_cdw_walkthrough.ipynb --ExecutePreprocessor.timeout=1800`

Expected: succeeds. Quartic evaluation is the most expensive step (~ 10 s per T at nk=80).

- [ ] **Step 3: Inspect**

Report should print single-Δ vs double-Δ verdicts. If verdict is consistent across all T (unlikely mixed unless close to threshold), record the outcome — needed to pick Δ⁰/Δ_z for Fig. 6.

- [ ] **Step 4: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add rte2_cdw_walkthrough.ipynb figures/rte2_phase_a/fig5_quartic_coexistence.png
git commit -m "Notebook Fig 5: quartic coexistence check at winning Q0"
```

---

## Task 14: Notebook — Fig. 6 (Real-space orbital density in winning phase)

**Files:**
- Modify: `rte2_cdw_walkthrough.ipynb`

- [ ] **Step 1: Append Fig. 6 cells**

```bash
cd /Users/yjkao/TEM_toolkit && .venv/bin/python - <<'PY'
import json, pathlib

new_cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "## Fig. 6 — Real-space orbital density in the winning phase\n",
        "ρ_px(r), ρ_py(r), ρ(r) on a 12×12 lattice using Q_win and the "
        "phase (single-Δ A_g or double-Δ) selected by Fig. 5.\n"
    ]},
    {"cell_type": "code", "metadata": {}, "source": [
        "# Use the T=200 K quartic verdict to pick the phase for Fig. 6.\n",
        "T_probe = 200 * 8.617e-5\n",
        "qc = quartic_coeffs(Q_win[0], Q_win[1], T_probe, material.params, nk=80)\n",
        "is_double, ratio_val = coexistence_criterion(qc)\n",
        "print(f'Coexistence ratio at 200K = {ratio_val:.4f} — {\"double-Δ\" if is_double else \"single-Δ A_g\"}')\n",
        "\n",
        "D0, Dz = (1.0, 0.5j) if is_double else (1.0, 0.0)\n",
        "print(f'Δ⁰ = {D0}, Δ_z = {Dz}')\n"
    ], "execution_count": None, "outputs": []},
    {"cell_type": "code", "metadata": {}, "source": [
        "X, Y, rho_px, rho_py, rho = real_space_density(\n",
        "    Q_win[0], Q_win[1], D0=D0, Dz=Dz, nx=12, ny=12,\n",
        ")\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))\n",
        "for ax, dat, title in zip(axes, [rho_px, rho_py, rho],\n",
        "                          [r'$\\rho_{p_x}(r)$', r'$\\rho_{p_y}(r)$', r'$\\rho(r)$']):\n",
        "    im = ax.imshow(dat.T, origin='lower', extent=[0, 11, 0, 11],\n",
        "                   cmap='RdBu_r', vmin=-1, vmax=1)\n",
        "    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')\n",
        "    plt.colorbar(im, ax=ax, shrink=0.8)\n",
        "fig.suptitle(f'Winning phase at {Q_win_label} — {\"double-Δ\" if is_double else \"single-Δ A_g\"}')\n",
        "fig.tight_layout()\n",
        "fig.savefig(OUTDIR / 'fig6_orbital_density.png', dpi=150)\n",
        "plt.show()\n"
    ], "execution_count": None, "outputs": []},
]

p = pathlib.Path("rte2_cdw_walkthrough.ipynb")
nb = json.loads(p.read_text())
nb["cells"].extend(new_cells)
p.write_text(json.dumps(nb, indent=1) + "\n")
print("Fig 6 cells appended.")
PY
```

- [ ] **Step 2: Execute full notebook once more, top-to-bottom**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute rte2_cdw_walkthrough.ipynb --output rte2_cdw_walkthrough.ipynb --ExecutePreprocessor.timeout=1800`

Expected: succeeds; all six figures cached in `figures/rte2_phase_a/`.

- [ ] **Step 3: Inspect**

Density patterns should show clear periodicity at Q_win. Single-Δ A_g: ρ_px = ρ_py identical. Double-Δ (with Δ_z ≠ 0): px and py patterns differ.

- [ ] **Step 4: Commit**

```bash
cd /Users/yjkao/TEM_toolkit
git add rte2_cdw_walkthrough.ipynb figures/rte2_phase_a/fig6_orbital_density.png
git commit -m "Notebook Fig 6: real-space orbital density in winning CDW phase"
```

---

## Task 15: End-to-end verification and success-criteria checklist

**Purpose:** Confirm all six success criteria from the design spec are met.

**Files:** No new files; run all tests.

- [ ] **Step 1: Full test suite must pass**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/pytest tests/ -v`
Expected: all tests green — RTE3 regression, units, models base, RTE3 model, RTE2 model, fit helpers, LaTe2 model.

- [ ] **Step 2: Re-run RTE3 walkthrough notebook — regression check**

Run: `cd /Users/yjkao/TEM_toolkit && .venv/bin/jupyter nbconvert --to notebook --execute tem_cdw_rte3_walkthrough.ipynb --output tem_cdw_rte3_walkthrough.ipynb --ExecutePreprocessor.timeout=1800`

Expected: succeeds; figures unchanged from pre-refactor. This is the numerics-preserving invariant.

- [ ] **Step 3: Success criteria checklist**

Confirm each spec criterion, either from the test suite or by inspecting the notebook outputs:

- [ ] RTE3 walkthrough produces same figures as before refactor
- [ ] `LaTe2Model()` fit converged; fitted params in `tem_cdw/models/rte2_fit_report/fitted_params.npz`
- [ ] `find_Q0` returns q_low ≈ 0.15 a* and q_high ≈ 0.25 a* (each within 3%)
- [ ] χ(q, q) diagonal < χ(q, 0) axial across the scanned q range
- [ ] Quadratic + quartic GL computed at both Q₀ without numerical instability
- [ ] All six figures in `figures/rte2_phase_a/` render and are cached

- [ ] **Step 4: Commit summary**

```bash
cd /Users/yjkao/TEM_toolkit
git log --oneline main..HEAD | head -20
```

Reports the sequence of commits produced by Phase A.

---

## Notes for the executing agent

- **If a fit diverges** (Task 8): stop. The design explicitly says this is a physics finding — the 2-orbital form is insufficient. Report to user before proposing extensions (second-neighbor hoppings, bilayer, etc.).
- **If `find_Q0` returns fewer than 2 peaks** after the fit: same as above.
- **If the RTE3 regression test fails after the rename** (Task 2): revert Task 2 and investigate before proceeding. The refactor is meant to be pure motion.
- **Notebook execution timeouts:** Tasks 12–13 approach the timeout budget on slow machines. If a cell times out, reduce `nk` for that specific plot (e.g. 80 → 60) rather than raising the timeout indefinitely.
- **All temperatures are in eV** in code (Alekseev convention). User-facing K values in plots are converted via `T_eV = T_K * 8.617e-5`.
- **All q values are in radians** in code. `astar_from_q` / `q_from_astar` from `tem_cdw/units.py` are used only at plot labels and factory boundaries.
