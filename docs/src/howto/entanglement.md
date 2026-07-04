# [Entanglement entropy and spectrum](@id howto_entanglement)

This page collects recipes for extracting the entanglement entropy and the entanglement spectrum from the gauge (bond) tensors of an MPS.
For general expectation values and correlators see [Computing observables](@ref howto_observables); for building the state objects used below see [Constructing states](@ref howto_states).
The reference page for these and related functions is [Observables and analysis](@ref lib_observables).

```@example entanglement
using MPSKit, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
```

---

## Setup: a TFIM ground state

The examples below reuse a spin-1/2 `FiniteMPS` and the transverse-field Ising Hamiltonian, optimized with DMRG so the entanglement structure reflects an actual ground state rather than a random tensor:

```@example entanglement
L = 8
ψ0 = FiniteMPS(L, ℂ^2, ℂ^8)

# single-site Pauli operators
X = σˣ()
Z = σᶻ()

lattice = fill(ℂ^2, L)
H = FiniteMPOHamiltonian(lattice, (i, i + 1) => -(X ⊗ X) for i in 1:(L - 1)) +
    FiniteMPOHamiltonian(lattice, (i,) => -0.5 * Z for i in 1:L)

ψ, envs, _ = find_groundstate(ψ0, H, DMRG(; maxiter = 10))
```

---

## 1. Entanglement entropy at a single cut

[`entropy`](@ref) returns the von Neumann entanglement entropy across the cut to the right of a given site.
For a `FiniteMPS` the site is a required argument:

```@example entanglement
entropy(ψ, L ÷ 2)   # entropy across the central cut
```

<!-- REVIEW: no claim is made here about the numeric value in relation to an area
     law, criticality, or the TFIM phase diagram at this field strength — maintainer
     to add any such interpretation. -->

---

## 2. Entropy profile across every cut

Collecting `entropy(ψ, i)` over the valid range of sites gives the full entropy profile of the chain:

```@example entanglement
[entropy(ψ, i) for i in 1:L]
```

!!! warning
    For `FiniteMPS` the cut site is required and must lie in `1:length(ψ)`.
    `site = 0` — a valid default for `InfiniteMPS` and `WindowMPS` (see recipe 5) — throws a `BoundsError` for `FiniteMPS`.

---

## 3. The entanglement spectrum

[`entanglement_spectrum`](@ref) returns the singular values of the gauge tensor to the right of a site, packaged as a sector-resolved vector:

```@example entanglement
spectrum = entanglement_spectrum(ψ, L ÷ 2)
```

The entropy can equivalently be computed directly from this spectrum with [`entropy`](@ref):

```@example entanglement
entropy(spectrum)
```

```@example entanglement
entropy(ψ, L ÷ 2) ≈ entropy(spectrum)
```

Both routes agree, since `entropy(ψ, site)` computes the entropy from exactly this spectrum internally.

---

## 4. Sector-resolved spectrum

Because the returned spectrum is indexed by symmetry sector, you can inspect the singular values sector by sector.
Use `keys` to list the sectors present at a cut, and index the spectrum with a sector to obtain its singular values:

```@example entanglement
collect(keys(spectrum))
```

```@example entanglement
spectrum[only(keys(spectrum))]
```

For the plain (no explicit symmetry) `FiniteMPS` built above there is a single sector, `Trivial()`, so all singular values live in one block.
`pairs(spectrum)` iterates `sector => values` pairs and is the natural entry point for a symmetric state where multiple sectors are populated at a cut:

```@example entanglement
collect(pairs(spectrum))
```

<!-- REVIEW: for symmetric states (e.g. built with a `Rep[G]` physical space, see
     howto_states recipe 9) the sector-resolved spectrum reflects the symmetry
     content of the entangled degrees of freedom across the cut; the maintainer
     should confirm any stronger physical statement (e.g. relating sector
     multiplicities to symmetry-protected degeneracies) before it is added here. -->

---

## 5. Entanglement of an infinite MPS

For `InfiniteMPS`, the cut site defaults to `0`, and `entropy` without a site argument returns one entropy per site in the unit cell:

```@example entanglement
ψ∞ = InfiniteMPS(ℂ^2, ℂ^8)
entropy(ψ∞)
```

```@example entanglement
entanglement_spectrum(ψ∞)   # site defaults to 0
```

<!-- REVIEW: ψ∞ here is a random InfiniteMPS, not a converged fixed point (e.g. via
     VUMPS); the maintainer may want a converged example if this page should also
     illustrate the entropy of an actual infinite ground state. -->

!!! note
    `WindowMPS` also supports `entropy(ψ, site)` with a required site argument, mirroring the `FiniteMPS` form.
    <!-- REVIEW: WindowMPS entanglement path is not covered by tests -->

---

## Plotting the spectrum

MPSKit defines an `entanglementplot` recipe via `RecipesBase`, but does not depend on Plots.jl itself.
To use it, add `using Plots` (or another Plots-backed package) in your own environment:

```julia
using Plots
entanglementplot(ψ; site = L ÷ 2)
```

!!! note
    `entanglementplot` is a plotting *recipe*: it only becomes available once `Plots` (or a compatible plotting package) is loaded.
    This block is not executed on this page to keep the docs build free of the Plots.jl dependency.

---

<!--
CLOSING NOTES FOR MAINTAINERS / DOCTEST-RUNNER
================================================

Cross-references used on this page:
  @ref howto_observables → docs/docs/src/howto/observables.md
  @ref howto_states      → docs/docs/src/howto/states.md
  @ref lib_observables    → docs/docs/src/lib/observables.md (assumed id per task
                            instructions; confirm this anchor exists once that
                            reference page is written/renamed)
  @ref entropy             → exported function; docstring in MPSKit source
  @ref entanglement_spectrum → exported function; docstring in MPSKit source

Shared example namespace: @example entanglement (all blocks run in document order).

Symbols demonstrated:
  - entropy(ψ, site::Int)            — FiniteMPS, required site
  - entropy(ψ)                       — InfiniteMPS, no site, returns Vector over unit cell
  - entropy(ψ, site)                 — InfiniteMPS, explicit site
  - entropy(spectrum::SectorVector)  — entropy from a precomputed spectrum
  - entanglement_spectrum(ψ, site)   — FiniteMPS, required site
  - entanglement_spectrum(ψ)         — InfiniteMPS, defaults to site = 0
  - keys(spectrum), pairs(spectrum), spectrum[sector] — sector-resolved access

Symbols in the verified API NOT demonstrated (mentioned in prose only):
  - entropy(ψ, site) / entanglement_spectrum(ψ, site) for WindowMPS — no runnable
    example per task instructions (untested path); flagged with REVIEW instead.
  - entanglementplot(state; site = 0, ...) — shown as a non-executed ```julia```
    block only, since it requires `using Plots` (heavy optional dependency not
    part of the docs environment by default).

Caveats / REVIEW flags placed in the page body:
  - FiniteMPS site range is 1:length(ψ); site = 0 throws BoundsError (recipe 2 warning).
  - No physics interpretation asserted for the TFIM entropy value, the sector
    structure at a cut, or the random (non-converged) InfiniteMPS example —
    all flagged REVIEW for maintainer sign-off.
  - WindowMPS entanglement path flagged REVIEW as untested in the repo.

Changes needed in OTHER files (do NOT edit those pages here):
  - docs/docs/make.jl: add "howto/entanglement.md" to the "How-to" section in `pages`.
  - Confirm the `lib_observables` @id exists on whatever reference page ends up
    hosting entropy/entanglement_spectrum docstrings; update the link here if the
    id differs.

Code blocks on this page are written to be run by doctest-runner; none of the
numeric outputs above have been hand-verified against a real build.
-->
