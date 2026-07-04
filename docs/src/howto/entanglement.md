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

!!! note
    `ψ∞` here is a random `InfiniteMPS`, not a converged ground state, so the values above illustrate the interface rather than any physical entanglement profile.
    For a physically meaningful result, compute the entropy of a state obtained from [`find_groundstate`](@ref) (for example via VUMPS).

!!! note
    `WindowMPS` also supports `entropy(ψ, site)` with a required site argument, mirroring the `FiniteMPS` form.

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
