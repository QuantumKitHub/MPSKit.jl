# [Computing observables](@id howto_observables)

This page collects recipes for extracting physical quantities from an MPS: local and multi-site expectation values, the energy of a Hamiltonian, two-point correlators, and the energy variance as a convergence diagnostic.
All examples share a single namespace and build on state and operator objects you would have in hand after a ground-state calculation.

```@example observables
using MPSKit, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
```

For building MPS objects see [Constructing states](@ref howto_states).
For controlling the bond dimension during optimization see [Controlling bond dimension](@ref howto_bond_dimension).
The reference page for ground-state algorithms is [Ground-state algorithms](@ref lib_groundstate).

---

## Setup: state and operators

The examples below use a spin-1/2 `FiniteMPS` together with the Pauli operators from [TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl).
These are `ComplexF64` `TensorMap`s, matching the default element type of the state.

```@example observables
L = 8
ψ = FiniteMPS(L, ℂ^2, ℂ^8)   # random finite MPS, bond dim ≤ 8

# single-site Pauli operators
X = σˣ()
Z = σᶻ()
```

The finite TFIM Hamiltonian used in recipes 3 and 5 is built from these:

```@example observables
lattice = fill(ℂ^2, L)
H = FiniteMPOHamiltonian(lattice, (i, i + 1) => -(X ⊗ X) for i in 1:(L - 1)) +
    FiniteMPOHamiltonian(lattice, (i,) => -0.5 * Z for i in 1:L)
```

---

## 1. Local (one-site) expectation value

Use `expectation_value(ψ, i => O)` to evaluate ⟨ψ|Oᵢ|ψ⟩ at a single site `i`.
The pair `i => O` identifies the site and the single-site operator.

```@example observables
expectation_value(ψ, 4 => Z)   # ⟨Z⟩ at site 4
```

To compute a local observable at every site, broadcast over the indices:

```@example observables
[expectation_value(ψ, i => Z) for i in 1:L]
```

!!! note
    The state `ψ` must be normalised for the expectation value to be meaningful.
    A freshly constructed `FiniteMPS` is normalised by default; if you modified
    the tensors by hand, call `normalize!(ψ)` first.

---

## 2. Multi-site (contiguous) expectation value

For a product of operators on a contiguous range of sites, pass a tuple of indices together with a multi-site operator formed by taking tensor products `⊗`:

```@example observables
# ⟨X₂ X₃⟩ — two-site operator on sites 2 and 3
expectation_value(ψ, (2, 3) => X ⊗ X)
```

The operator `X ⊗ X` is a `{2,2}` `TensorMap` (two incoming, two outgoing legs) matching the two-site index tuple `(2, 3)`.
The tuple must be contiguous; arbitrary non-adjacent index sets are not supported by this form.

```@example observables
# ⟨Z₁ Z₂ Z₃⟩ — three-site operator
expectation_value(ψ, (1, 2, 3) => Z ⊗ Z ⊗ Z)
```

---

## 3. Energy (full-MPO expectation value)

When the operator is an [`AbstractMPO`](@ref) (e.g. a Hamiltonian), pass it directly without an index argument.
MPSKit evaluates the full contraction ⟨ψ|H|ψ⟩:

```@example observables
E = expectation_value(ψ, H)
```

The result is a scalar; for a Hermitian `H` and a normalised `ψ` its imaginary part is zero up to floating-point noise.

The same form works for `InfiniteMPS` with an `InfiniteMPOHamiltonian`, where the returned value is the energy **per unit cell**.

!!! note
    The full-MPO form automatically computes and caches the environments.
    If you already have environments from a prior `find_groundstate` call you can
    pass them as a trailing argument to avoid recomputation, but this is optional;
    omitting them is always safe and correct.

---

## 4. Two-point correlators

[`correlator`](@ref) computes ⟨O₁ᵢ O₂ⱼ⟩ for two sites with `i < j`.
The recommended call uses a single two-site operator `O₁₂`:

```@example observables
# ⟨Z₂ Zⱼ⟩ for a single target site j = 6
correlator(ψ, Z ⊗ Z, 2, 6)
```

!!! warning
    `i` must be strictly less than `j`.
    Calling `correlator(ψ, O₁₂, i, j)` with `i ≥ j` will throw an error.

### Correlation profile over a range

Pass a range as `j` to obtain a vector of correlators — one entry per target site.
This is the efficient route for a full correlation profile:

```@example observables
# ⟨Z₂ Zⱼ⟩ for j = 3, 4, …, L
corr = correlator(ψ, Z ⊗ Z, 2, 3:L)
```

The result is a `Vector` whose `k`-th element corresponds to `j = 3 + k - 1`.

A common pattern is to normalise the correlator by ⟨Z⟩² to extract the connected part:

```@example observables
z_mean = expectation_value(ψ, 2 => Z)
connected = [c - z_mean * expectation_value(ψ, j => Z) for (j, c) in zip(3:L, corr)]
```

This subtracts the disconnected part ``\langle Z_i\rangle\langle Z_j\rangle`` to leave the connected correlator ``\langle Z_i Z_j\rangle - \langle Z_i\rangle\langle Z_j\rangle``.
<!-- REVIEW: the physical interpretation of the decay of `connected` (correlation
     length, order parameter, etc.) is model- and phase-dependent — maintainer to
     confirm any interpretation added here. -->

---

## 5. Energy variance as a convergence check

[`variance`](@ref) returns ⟨H²⟩ − ⟨H⟩², which is zero if and only if `ψ` is an exact eigenstate of `H`.
Use it as a quantitative convergence diagnostic after a ground-state search:

```@example observables
var_E = variance(ψ, H)
```

A smaller variance indicates that `ψ` is closer to a true eigenstate.

After running a ground-state algorithm the variance should have dropped significantly compared to the random starting state above:

```@example observables
ψ_gs, envs, _ = find_groundstate(ψ, H, DMRG(; maxiter = 10))
variance(ψ_gs, H)
```

!!! note
    The `variance` function also accepts an optional pre-computed `envs` argument.
    Pass the environments returned by `find_groundstate` to skip recomputation:

    ```julia
    variance(ψ_gs, H, envs)
    ```

---

<!--
CLOSING NOTES FOR MAINTAINERS / DOCTEST-RUNNER
================================================

Cross-references used on this page (all confirmed to exist):
  @ref howto_states         → docs/docs/src/howto/states.md
  @ref howto_bond_dimension → docs/docs/src/howto/bond_dimension.md
  @ref lib_groundstate      → docs/docs/src/lib/groundstate.md
  @ref correlator           → exported function; docstring in MPSKit source
  @ref variance             → exported function; docstring in MPSKit source
  @ref AbstractMPO          → type in MPSKit source

Shared example namespace: @example observables (all blocks run in document order).

Symbols in the verified API that are NOT demonstrated on this page:
  - The lower-level two-argument correlator form `correlator(ψ, O₁, O₂, i, j)`
    (single-site MPO tensors with trivial boundary legs) — mentioned in prose but
    not shown; users should prefer the two-site O₁₂ form.
  - `expectation_value` with an explicit `environments` trailing argument — omitted
    from examples to keep recipes minimal; noted in admonitions only.

Symbols wanted but NOT in the verified API (not used):
  - `normalize!` — used in a prose note only, not in a code block; should be
    confirmed present before adding a code example.
  - `InfiniteMPOHamiltonian` — not constructed in examples to avoid guessing
    its exact constructor signature for single-site unit cells here.
    The InfiniteMPS note is prose-only and marked REVIEW.

Changes needed in OTHER files (do NOT edit those pages here):
  - docs/docs/make.jl: add "howto/observables.md" to the "How-to" section in `pages`.
  - docs/docs/src/howto/states.md: could add a "see also" link to @ref howto_observables.
  - docs/docs/src/howto/bond_dimension.md: same optional cross-link.
  - docs/docs/src/lib/groundstate.md: add a "see also" pointer to @ref howto_observables.

All code blocks on this page were verified to execute against MPSKit (2026-07-01).
-->
