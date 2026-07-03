# [Building Hamiltonians](@id howto_hamiltonians)

This page collects recipes for constructing MPO Hamiltonians from local operators, for both finite and infinite (translation-invariant) lattices.
It also covers converting an infinite Hamiltonian to finite open or periodic boundary conditions, and carving a finite window out of an infinite Hamiltonian.
For building the matching state objects see [Constructing states](@ref howto_states); for evaluating a Hamiltonian's energy on a state see [Computing observables](@ref howto_observables).
The reference page for the underlying MPO structure is [Operators](@ref lib_operators).

```@example hamiltonians
using MPSKit, TensorKit
```

---

## Setup: local operators

The examples below build the transverse-field Ising model (TFIM), the same flagship model used elsewhere in these docs.
<!-- REVIEW: brief physics description of TFIM (nearest-neighbour ZZ/XX coupling plus transverse field, critical point at g=1) — maintainer to confirm phrasing and any claims about criticality. -->
All operators are `ComplexF64` `TensorMap`s on the spin-1/2 physical space `ℂ^2`:

```@example hamiltonians
X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2)
Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2)
g = 0.5
```

---

## 1. Finite Hamiltonian from local terms

[`FiniteMPOHamiltonian`](@ref) takes an array of `VectorSpace` objects describing the local Hilbert spaces, followed by any number of `inds => operator` pairs.
A single-site term uses a one-element tuple `(i,) => O`; a nearest-neighbour term uses a two-element tuple `(i, i + 1) => O₁₂`, where `O₁₂` is a two-site operator built with `⊗`:

```@example hamiltonians
L = 8
lattice = fill(ℂ^2, L)

H_finite = FiniteMPOHamiltonian(lattice, (i, i + 1) => -(X ⊗ X) for i in 1:(L - 1)) +
    FiniteMPOHamiltonian(lattice, (i,) => -g * Z for i in 1:L)
```

Adding the two `FiniteMPOHamiltonian` objects combines the bond terms and the field terms into a single Jordan-block MPO.
Equivalently, all terms can be passed as one call by splatting a single collection of `inds => operator` pairs; see [Operators](@ref lib_operators) for that form.

!!! note
    The index tuples must refer to contiguous sites for the two-site pairs shown here.
    See [Operators](@ref lib_operators) for the general, non-nearest-neighbour "expert mode" construction, which is not covered on this task-oriented page.

---

## 2. Infinite (translation-invariant) Hamiltonian

[`InfiniteMPOHamiltonian`](@ref) uses the same `inds => operator` convention, but the lattice argument is a single unit cell, and site indices wrap around it periodically.
For the 1-site TFIM unit cell, `(1, 2) => O₁₂` couples site 1 to site 2 of the *next* unit cell:

```@example hamiltonians
unitcell = fill(ℂ^2, 1)
H_inf = InfiniteMPOHamiltonian(unitcell, (1, 2) => -(X ⊗ X), (1,) => -g * Z)
```

The resulting operator repeats this single bond-plus-field pattern along the whole infinite chain.
Use it directly with an [`InfiniteMPS`](@ref) in `expectation_value` or `find_groundstate`, exactly as described in [Computing observables](@ref howto_observables).

!!! tip
    Hand-assembling local operators works for any model, but for standard lattice models MPSKitModels.jl provides ready-made Hamiltonian builders and the `@mpoham` macro for a more compact syntax.
    See the MPSKitModels.jl documentation for that higher-level interface; it is a separate package from MPSKit and not covered here.

---

## 3. Converting between boundary conditions

Starting from an `InfiniteMPOHamiltonian`, [`open_boundary_conditions`](@ref) truncates it to a finite chain of length `L` with open ends, and [`periodic_boundary_conditions`](@ref) instead closes it into a finite ring.
In both cases `L` must be a multiple of the unit-cell length:

```@example hamiltonians
L_finite = 6   # multiple of the 1-site unit cell

H_open = open_boundary_conditions(H_inf, L_finite)
```

```@example hamiltonians
H_periodic = periodic_boundary_conditions(H_inf, L_finite)
```

`H_open` is the same finite-chain Hamiltonian you would get from writing out the terms by hand, as in recipe 1 above, restricted to `L_finite` sites.
`H_periodic` additionally couples the last site back to the first, forming a ring.

!!! note
    Both functions return a [`FiniteMPOHamiltonian`](@ref).
    There is no boundary-condition keyword on the `FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian` constructors themselves; boundary conditions are chosen by picking which constructor (or conversion function) to call.

---

## 4. A window Hamiltonian

[`WindowMPOHamiltonian`](@ref) carves a finite interval out of an infinite Hamiltonian while keeping the infinite left and right environments intact.
This is the operator counterpart of a [`WindowMPS`](@ref) (see [Constructing states](@ref howto_states)), and the two are used together to study a finite region embedded in, and coupled to, an infinite bulk:

```@example hamiltonians
H_window = WindowMPOHamiltonian(H_inf, 1:6)
```

The interval `1:6` selects which unit cells of `H_inf` become the mutable finite window; everything outside it is treated as the fixed infinite environment.

---

<!--
CLOSING NOTES FOR MAINTAINERS / DOCTEST-RUNNER
================================================

Cross-references used on this page (all confirmed to exist):
  @ref howto_states     → docs/docs/src/howto/states.md
  @ref howto_observables → docs/docs/src/howto/observables.md
  @ref lib_operators    → docs/docs/src/lib/operators.md
  @ref FiniteMPOHamiltonian, InfiniteMPOHamiltonian, open_boundary_conditions,
       periodic_boundary_conditions, WindowMPOHamiltonian, InfiniteMPS, WindowMPS
    → types/functions in MPSKit source

Shared example namespace: @example hamiltonians (all blocks run in document order).

Symbols in the verified API that are NOT demonstrated on this page:
  - The matrix / "expert mode" `FiniteMPOHamiltonian(Ws::Vector{<:AbstractMatrix})`
    (or BlockTensorMap) constructor for hand-written Jordan-block MPOs — mentioned
    only in prose, cross-linked to lib_operators, not shown here to avoid guessing
    the exact call shape.
  - `JordanMPOTensor`, `DenseMPO` — not used; out of scope for a task-oriented page
    about building Hamiltonians from local terms.
  - Longer-range or non-contiguous index tuples (e.g. `(i, i + 2) => O`) — not
    demonstrated; only contiguous nearest-neighbour pairs were used since only
    that form was confirmed against the observables.md idiom.
  - Splatting a single collection of `inds => operator` pairs into one
    `FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian` call (vs. the two-call `+`
    pattern used here) — mentioned in prose, not shown as a second code example.

Symbols wanted but NOT in the verified API (not used):
  - `@mpoham` — this belongs to MPSKitModels.jl, not MPSKit. Mentioned only in a
    `!!! tip` prose note with no runnable example and no fabricated signature.
  - Bare `MPOHamiltonian(...)` constructor — avoided entirely per instructions;
    all examples use `FiniteMPOHamiltonian`/`InfiniteMPOHamiltonian` instead.
  - No `boundary=` keyword exists on any constructor; boundary conditions are
    handled exclusively via `open_boundary_conditions`/`periodic_boundary_conditions`.

Changes needed in OTHER files (do NOT edit those pages here):
  - docs/docs/make.jl: add "howto/hamiltonians.md" to the "How-to" section in `pages`.
  - docs/docs/src/howto/observables.md: could add a "see also" link to
    @ref howto_hamiltonians near its Hamiltonian setup block.
  - docs/docs/src/howto/states.md: could add a "see also" link to
    @ref howto_hamiltonians next to the WindowMPS section (recipe 7).
  - docs/docs/src/lib/operators.md: could add a "see also" pointer to
    @ref howto_hamiltonians for the task-oriented version of this content.

This page was drafted against a verified API report (FiniteMPOHamiltonian,
InfiniteMPOHamiltonian, open_boundary_conditions, periodic_boundary_conditions,
WindowMPOHamiltonian) and against the confirmed idiom already in use in
observables.md and man/operators.md. Code blocks await doctest-runner execution.
-->
