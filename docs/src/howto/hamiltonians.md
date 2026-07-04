# [Building Hamiltonians](@id howto_hamiltonians)

The examples on this page use MPSKit.jl, TensorKit.jl, and TensorKitTensors.jl.
See [Installation](@ref tutorial_installation) for how to add these packages to your environment.

This page collects recipes for constructing MPO Hamiltonians from local operators, for both finite and infinite (translation-invariant) lattices.
It also covers converting an infinite Hamiltonian to finite open or periodic boundary conditions, and carving a finite window out of an infinite Hamiltonian.
For building the matching state objects see [Constructing states](@ref howto_states); for evaluating a Hamiltonian's energy on a state see [Computing observables](@ref howto_observables).
The reference page for the underlying MPO structure is [Operators](@ref lib_operators).

```@example hamiltonians
using MPSKit, TensorKit
using TensorKitTensors.SpinOperators: σˣ, σᶻ
```

---

## Setup: local operators

The examples below build the transverse-field Ising model (TFIM), the same flagship model used elsewhere in these docs.
It couples neighbouring spins through `X ⊗ X` and applies a transverse field of strength `g` along `Z`.
The model has a quantum phase transition at `g = 1`, separating an ordered (ferromagnetic) phase at small `g` from a disordered (paramagnetic) phase at large `g`.
The single-site Pauli operators come from [TensorKitTensors.jl](https://github.com/QuantumKitHub/TensorKitTensors.jl), which returns `ComplexF64` `TensorMap`s on the spin-1/2 physical space `ℂ^2`:

```@example hamiltonians
X = σˣ()
Z = σᶻ()
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
