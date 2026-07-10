# [Statistical mechanics](@id howto_statmech)

```@meta
DocTestSetup = quote
    using MPSKit, MPSKitModels, TensorKit
end
```

This page collects recipes for the *boundary-MPS* (transfer-matrix) approach to two-dimensional classical statistical mechanics.
The partition function of a classical lattice model is written as a contraction of a two-dimensional tensor network, one row of which is an [`InfiniteMPO`](@ref) — the *transfer matrix*.
Contracting the network in the thermodynamic limit amounts to finding the leading eigenvector of that transfer matrix, which MPSKit approximates by an [`InfiniteMPS`](@ref) via [`leading_boundary`](@ref).

The examples on this page use MPSKit.jl, MPSKitModels.jl, and TensorKit.jl.
See [Installation](@ref tutorial_installation) for how to add these packages to your environment.

```@example statmech
using MPSKit, MPSKitModels, TensorKit
```

For the structure of MPOs see [Operators and Hamiltonians](@ref concept_operators_and_hamiltonians).
For a full worked case using anyonic symmetries, see the gallery example [The Hard Hexagon model](@ref "The Hard Hexagon model").

---

## 1. Build the transfer matrix

MPSKitModels ships ready-made transfer-matrix MPOs for several classical models.
The two-dimensional classical Ising model is provided by `classical_ising`; other options include `sixvertex` and `hard_hexagon`.

```@example statmech
mpo = classical_ising()
```

The returned object is an `InfiniteMPO` with a single-site unit cell: one tensor whose four legs are the two horizontal (virtual) and two vertical (physical) bonds of the Boltzmann-weight tensor.
Its physical space is read off with `physicalspace`:

```@example statmech
P = physicalspace(mpo, 1)
```

By default `classical_ising` uses the inverse temperature `beta = log(1 + sqrt(2)) / 2`.

<!-- REVIEW: this default `beta = log(1+√2)/2` is the exact 2D-Ising critical inverse temperature βc (Onsager / Kramers–Wannier self-dual point). I state it as textbook fact; maintainer to confirm the one-line characterization. -->

---

## 2. Find the leading boundary MPS

[`leading_boundary`](@ref) approximates the dominant eigenvector of the transfer matrix by an `InfiniteMPS`.
Supply an initial guess with a chosen bond dimension and an optimization algorithm — [`VUMPS`](@ref) is the usual choice.

The transfer matrix of a classical model is generally **not Hermitian**, so pass a non-Hermitian eigensolver to `VUMPS`:

```@example statmech
alg = VUMPS(;
    verbosity = 0,
    alg_eigsolve = MPSKit.Defaults.alg_eigsolve(; ishermitian = false),
)

ψ₀ = InfiniteMPS([P], [ℂ^16])          # initial guess, bond dimension 16
ψ, envs, ϵ = leading_boundary(ψ₀, mpo, alg)
ϵ                                       # final convergence error
```

`leading_boundary` returns a triple `(ψ, environments, ϵ)`: the converged boundary MPS, its environment manager, and the final convergence error.
Reuse the returned `envs` in subsequent calls to avoid recomputing environments.

!!! note
    The initial guess sets the bond dimension `D` of the boundary MPS.
    A larger `D` gives a better approximation of the leading eigenvector; near a critical point the accessible correlation length grows with `D` (see [Controlling bond dimension](@ref howto_bond_dimension) for growing `D` on the fly).

---

## 3. Free energy and partition function per site

The expectation value of the transfer matrix in the converged boundary MPS is the partition function per site ``\Lambda = \mathcal{Z}^{1/N}`` in the thermodynamic limit:

```@example statmech
Λ = expectation_value(ψ, mpo)
```

For a Hermitian-normalised model the imaginary part is zero up to floating-point noise.
The free energy per site follows from ``f = -\tfrac{1}{\beta}\log\Lambda``:

```@example statmech
β = log(1 + sqrt(2)) / 2
f = -1 / β * log(real(Λ))
```

<!-- REVIEW: `f` is computed here from this MPO's normalization; I have NOT cross-checked its value against Onsager's exact free energy per site, and sign/normalization conventions of `classical_ising` may differ. Maintainer to verify the physical value. -->

---

## 4. Correlation length and entanglement entropy

The boundary MPS encodes the correlations of the two-dimensional system.
[`correlation_length`](@ref) returns the (largest) correlation length of the transfer matrix, and [`entropy`](@ref) the entanglement entropy of the boundary MPS across a virtual bond:

```@example statmech
ξ = correlation_length(ψ)
```

```@example statmech
S = real(first(entropy(ψ)))
```

At a critical point the true correlation length diverges; the finite bond dimension of the boundary MPS cuts it off at a finite value that grows with `D`.
This finite-entanglement scaling of `S` against `log(ξ)` is exactly what the gallery example [The Hard Hexagon model](@ref "The Hard Hexagon model") exploits to extract a central charge.

<!-- REVIEW: statement that at criticality the exact ξ diverges and the finite-D boundary MPS cuts it off (growing with D) is standard finite-entanglement-scaling lore; maintainer to confirm phrasing. -->

---

## 5. Symmetric transfer matrices

When the classical model has a global symmetry, the transfer matrix can be built from symmetric tensors, which makes the boundary computation cheaper and more stable.
`classical_ising` accepts a symmetry type; the ``\mathbb{Z}_2`` spin-flip symmetry gives a `Z2Irrep`-graded MPO:

```@example statmech
mpo_z2 = classical_ising(Z2Irrep)
P_z2 = physicalspace(mpo_z2, 1)
```

The workflow is identical — only the virtual space of the initial guess is now a graded space:

```@example statmech
V_z2 = Z2Space(0 => 8, 1 => 8)
ψ_z2, = leading_boundary(InfiniteMPS([P_z2], [V_z2]), mpo_z2, alg)
real(expectation_value(ψ_z2, mpo_z2))
```

For anyonic (non-invertible) symmetries the same recipe applies with a `Vect[FibonacciAnyon]` virtual space — this is the case worked out in [The Hard Hexagon model](@ref "The Hard Hexagon model").
See [Symmetries](@ref concept_symmetries) for how to choose graded spaces.

---

## 6. Multi-row unit cells

If the transfer matrix has a unit cell spanning several rows, the boundary object becomes a [`MultilineMPS`](@ref) and the operator a [`MultilineMPO`](@ref).
`repeat` stacks copies of a single-row MPO into a multi-row transfer matrix:

```@example statmech
mmpo = repeat(mpo, 2, 1)     # 2 rows × 1 column
```

Build a `MultilineMPS` with one `InfiniteMPS` per row and call `leading_boundary` exactly as before; the returned environments are a `MultilineEnvironments`:

```@example statmech
mψ = MultilineMPS([InfiniteMPS([P], [ℂ^12]), InfiniteMPS([P], [ℂ^12])])
mψ, menvs, = leading_boundary(mψ, mmpo, alg)
real(expectation_value(mψ, mmpo))
```

The `expectation_value` of a multi-row transfer matrix is the product of the per-site weights over the rows of the unit cell; divide `log` of it by the number of rows to recover the per-site free energy.

<!-- REVIEW: I verified numerically that the 2-row expectation value ≈ (single-row value)^2, consistent with "product over rows"; maintainer to confirm the free-energy-per-site normalization statement (divide log by number of rows). -->

---

## See also

- [The Hard Hexagon model](@ref "The Hard Hexagon model") — a full worked study (central charge from finite-entanglement scaling) using an anyonic transfer matrix.
- [Operators and Hamiltonians](@ref concept_operators_and_hamiltonians) — MPO structure and construction.
- [Controlling bond dimension](@ref howto_bond_dimension) — growing the boundary-MPS bond dimension.
- [Computing observables](@ref howto_observables) — `expectation_value`, `correlation_length`, and related tools.

```@meta
DocTestSetup = nothing
```
