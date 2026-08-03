```@meta
DocTestSetup = quote
    using MPSKit, MPSKitModels, TensorKit
end
```

# [Symmetries](@id concept_symmetries)

[TensorKit for MPS users](@ref concept_vector_spaces) already showed the key fact: a symmetry in MPSKit is not a flag passed to an algorithm, it is a property of the vector *spaces* that a tensor is built from, and every algorithm is written once, generically, for any such space.
That page built the mental model of a `TensorMap` and introduced graded spaces through a single ℤ₂ example.
This page stays at the same level of abstraction but widens the lens: what kinds of symmetry a graded space can encode, what changes qualitatively as you move from an abelian group to a non-abelian one or to fermionic or anyonic statistics, and — the question every user eventually asks — when the extra bookkeeping of a bigger symmetry group is actually worth it.
For the hands-on version of the ℤ₂ case worked all the way through a ground-state search, see [Using symmetries](@ref tutorial_using_symmetries); this page explains the reasoning behind that recipe and extends it to the other symmetry classes MPSKit supports.

As on the sibling page, none of the symmetry machinery lives in `MPSKit` or `MPSKitModels` itself: `using MPSKit` does not bring a single sector or space type into scope, and neither does `using MPSKitModels`.
Every symmetric object — `Z2Irrep`, `U1Irrep`, `SU2Irrep`, `Z2Space`, `U1Space`, and so on — comes from `TensorKit`, so every example on this page loads all three packages explicitly.

```@example symmetries
using MPSKit, MPSKitModels, TensorKit
```

## Sectors, charges, and block-sparsity

A **sector** is a label for an irreducible representation of the symmetry: for a group symmetry it is one irrep, and a graded space is built by declaring how many copies ("degeneracy" or "multiplicity") of each sector it contains.
[TensorKit for MPS users](@ref concept_vector_spaces) did this for ℤ₂ with `Z2Space(0 => 1, 1 => 1)`; the same `sector => degeneracy` syntax works for every symmetry, so a U(1)-graded space that keeps track of, say, a conserved particle number or magnetization from `-1` to `1` reads:

```@example symmetries
V = U1Space(-1 => 1, 0 => 1, 1 => 1)
dim(V)
```

The space still knows its full sector content, queryable with [`sectors`](https://quantumkithub.github.io/TensorKit.jl/stable/) and [`dim`](https://quantumkithub.github.io/TensorKit.jl/stable/) applied to a specific sector:

```@example symmetries
collect(sectors(V))
```

```@example symmetries
[dim(V, c) for c in sectors(V)]
```

Charge conservation is the statement that a symmetric tensor may only have nonzero entries between sectors whose charges add up correctly (for a Hamiltonian term, incoming and outgoing charge must match).
Concretely this means the tensor is **block-diagonal** in the sector label: what would be one dense array for a plain `ℂ^n` space becomes a handful of smaller, independent dense blocks, one per allowed sector combination, and the entries that connect different sectors are not merely zero — they are never allocated or touched at all.
This is the mechanism behind everything that follows: the *type* of symmetry only changes what the sector labels are and how they combine (their *fusion rules*); the block-sparse storage and the charge-conservation bookkeeping are handled identically underneath.

## A taxonomy of symmetry types

MPSKitModels' [`heisenberg_XXX`](https://quantumkithub.github.io/MPSKitModels.jl/stable/) model is a convenient single thread through the taxonomy, because the same Heisenberg Hamiltonian can be built with a trivial symmetry or with any of the three main non-trivial types below, purely by passing a different sector type as the first argument:

```@example symmetries
H_triv = heisenberg_XXX(FiniteChain(4); spin = 1 // 2)
H_Z2 = heisenberg_XXX(Z2Irrep, FiniteChain(4); spin = 1 // 2)
H_U1 = heisenberg_XXX(U1Irrep, FiniteChain(4); spin = 1 // 2)
H_SU2 = heisenberg_XXX(SU2Irrep, FiniteChain(4); spin = 1 // 2)
```

Four `Hamiltonian`s, four different tensor structures, one physical model.

### Abelian symmetries: ℤ_N and U(1)

`Z2Irrep`, `Z3Irrep`, `Z4Irrep`, and the general `ZNIrrep`, together with `U1Irrep`, are the abelian family: their sectors are literally the elements of ℤ_N or of the integers (or half-integers), and two sectors fuse by addition modulo N, or ordinary addition for U(1).
Every irrep is one-dimensional, so an abelian symmetry buys exactly the block-sparsity described above and nothing more: `H_Z2` above encodes the same spin-flip parity used throughout [Using symmetries](@ref tutorial_using_symmetries), while `H_U1` encodes conservation of total magnetization ``S^z_{\mathrm{tot}}``, with sectors running over the possible values of ``S^z_{\mathrm{tot}}``.
`transverse_field_ising` is a useful reminder that not every model has every symmetry available: it accepts `Trivial`, `Z2Irrep`, or `FermionParity`, but raises an error for `U1Irrep`, because the transverse-field Ising model genuinely only has the ℤ₂ spin-flip symmetry — there is no conserved U(1) charge to exploit.

### Non-abelian symmetries: SU(2)

`SU2Irrep` sectors are labelled by a total spin ``j = 0, \tfrac12, 1, \tfrac32, \dots``, and fusing two of them follows the angular-momentum addition (Clebsch–Gordan) rule rather than simple addition: fusing spin ``j_1`` and ``j_2`` can produce any ``j`` from ``|j_1-j_2|`` to ``j_1+j_2``.
The qualitative difference from the abelian case is that each sector ``j`` is not one-dimensional but ``(2j+1)``-dimensional, and a symmetric tensor need only store the multiplicity of each ``j`` — the internal ``(2j+1)`` structure of every multiplet is fixed by representation theory and is never stored explicitly.
`H_SU2` above is built exactly this way: it is the same Heisenberg chain, only now every eigenstate additionally carries a total-spin label, and the tensors only ever store one number per multiplet rather than one number per individual magnetic sublevel.
The next section makes this saving concrete.

### Fermionic symmetries and product sectors

`FermionParity` grades a space into an even and an odd fermion-number sector, and — crucially — TensorKit's fermionic tensor category attaches the anticommutation sign directly to the braiding of `FermionParity`-graded legs, so that once physical and virtual legs carry this sector, index permutations automatically pick up the correct fermionic signs instead of requiring the sign rule to be implemented by hand in every algorithm [mortier2025](@cite).
Models with more than one physical species combine sectors with `⊠` (typed `\boxtimes`) into a `ProductSector`, for instance an odd fermion paired with unit U(1) charge:

```@example symmetries
FermionParity(1) ⊠ U1Irrep(1)
```

`hubbard_model` exercises this directly: it takes an independent *particle* symmetry and *spin* symmetry, and assembles the physical space internally out of `FermionParity ⊠ (particle symmetry)` and `FermionParity ⊠ (spin symmetry)` pieces.
Choosing U(1) for particle number and SU(2) for spin gives the maximally symmetric Hubbard chain:

```@example symmetries
H_hub = hubbard_model(ComplexF64, U1Irrep, SU2Irrep, FiniteChain(4); t = 1.0, U = 8.0)
```

The first argument is the scalar element type, required here because a lattice is given explicitly; the two symmetry types then set the particle-number and spin symmetries in that order.

By contrast, `bose_hubbard_model` only accepts `Trivial` or `U1Irrep`: bosons carry no parity grading, so there is no fermionic sign to encode and no spin degree of freedom to make non-abelian.
At the far end of this spectrum, `quantum_chemistry_hamiltonian` does not expose a symmetry choice at all — it always builds its tensors with the fixed, maximal ``U(1) \boxtimes SU(2) \boxtimes \mathrm{FermionParity}`` symmetry (particle number, total spin, and fermionic sign), because for realistic molecular Hamiltonians that full symmetry is essentially always worth imposing.

### Anyonic symmetries

The generality goes further than groups.
Sectors such as `FibonacciAnyon` or `IsingAnyon` are not group representations at all — their fusion rules come from a modular tensor category — yet because every MPSKit algorithm is written against the abstract `Sector` interface, they are handled by exactly the same code paths, with no special-casing.
The [hard-hexagon model](@ref "The Hard Hexagon model") example puts this to work: its transfer matrix is built from `FibonacciAnyon`-graded tensors (`Vect[FibonacciAnyon](:I => …, :τ => …)`), and the standard statistical-mechanics workflow computes its partition function just as it would for an ordinary symmetry.

## When does SU(2) pay off

Two distinct effects are at play whenever a symmetry is switched on, and it is worth separating them because only one of them scales with the size of the symmetry group.

The first effect is the block-sparsity already described: at a fixed total bond dimension, the computer multiplies several smaller dense blocks instead of one large one, and the (forbidden) cross-sector entries are never stored.
[Using symmetries](@ref tutorial_using_symmetries) demonstrates this concretely for ℤ₂: the same 16-dimensional bond becomes two roughly-8-dimensional blocks.
This first effect is present for *any* symmetry, abelian or not, and its benefit grows with the number of distinct sectors the bond dimension gets spread over.

The second effect is specific to non-abelian symmetries and is qualitatively larger: because a whole ``(2j+1)``-dimensional multiplet is represented by a single stored block, the *number of stored parameters* needed to reach a given *total*, physical bond dimension shrinks.
This can be checked directly: build a graded SU(2) space and compare its total dimension against the multiplicities it actually stores per sector.

```@example symmetries
V_SU2 = SU2Space(0 => 2, 1 // 2 => 4)
dim(V_SU2)
```

```@example symmetries
[dim(V_SU2, c) for c in sectors(V_SU2)]
```

The total dimension `dim(V_SU2)` is `10`, because each spin-``j`` sector contributes its ``(2j+1)``-fold multiplet: ``2 \times (2\cdot 0 + 1) + 4 \times (2\cdot\tfrac12 + 1) = 2 + 8 = 10``.
But `dim(V_SU2, c)` returns the *stored* multiplicity of each sector — here `[2, 4]`, just six numbers in total — because the ``(2j+1)`` internal structure of every multiplet is fixed by representation theory and never stored.
For an abelian symmetry the two coincide (every irrep is one-dimensional, so the multiplicities and the total dimension agree, as with the U(1) space above); it is precisely for a non-abelian group that the stored count falls below the physical dimension.

The gap between the physical dimension (`10`) and the six numbers actually stored is the source of SU(2)'s reputation for letting DMRG reach much larger effective bond dimensions at the same computational cost — the same principle used to push non-abelian symmetric uniform MPS to large SU(3) bond dimensions in practice [devos2022](@cite).

None of this is free.
Every symmetric block carries the overhead of tracking fusion trees and recombining Clebsch–Gordan coefficients whenever legs are permuted or contracted, and for a non-abelian group this bookkeeping is genuinely more expensive per block than for an abelian one.
In practice this means SU(2) (or any non-abelian symmetry) is worth reaching for when the physics genuinely has that symmetry — a spin chain with full rotational invariance, for instance — and when the bond dimension is large enough that the multiplet-reduction saving dominates the per-block overhead; for small bond dimensions, or for a symmetry the Hamiltonian does not actually have, the abelian or even trivial case is often simpler and just as fast.

## Fixing the total charge

Sector labels are not only a storage optimization: they are physical quantum numbers, and MPSKit lets a calculation target a specific one directly.

For an MPS, the total charge is fixed by giving the state a non-trivial `left` or `right` virtual space, rather than the default unit (trivial-charge) one:

```@example symmetries
ψ_odd = FiniteMPS(
    4, Z2Space(0 => 1, 1 => 1), Z2Space(0 => 2, 1 => 2);
    left = Z2Space(1 => 1)
)
left_virtualspace(ψ_odd, 1)
```

Every tensor in `ψ_odd` is now forced, by charge conservation, to represent a state of odd total parity — there is no way for a symmetric MPS built this way to drift into the even sector.
The same idea appears for excited states and for transfer-matrix spectra: the `sector` keyword of [`excitations`](@ref) and of `transfer_spectrum` restricts the search to a chosen total charge instead of the default trivial one, exactly as used to isolate the odd-parity excitation of the TFIM in [Using symmetries](@ref tutorial_using_symmetries).
[Excited states](@ref howto_excitations) collects further recipes for working with `sector`, and [Constructing states](@ref howto_states) collects the analogous recipes for building states with a prescribed symmetry and charge.

## Where to go next

- For the tensor mechanics underneath all of this — spaces, `TensorMap`s, index conventions — see [TensorKit for MPS users](@ref concept_vector_spaces).
- For the fully worked ℤ₂ example, from Hamiltonian to ground state to a sector-targeted excitation, see [Using symmetries](@ref tutorial_using_symmetries).
- For how symmetric tensors assemble into states and operators, see [Matrix product states](@ref concept_matrix_product_states) and [Operators and Hamiltonians](@ref concept_operators_and_hamiltonians).
- For task recipes that use a `sector` or a charged virtual space, see [Constructing states](@ref howto_states), [Excited states](@ref howto_excitations), and [Entanglement entropy and spectrum](@ref howto_entanglement).
