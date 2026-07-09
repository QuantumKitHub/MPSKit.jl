<!-- PLAN DEVIATION (for maintainer): RESTRUCTURE_PLAN.md asked for a U(1) redo of the
flagship, but MPSKitModels' `transverse_field_ising` only accepts Trivial, Z2Irrep, or
FermionParity and throws an ArgumentError for U1Irrep. The TFIM's actual symmetry is Z2,
so this tutorial uses Z2Irrep throughout and points to the U(1)/SU(2) material
(howto/states.md §9 and the Heisenberg examples) for the larger-group payoff. -->

# [Using symmetries](@id tutorial_using_symmetries)

In [Your first ground state](@ref tutorial_first_groundstate) and [The thermodynamic limit](@ref tutorial_thermodynamic_limit) we treated the transverse-field Ising model (TFIM) as a generic spin chain.
But the TFIM is not generic: it has a symmetry, and in this tutorial we teach MPSKit about it.

Recall the Hamiltonian,

```math
H = -J\left(\sum_{\langle i,j\rangle} \sigma^z_i\,\sigma^z_j + g\sum_i \sigma^x_i\right),
```

and consider the *global spin flip* ``P = \prod_i \sigma^x_i``, which flips every spin at once.
Conjugating by ``P`` sends ``\sigma^z_i \to -\sigma^z_i``, so the interaction term ``\sigma^z_i\sigma^z_j`` picks up two minus signs and is unchanged, while the field term ``\sigma^x_i`` commutes with ``P`` trivially.
Hence ``H`` commutes with ``P``.
Since ``P^2 = 1``, this is a ``\mathbb{Z}_2`` symmetry, and every eigenstate of ``H`` can be labelled by a parity quantum number: *even* (``P = +1``) or *odd* (``P = -1``).

MPSKit, through the TensorKit tensor backend, can bake this symmetry directly into the tensors of the MPS.
Doing so buys you two things.
First, the tensors become **block-sparse**: at the same total bond dimension the computer multiplies smaller dense blocks, which is faster.
Second, every state you compute carries an explicit **sector label**, so "the lowest odd-parity excitation" becomes something you can ask for directly.
This tutorial demonstrates both, by redoing the finite-chain TFIM calculation once without and once with the symmetry.

## Loading the packages

Every code block on this page shares one Julia session, so we load the packages once.
`Z2Irrep` and `Z2Space`, the symmetry-aware building blocks used below, come from TensorKit.

```@example using-symmetries
using MPSKit, MPSKitModels, TensorKit
```

## 1. Recap: the ground state without symmetry

We start from the workflow of the first tutorial: a chain of `L = 16` sites, a random `FiniteMPS`, and a DMRG ground-state search.
Two small changes from before: we set the field to `g = 2.0`, and we use a total bond dimension of 16.

Why `g = 2.0`?
This puts us deep in the paramagnetic phase, where the ground state respects the spin-flip symmetry.
That matters for what comes next: an MPS built from symmetric tensors lives in exactly one parity sector and *cannot* spontaneously break the symmetry, so a fair comparison needs a point where the true ground state is symmetric to begin with.

```@example using-symmetries
L = 16
H = transverse_field_ising(FiniteChain(L); g = 2.0)
ψ₀ = FiniteMPS(L, ℂ^2, ℂ^16)
ψ, envs, ϵ = find_groundstate(ψ₀, H, DMRG(; verbosity = 0))
E = expectation_value(ψ, H)
```

This is our reference number: the ground-state energy computed with plain, symmetry-oblivious tensors.

## 2. The same model, with the symmetry made explicit

To exploit the symmetry we change two lines: the Hamiltonian and the initial state.

For the Hamiltonian, we pass the symmetry as an extra first argument.
`transverse_field_ising(Z2Irrep, ...)` builds the *same* Hamiltonian as before, but out of tensors that manifestly commute with the spin flip:

```@example using-symmetries
H_Z2 = transverse_field_ising(Z2Irrep, FiniteChain(L); g = 2.0)
```

For the state, the plain spaces `ℂ^2` and `ℂ^16` are replaced by *graded* spaces that keep track of parity:

```@example using-symmetries
ψ₀_Z2 = FiniteMPS(L, Z2Space(0 => 1, 1 => 1), Z2Space(0 => 8, 1 => 8))
```

The syntax reads as a list of `sector => dimension` pairs, where sector `0` is the even (``P = +1``) irrep of ``\mathbb{Z}_2`` and sector `1` is the odd (``P = -1``) one:

- The physical space `Z2Space(0 => 1, 1 => 1)` is the familiar two-dimensional spin-1/2 site, now split into its symmetry content: one even state and one odd state.
- The virtual space `Z2Space(0 => 8, 1 => 8)` says the bond carries 8 states of even parity and 8 of odd parity — 16 in total, matching the `ℂ^16` of the plain run, so the two calculations have exactly the same variational power.

From here the workflow is unchanged:

```@example using-symmetries
ψ_Z2, envs_Z2, ϵ_Z2 = find_groundstate(ψ₀_Z2, H_Z2, DMRG(; verbosity = 0))
E_Z2 = expectation_value(ψ_Z2, H_Z2)
```

Both runs found the same ground state, and the two energies agree to numerical precision:

```@example using-symmetries
E, E_Z2
```

!!! note "Same physics, different bookkeeping"
    Nothing about the model changed — only the way its tensors are stored.
    The symmetric calculation restricts the search to states of definite (here: even) parity, and stores only the tensor blocks the symmetry allows to be nonzero.

## 3. The payoff, part 1: block-sparse tensors

Where did the symmetry go?
Into the *structure* of the state.
Ask for the virtual space at the central bond and you no longer get an anonymous `ℂ^16`, but a space that knows its sector decomposition:

```@example using-symmetries
V = left_virtualspace(ψ_Z2, L ÷ 2)
```

Its total dimension is still 16:

```@example using-symmetries
dim(V)
```

The same sector labels show up in every quantity derived from the state.
The entanglement spectrum at the central cut, for instance, now comes back resolved by sector — compare [Entanglement entropy and spectrum](@ref howto_entanglement), where the same call on an unsymmetric state produced a single `Trivial()` block:

```@example using-symmetries
spectrum = entanglement_spectrum(ψ_Z2, L ÷ 2)
collect(keys(spectrum))
```

Iterating `pairs` gives each sector together with its singular values:

```@example using-symmetries
collect(pairs(spectrum))
```

This block structure is where the speedup comes from: instead of multiplying one dense 16-dimensional bond index, the computer multiplies two independent blocks of roughly half that size, and the forbidden matrix elements between the sectors are never stored or touched at all.
At bond dimension 16 the difference is negligible, but the saving grows with the bond dimension and with the size of the symmetry group.

## 4. The payoff, part 2: sectors label the physics

The sector labels are not just an implementation detail — they classify the eigenstates of ``H``, and MPSKit lets you target a sector directly.

In the paramagnetic phase the lowest excitation of the TFIM is, roughly speaking, a single flipped spin.
Flipping one spin changes the parity of the state, so this excitation lives in the *odd* sector — a different sector than the (even) ground state.

The [`excitations`](@ref) function computes excited states on top of a converged ground state; on a finite chain it takes the Hamiltonian, an algorithm, the ground state, and its environments, and returns energies measured *above* the ground state.
By default it searches the trivial (even) sector:

```@example using-symmetries
Es_even, ϕs_even = excitations(H_Z2, QuasiparticleAnsatz(), ψ_Z2, envs_Z2; num = 1)
Es_even[1]
```

The `sector` keyword redirects the search to the odd sector:

```@example using-symmetries
Es_odd, ϕs_odd = excitations(
    H_Z2, QuasiparticleAnsatz(), ψ_Z2, envs_Z2;
    num = 1, sector = Z2Irrep(1)
)
Es_odd[1]
```

The odd-sector excitation is indeed the lower one:

```@example using-symmetries
Es_odd[1] < Es_even[1]
```

Without the symmetry built into the tensors, this question could not even be posed: the plain calculation of Section 1 has no notion of parity to select on.
More ways to use `excitations` — dispersion relations, other algorithms, infinite chains — are collected in [Excited states](@ref howto_excitations).

## Where to go next

You have run the flagship TFIM calculation with its ``\mathbb{Z}_2`` symmetry made explicit: the same physics at the same total bond dimension, but with block-sparse tensors and sector labels on everything the calculation produces.

The same syntax scales up to larger symmetry groups, where the payoff grows.
For a U(1) symmetry (particle number, magnetization) the graded spaces list integer or half-integer charges instead of parities — worked constructions are in [Constructing states](@ref howto_states), Section 9.
For non-abelian symmetries such as SU(2) the gains are more dramatic still, because each symmetric block then represents an entire multiplet of states; the spin-1 Haldane chain and XXZ Heisenberg pages in the examples gallery show this in action.

To continue the tutorial track, [Quasiparticle excitations](@ref tutorial_excitations) develops the excitation calculation of Section 4 into a full dispersion relation; the recipe collection for excited states is [Excited states](@ref howto_excitations), and the one for building symmetric states is [Constructing states](@ref howto_states).

<!-- Maintainer notes (not rendered):
- tutorials/thermodynamic_limit.md still carries "TODO(link): using_symmetries tutorial"
  (and one for the excitations tutorial); those can now point at
  (@ref tutorial_using_symmetries) / (@ref tutorial_excitations).
- docs/src/examples/index.md has no @id anchor, so the examples-gallery mention above is
  plain text; adding an anchor there would let this page (and others) link it properly.
- Plan deviation from RESTRUCTURE_PLAN.md recorded in the comment at the top of this
  file: Z2Irrep instead of U(1), because transverse_field_ising rejects U1Irrep.
-->
