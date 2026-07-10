```@meta
DocTestSetup = quote
    using MPSKit, TensorKit
end
```

# [Finite versus infinite MPS](@id concept_finite_vs_infinite)

The [matrix product state](@ref concept_matrix_product_states) machinery — the site tensors, the virtual bonds, the canonical gauge — is shared by two rather different physical objects, and MPSKit gives each its own type.
A [`FiniteMPS`](@ref) is the wavefunction of a chain with a definite number of sites and two open ends: a genuine vector in a finite-dimensional Hilbert space.
An [`InfiniteMPS`](@ref) instead stores a small, repeating *unit cell* of tensors and imagines it tiled forever along the chain, so that it represents a translation-invariant state directly in the thermodynamic limit `L = ∞`.
This page explains what that difference *means* — why the two share almost all of their code yet answer subtly different questions, and in particular why an infinite state is always normalized to one while a finite state is not.
It is about understanding rather than construction: to *build* either kind of state see [Constructing states](@ref howto_states), and for the type signatures see the [States](@ref lib_states) reference.

## Two different objects

A [`FiniteMPS`](@ref) is what you reach for whenever the system genuinely has a fixed size and boundaries: a chain of `N` sites, each a separate mutable tensor, with trivial (dimension-one) bonds capping the two ends.
It is a literal, if compressed, representation of a state vector `|ψ⟩` living in the tensor-product Hilbert space of those `N` sites, and every question you could ask of an ordinary state vector — its norm, its overlap with another state, an expectation value at a particular site — has a finite, exactly computable answer.
The open ends are part of the physics: sites near a boundary are in a different environment from sites in the bulk, and any measured quantity still carries a dependence on the length `N`.

An [`InfiniteMPS`](@ref) throws both of those features away on purpose.
It represents a state that is exactly invariant under translation by one unit cell, so there is no boundary anywhere and no length `N` left to depend on.
What is actually stored is a finite list of tensors — the unit cell — together with the gauge data needed to treat the infinite periodic contraction; indexing the state is periodic, so `ψ.AL[i]` and `ψ.AL[i + length(ψ)]` return the same tensor.
This is the representation used throughout [The thermodynamic limit](@ref tutorial_thermodynamic_limit), where the payoff — no boundary effects, no finite-size extrapolation — is put to work on the transverse-field Ising model.

## The unit cell

The single number that characterizes the periodicity of an [`InfiniteMPS`](@ref) is its unit-cell length, returned by `length`.
The most common choice is a one-site unit cell, in which a single tensor is repeated across the whole chain:

```@example finite-infinite
using MPSKit, TensorKit # hide
ψ_infinite = InfiniteMPS(ℂ^2, ℂ^8)
length(ψ_infinite)
```

A larger unit cell is specified by passing a vector of physical and virtual spaces, one entry per site of the cell:

```@example finite-infinite
ψ_cell = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^8, ℂ^8])
length(ψ_cell)
```

The unit-cell length is not a free accuracy knob like the bond dimension; it is a physical statement about the *period* of the state you intend to represent.
A translation-invariant ansatz of period `L` can only capture states whose own spatial period divides `L`.
<!-- REVIEW: commensurability claim. A one-site InfiniteMPS is invariant under single-site translation, so it cannot represent a state that spontaneously breaks translation down to period 2 (e.g. an antiferromagnetic / dimerized / period-2 ordered ground state); such a state needs a unit cell whose length is a multiple of its period. Please confirm this framing (and the "divides L" phrasing) is stated the way you'd want physically. -->
Choosing a cell that is commensurate with the physical period of the model — the magnetic period of an ordered phase, or a period imposed by the Hamiltonian's own unit cell — is therefore a modelling decision, not a numerical one, and picking too small a cell forces the algorithm to approximate a state it structurally cannot represent.
A [`FiniteMPS`](@ref), by contrast, has no notion of a unit cell at all: its `length` is simply the number of physical sites, and each of those sites carries its own independent tensor.

## Why an infinite MPS is normalized to one

The sharpest practical consequence of the finite/infinite distinction shows up in the norm, and it is worth understanding rather than memorizing.

For a [`FiniteMPS`](@ref) the norm is exactly the Euclidean norm `√⟨ψ|ψ⟩` of the state vector it represents — a genuine, finite number.
The space-based constructors normalize by default, so a freshly built state has norm one, but nothing forces that: the norm is a real degree of freedom you can set at will, and rescaling the state rescales it in the obvious way.

```@example finite-infinite
ψ_finite = FiniteMPS(rand, ComplexF64, 16, ℂ^2, ℂ^8)
norm(ψ_finite)
```

```@example finite-infinite
norm(3 * ψ_finite)
```

For an [`InfiniteMPS`](@ref) that same quantity does not exist.
The overlap `⟨ψ|ψ⟩` of an infinite state is, formally, a product of one transfer-matrix factor per unit cell, so for a chain of `n` cells it grows (or decays) like `λⁿ`, where `λ` is the leading eigenvalue of the transfer matrix.
As `n → ∞` this is `0` if `λ < 1` and `∞` if `λ > 1`, and the *only* value that yields a finite, well-defined state is `λ = 1`.
<!-- REVIEW: the ⟨ψ|ψ⟩ = λⁿ scaling argument and the "only λ = 1 is finite" conclusion are the standard textbook justification for per-site normalization of an iMPS, but the source contains no docstring stating it (the api-explorer found only the per-tensor normalize!(C)/normalize!(AC) construction steps and the isapprox(dot, 1) equality convention). The transfer_spectrum ≈ 1 example below is real evidence, but please check the physical wording of the λⁿ argument. -->
MPSKit therefore fixes the gauge so that the transfer matrix has leading eigenvalue exactly one, which we can read straight off its spectrum:

```@example finite-infinite
first(transfer_spectrum(ψ_infinite)) ≈ 1
```

With that fixed, `norm` of an [`InfiniteMPS`](@ref) is defined *per site* rather than globally: it is the norm of a single center-gauged unit-cell tensor, and it is always one.

```@example finite-infinite
norm(ψ_infinite) ≈ 1
```

```@example finite-infinite
norm(ψ_infinite) ≈ norm(ψ_infinite.AC[1])
```

Because the normalization is intensive, it does not grow with the unit cell: a two-site cell is normalized to one just as a one-site cell is.

```@example finite-infinite
norm(ψ_cell) ≈ 1
```

This is why scalar multiplication of an [`InfiniteMPS`](@ref) is simply not defined — there is no overall amplitude to rescale — and why every physically meaningful quantity in the infinite setting is a *density*.
The energy returned for the state is an energy per site, an order parameter is measured at one representative site of the cell, and quantities with no finite-chain analogue, such as the [`correlation_length`](@ref), are extracted from the transfer-matrix spectrum of the uniform state rather than from any global overlap.

## The same algorithms, two settings

Because the two types share the canonical-form vocabulary, most of MPSKit's high-level entry points accept either one, and it is the *algorithm* passed to them that is specialized to the finite or the infinite case.
Ground-state search is the clearest example: [`find_groundstate`](@ref) dispatches on the state it is handed, running [`DMRG`](@ref) — which sweeps back and forth across a chain with two ends — for a [`FiniteMPS`](@ref), and [`VUMPS`](@ref) or [`IDMRG`](@ref)/[`IDMRG2`](@ref) — which converge a single uniform unit cell — for an [`InfiniteMPS`](@ref).
The distinction is not incidental: a boundary-sweeping method like DMRG has no meaning without ends to sweep between, while VUMPS' re-gauging step, which replaces every tensor in the chain at once, only makes sense for a genuinely translation-invariant state.
Some routines instead span both worlds: [`TDVP`](@ref) time-evolves finite and infinite states alike, its two-site bond-growing variant existing only for the finite case.
For which algorithm fits which task — and why one is preferred over another within each column — see [The algorithm landscape](@ref concept_algorithm_landscape).

Two further state types sit between the finite and infinite poles rather than at them, reusing the same machinery: a [`WindowMPS`](@ref) embeds a finite, mutable window inside two infinite environments, and a [`MultilineMPS`](@ref) stacks several infinite states to represent two-dimensional networks.
Both are introduced in [Constructing states](@ref howto_states).

## Where to go next

- For the flagship finite-then-infinite walkthrough of the same model, see [The thermodynamic limit](@ref tutorial_thermodynamic_limit).
- For the gauge and canonical-form machinery both types share, see [Matrix product states](@ref concept_matrix_product_states).
- For choosing the right algorithm in each setting, see [The algorithm landscape](@ref concept_algorithm_landscape).
- For how to construct each state type, see [Constructing states](@ref howto_states); for type signatures, the [States](@ref lib_states) reference.
```
