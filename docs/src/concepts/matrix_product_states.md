```@meta
DocTestSetup = quote
    using MPSKit, TensorKit
end
```

# [Matrix product states](@id concept_matrix_product_states)

A matrix product state (MPS) represents the wavefunction of a one-dimensional quantum system as a chain of tensors, one per site, contracted along shared *virtual* bonds.
The physical indices carry the local degrees of freedom, while the virtual bonds carry the entanglement between the two halves of the system that meet at that bond.

```@raw html
<img src="./finite_mps_definition.png" alt="A finite MPS drawn as a chain of tensors, each with one physical leg pointing out and virtual legs joining it to its neighbours." class="color-invertible"/>
```

*The diagram shows a finite MPS as a row of site tensors, each carrying a physical index and linked to its neighbours through virtual bonds; the two ends carry trivial (dimension-one) boundary bonds.*
This page explains the *gauge freedom* inherent in that representation and the *canonical forms* MPSKit uses to fix it, so that the `AL`, `AR`, `C`, and `AC` you see throughout the API stop looking like arbitrary labels.
It is about understanding rather than construction: for how to *build* a state see [Constructing states](@ref howto_states), and for the full type signatures see the [States](@ref lib_states) reference.

## Gauge freedom

The tensors that make up an MPS are not uniquely determined by the physical state they encode.
On any virtual bond you can insert an invertible matrix `C` together with its inverse `C⁻¹`, since their product is the identity and leaves the contracted network unchanged.
Absorbing `C` into the tensor on one side of the bond and `C⁻¹` into the tensor on the other redefines both local tensors while representing exactly the same physical state.

```@raw html
<img src="./mps_gauge_freedom.png" alt="Inserting C times its inverse on a virtual bond and reabsorbing each factor into the neighbouring tensor, leaving the physical state unchanged." class="color-invertible"/>
```

*The diagram shows an identity `C · C⁻¹` inserted on a virtual bond, with each factor then absorbed into the tensor on its side of the bond — a change of representation that leaves the physical state untouched.*

This freedom is not a nuisance to be tolerated; it is a resource.
Because the local tensors can be reshaped at will, we can choose the gauge on every bond to give the tensors especially convenient properties, without ever changing the state they describe.
The two choices below are the ones that matter in practice.

## Canonical forms

At each site there are two particularly convenient gauges, the *left*- and *right-canonical* forms.

In the left-canonical form a site tensor is a **left isometry**: contracting it with its own conjugate over the left virtual and physical indices yields the identity on the right virtual space.
By convention these tensors are called `AL`.

```jldoctest mps_states
julia> state = FiniteMPS(rand, ComplexF64, 10, ℂ^2, ℂ^4);

julia> al = state.AL[3];

julia> al' * al ≈ id(right_virtualspace(al))
true
```

In the right-canonical form a site tensor is instead a **right isometry**, an identity when contracted over its right virtual and physical indices; these are called `AR`.
The check uses TensorKit's `repartition` to regroup the tensor's indices so that the isometry contraction can be written directly.

```jldoctest mps_states
julia> ar = state.AR[3];

julia> repartition(ar, 1, 2) * repartition(ar, 1, 2)' ≈ id(left_virtualspace(ar))
true
```

The two forms can be mixed: every tensor to the left of a chosen bond is put in the left gauge and every tensor to its right in the right gauge.
The gauge transformation sitting on that one bond can no longer be absorbed without spoiling the isometry property on one side, so it remains as an explicit **center bond tensor** `C`.
`C` is exactly the transformation that relates the left- and right-gauged tensors across its bond.
For convenience a single site tensor can also be left in the *center-site* form `AC`, which is the center tensor absorbed into the neighbouring isometry from either side:

```jldoctest mps_states
julia> al * state.C[3] ≈ state.AC[3]
true
```

Equivalently, absorbing the center tensor on bond `2` into the right isometry at site `3` reproduces the same center-site tensor:

```jldoctest mps_states
julia> repartition(state.C[2] * repartition(ar, 1, 2), 2, 1) ≈ state.AC[3]
true
```

These relations — `AL' * AL = 1`, `AR * AR' = 1`, and `AC = AL · C = C · AR` — hold for any validly gauged MPS, which is why the checks above return `true` even for a random state.

## Automatic gauge management

MPS algorithms move through these forms constantly: a DMRG sweep, for instance, carries the center site across the chain, gauging each tensor as it goes.
Doing that bookkeeping by hand would be tedious and error-prone, so the state objects do it for you.
A [`FiniteMPS`](@ref) (and likewise an [`InfiniteMPS`](@ref)) behaves as an automatic gauge manager: querying `state.AL`, `state.AR`, `state.C`, or `state.AC` returns the requested form, computing and caching it on demand and recomputing it when the underlying tensors have changed.
The intended experience is that you never think about how the state is gauged — it is handled automagically.

!!! warning "In-place mutation defeats the cache"
    A `FiniteMPS` detects that a form needs recomputing only when a tensor is *replaced* through an indexing assignment.
    Changing a tensor's data in place keeps the same object, so the automatic recomputation is not triggered and stale gauged tensors may be returned.
    Assign a new tensor rather than mutating an existing one.

### The center-gauge overlap insight

The payoff of the mixed gauge is visible in a computation as basic as the norm.
To compute the overlap of a state with itself, bring any bond into the center gauge.
Everything to the left of that bond is built from left isometries and contracts to the identity, everything to the right is built from right isometries and does the same, and the entire network collapses to the overlap of the center bond tensor `C` with itself.
The overlap is therefore the same whichever bond you pick:

```jldoctest mps_states
julia> using LinearAlgebra

julia> d = dot(state, state);

julia> all(c -> dot(c, c) ≈ d, state.C)
true
```

This is not a special trick for the norm; the same collapse-to-the-center reasoning is what makes environments (see [Environments](@ref concept_environments)) and local expectation values cheap to evaluate in the canonical gauge.

## Finite versus infinite gauging

The gauge machinery is shared between finite and infinite states, but the way the forms are kept current differs, because the two have very different structure.

A [`FiniteMPS`](@ref) has genuine boundaries and mutable per-site tensors, so it gauges *lazily*: each form is recomputed only for the tensors it actually depends on, and invalidation is decided by object identity (`===`) — replacing a tensor marks the left-gauged tensors to its right and the right-gauged tensors to its left as stale, leaving the rest cached.
An [`InfiniteMPS`](@ref) instead repeats a finite unit cell periodically, so there is no left or right end to anchor a partial recompute: every tensor lies both to the right and to the left of any change, and all forms are recomputed together whenever a tensor changes.
<!-- REVIEW: the finite-vs-infinite characterization (finite = lazy per-tensor recompute keyed on ===; infinite = all forms recomputed together because tensors repeat periodically) is taken from the legacy man/states.md prose; confirm it still matches the current implementation. -->

## Variants

Two further state types reuse the same canonical-form vocabulary for more specialized settings:

- A [`WindowMPS`](@ref) represents a finite window of mutable tensors embedded in an infinite environment on both sides — a finite region living inside two [`InfiniteMPS`](@ref) tails.
- A [`MultilineMPS`](@ref) is a stack of [`InfiniteMPS`](@ref) objects used to represent the two-dimensional networks that arise in boundary-MPS methods.

## Where to go next

- To build states from tensors, from spaces, or as product states, see [Constructing states](@ref howto_states).
- For the full type signatures and docstrings, see the [States](@ref lib_states) reference.
- For how the canonical gauge makes contractions cheap, see [Environments](@ref concept_environments).
