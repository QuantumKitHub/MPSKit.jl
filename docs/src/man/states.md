# [States](@id um_states)

```@setup states
using MPSKit
using TensorKit
using LinearAlgebra: dot
```

## FiniteMPS

A [`FiniteMPS`](@ref) is - at its core - a chain of MPS tensors.

```@raw html
<img src="../finite_mps_definition.png" alt="finite MPS" class="color-invertible"/>
```

### Construction

If you already have the state of interest, it is straightforward to convert it to an MPS as follows:

```@example states
ψ_dense = rand((ℂ^2)^4)
ψ_mps = FiniteMPS(ψ_dense)
```

However, typically this is not the case, as storing the full state becomes expensive rather quickly.
Then, `FiniteMPS` are best constructed by first specifying a [`FiniteMPSManifold`](@ref) that encodes the physical and (maximal) virtual spaces:

```@example states
pspaces = fill(ℂ^2, 10)         # physical spaces (Vector)
max_virtualspace = ℂ^4          # single max virtual space
manifold = FiniteMPSManifold(pspaces, max_virtualspace)
ψ = rand(manifold)              # random normalized FiniteMPS
```

Finally, it is also possible to build them from explicit MPS tensors, by passing them directly.
Here we construct the

```@example states
As = [rand(ComplexF64, ℂ^1 ⊗ ℂ^2  ← ℂ^1) for _ in 1:L];
ψ_from_As = FiniteMPS(As)
```

!!! warning "Full rank spaces"

    As the `FiniteMPS` object handles tensors in well-chosen gauges, the virtualspaces, as well as the associated tensors might reduce in size.
    This can be achieved in a lossless manner whenever the spaces are not full rank, in the following sense:
    ```julia
    left_virtualspace(A) ⊗ physicalspace(A) ≿ right_virtualspace(A) &&
        left_virtualspace(A)' ≾ physicalspace(A) ⊗ right_virtualspace(A)'
    ```

!!! warning "Edge spaces"

    It is possible for a `FiniteMPS` object to have non-trivial left- and/or right edge spaces.
    This can be convenient whenever the state is embedded in a larger system (e.g. as part of a [`WindowMPS`](@ref)), or to allow for non-trivially charged symmetric states.
    Therefore, be mindful that when constructing a `FiniteMPS` from tensors directly, you need to handle the edges separately.

### Gauging and canonical forms

An MPS representation is not unique: for every virtual bond we can insert $C \cdot C^{-1}$ without altering the state.
Then, by redefining the tensors on both sides of the bond to include one factor each, we can change the representation.

```@raw html
<img src="../mps_gauge_freedom.png" alt="MPS gauge freedom" class="color-invertible"/>
```

There are two particularly convenient choices for the gauge at a site, the so-called left and right canonical form.
For the left canonical form, all tensors to the left of a site are gauged such that they become left-isometries.
By convention, we call these tensors `AL`.

```@example states
al = state.AL[3]
al' * al ≈ id(right_virtualspace(al))
```

Similarly, the right canonical form turns the tensors into right-isometries.
By convention, these are called `AR`.

```@example states
ar = state.AR[3]
repartition(ar, 1, 2) * repartition(ar, 1, 2)' ≈ id(left_virtualspace(ar))
```

It is also possible to mix and match these two forms, where all tensors to the left of a given site are in the left gauge, while all tensors to the right are in the right gauge.
In this case, the final gauge transformation tensor can no longer be absorbed, since that would spoil the gauge either to the left or the right.
This center-gauged tensor is called `C`, which is also the gauge transformation to relate left- and right-gauged tensors.
Finally, for convenience it is also possible to leave a single MPS tensor in the center gauge, which we call `AC = AL * C`

```@example states
c = state.C[3] # to the right of site 3
c′ = state.C[2] # to the left of site 3
al * c ≈ state.AC[3] ≈ repartition(c′ * repartition(ar, 1, 2), 2, 1)
```

These forms are often used throughout MPS algorithms, and the [`FiniteMPS`](@ref) object acts as an automatic manager for this.
It will automatically compute and cache the different forms, and detect when to recompute whenever needed.
For example, in order to compute the overlap of an MPS with itself, we can choose any site and bring that into the center gauge.
Since then both the left and right side simplify to the identity, this simply becomes the overlap of the gauge tensors:

```@example states
d = dot(state, state)
all(c -> dot(c, c) ≈ d, state.C)
```

### Implementation details

Behind the scenes, a `FiniteMPS` has 4 fields

```julia
ALs::Vector{Union{Missing,A}}
ARs::Vector{Union{Missing,A}}
ACs::Vector{Union{Missing,A}}
Cs::Vector{Union{Missing,B}}
```

and calling `AL`, `AR`, `C` or `AC` returns lazy views over these vectors that instantiate the tensors whenever they are requested.
Similarly, changing a tensor will poison the `ARs` to the left of that tensor, and the `ALs` to the right.
The idea behind this construction is that one never has to worry about how the state is gauged, as this gets handled automagically.

!!! warning
    While a `FiniteMPS` can automatically detect when to recompute the different gauges, this requires that one of the tensors is set using an indexing operation.
    In particular, in-place changes to the different tensors will not trigger the recomputation.

## InfiniteMPS

An [`InfiniteMPS`](@ref) represents a periodically repeating unit cell of MPS tensors.

### Construction

Similar to `FiniteMPS`, the easiest way of constructing an `InfiniteMPS` is by specifying an [`InfiniteMPSManifold`](@ref) describing one unit cell:

```@example states
pspaces = [ℂ^2, ℂ^2]       # 2-site unit cell
vspaces = [ℂ^4, ℂ^5]       # virtual space to the left of each site
manifold = InfiniteMPSManifold(pspaces, vspaces)
ψinf = rand(manifold)
```

Alternatively, we may also start from explicit site tensors:

```@example states
As = [rand(ComplexF64, imanifold[i]) for i in 1:length(imanifold)]
ψinf2 = InfiniteMPS(As)
```

### Gauging and canonical forms

Much like for `FiniteMPS`, we can again query the gauged tensors `AL`, `AR`, `C` and `AC`.
Here however, the implementation is much easier, since they all have to be recomputed whenever a single tensor changes.
This is a result of periodically repeating the tensors, every `AL` is to the right of the changed site, and every `AR` is to the left.
As a result, the fields are simply

```julia
AL::PeriodicArray{A,1}
AR::PeriodicArray{A,1}
C::PeriodicArray{B,1}
AC::PeriodicArray{A,1}
```

## WindowMPS

A [`WindowMPS`](@ref) or segment MPS can be seen as a mix between an [`InfiniteMPS`](@ref) and a [`FiniteMPS`](@ref).
It represents a window of mutable tensors (a finite MPS), embedded in an infinite environment (two infinite MPSs).
It can therefore be created accordingly, ensuring that the edges match:

```@example states
infinite_state = rand(InfiniteMPSManifold(ℂ^2, ℂ^4))
finite_manifold = FiniteMPSManifold(fill(ℂ^2, 5), ℂ^4; left_virtualspace=ℂ^4, right_virtualspace=ℂ^4)
finite_state = rand(ComplexF64, finite_manifold)
window = WindowMPS(infinite_state, finite_state, infinite_state)
```

Algorithms will then act on this window of tensors, while leaving the left and right infinite states invariant.

## MultilineMPS

A two-dimensional classical partition function can often be represented by an infinite tensor network.
There are many ways to evaluate such a network, but here we focus on the so-called boundary MPS methods.
These first reduce the problem from contracting a two-dimensional network to the contraction of a one-dimensional MPS, by finding the fixed point of the row-to-row (or column-to-column) transfer matrix.
In these cases however, there might be a non-trivial periodicity in both the horizontal as well as vertical direction.
Therefore, in MPSKit they are represented by [`MultilineMPS`](@ref), which are simply a repeating set of [`InfiniteMPS`](@ref).

```@example states
state = MultilineMPS(fill(infinite_state, 2))
```

They offer some convenience functionality for using cartesian indexing (row - column):

You can access properties by calling
```@example states
row = 2
col = 2
al = state.AL[row, col];
```

These objects are also used extensively in the context of [PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl).

