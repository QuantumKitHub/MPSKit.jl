# [Bond dimension](@id lib_bond_dimension)

Reference for changing the bond dimension of a state — expanding or truncating its virtual spaces — and for inspecting those virtual spaces directly.
For a task-oriented walkthrough see [Controlling bond dimension](@ref howto_bond_dimension); the full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Interface

```@docs; canonical=false
changebonds
changebonds!
```

## Expansion and truncation algorithms

```@docs; canonical=false
OptimalExpand
RandExpand
SvdCut
VUMPSSvdCut
SketchedExpand
```

!!! note
    `SketchedExpand` is experimental: it uses randomized controlled bond expansion (CBE), so its reported error estimate is itself randomized, and it is only defined for `FiniteMPS`.

## Inspecting the virtual spaces

The bond dimension of an MPS or MPO is the dimension of the virtual space living on a given bond.
The accessors below return that `VectorSpace`, whose `dim` gives the numeric bond dimension.

```@docs; canonical=false
left_virtualspace
right_virtualspace
physicalspace
```
