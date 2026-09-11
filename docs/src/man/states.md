# [States](@id um_states)

```@setup states
using MPSKit
using TensorKit
using LinearAlgebra: dot
```

## FiniteMPS

A [`FiniteMPS`](@ref) is - at its core - a chain of mps tensors.

```@raw html
<img src="./finite_mps_definition.png" alt="finite MPS" class="color-invertible"/>
```

### Usage

A `FiniteMPS` can be created by passing in a vector of tensormaps:

```@example states
L = 10
data = [rand(ComplexF64, ℂ^1 ⊗ ℂ^2  ← ℂ^1) for _ in 1:L];
state = FiniteMPS(data)
```

Or alternatively by specifying its structure

```@example states
max_bond_dimension = ℂ^4
physical_space = ℂ^2
state = FiniteMPS(rand, ComplexF64, L, physical_space, max_bond_dimension)
```

You can take dot products, renormalize!, expectation values,....

### Gauging and canonical forms

An MPS representation is not unique: for every virtual bond we can insert $C \cdot C^{-1}$ without altering the state.
Then, by redefining the tensors on both sides of the bond to include one factor each, we can change the representation.

```@raw html
<img src="./mps_gauge_freedom.png" alt="MPS gauge freedom" class="color-invertible"/>
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

An [`InfiniteMPS`](@ref) can be thought of as being very similar to a finite mps, where the set of tensors is repeated periodically.

It can also be created by passing in a vector of `TensorMap`s:

```@example states
data = [rand(ComplexF64, ℂ^4 ⊗ ℂ^2  ← ℂ^4) for _ in 1:2]
state = InfiniteMPS(data)
```

or by initializing it from given spaces

```@example states
phys_spaces = fill(ℂ^2, 2)
virt_spaces = [ℂ^4, ℂ^5] # by convention to the right of a site
state = InfiniteMPS(phys_spaces, virt_spaces)
```

Note that the code above creates an `InfiniteMPS` with a two-site unit cell, where the given virtual spaces are located to the right of their respective sites.

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
infinite_state = InfiniteMPS(ℂ^2, ℂ^4)
finite_state = FiniteMPS(5, ℂ^2, ℂ^4; left=ℂ^4, right=ℂ^4)
window = WindowMPS(infinite_state, finite_state, infinite_state)
```

Algorithms will then act on this window of tensors, while leaving the left and right infinite states invariant.

## MultilineMPS

A two-dimensional classical partition function can often be represented by an infinite tensor network.
There are many ways to evaluate such a network, but here we focus on the so-called boundary MPS methods.
These first reduce the problem from contracting a two-dimensional network to the contraction of a one-dimensional MPS, by finding the fixed point of the row-to-row (or column-to-column) transfer matrix.
In these cases however, there might be a non-trivial periodicity in both the horizontal as well as vertical direction.
Therefore, in MPSKit they are represented by [`MultilineMPS`](@ref), which are simply a repeating set of MPS lines, one per row of the network.

```@example states
state = MultilineMPS(fill(infinite_state, 2))
```

Properties are accessed with cartesian (row, column) indexing:

```@example states
al = state.AL[2, 1];
```

### The row-shift convention

Row `i` of a [`MultilineMPO`](@ref) maps row `i` of the network onto row `i + 1`.
`environments`, the derivative operators and [`dominant_eigenvalue`](@ref) all follow it, pairing `state[i + 1]` as the bra against `O[i]` acting on the ket `state[i]`.
Within a row, `state.AL[i, j]` and friends behave exactly as they would for `state[i]::InfiniteMPS`.

Bra and ket are therefore different lines, so a `MultilineMPS` is not a state whose expectation value makes sense to take.

### One fixed point, many lines

It is worth mentioning explicitly what the point of the extra rows is, since it is easy to read a `MultilineMPS` as a stack of independent states.
It is not.
However many rows the operator has, the network has exactly one boundary fixed point, namely the MPS that returns to itself after being pushed through all of the rows.
The lines of a `MultilineMPS` are bookkeeping for that single problem.
Line `i + 1` is the boundary after row `i` has been applied, so this lets [`leading_boundary`](@ref) cut it into `nrows` coupled subproblems instead of contracting every row into one operator.
Each subproblem contributes a partial factor, and only their product is the eigenvalue of the fixed point.
Independently of that, the `ncols` sites of the unit cell contribute factors of their own on top.
[`dominant_eigenvalue`](@ref) accumulates both.

Applying an operator therefore acts on an ordinary [`InfiniteMPS`](@ref), advancing it by one full period:

```julia
O * ψ == O[end] * (… * (O[2] * (O[1] * ψ)))
```

### Subtleties

- **`size` vs. iteration:** `size(state)` is the `(nrows, ncols)` lattice shape, while `length`, iteration and `state[i]` refer to the lines, so `length(state) == nrows`. See [`Multiline`](@ref MPSKit.Multiline).
- **Norms:** `dot`/`norm` sum over rows, so `norm(state) == sqrt(nrows)` for `nrows` normalized rows.
- **Finite lines** are accepted by the type and the vector constructor so that finite networks can be built and inspected, but no algorithm supports them yet.

These objects are also used extensively in the context of [PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl).

