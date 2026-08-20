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

They offer some convenience functionality for using cartesian indexing (row - column):

You can access properties by calling
```@example states
row = 2
col = 2
al = state.AL[row, col];
```

### The row-shift convention

The row direction and the column direction of a `MultilineMPS` play different roles.
Within a row, a `MultilineMPS` behaves exactly like the `InfiniteMPS` it repeats: columns are periodic, and `state.AL[row, col]`, `state.AR[row, col]`, `state.C[row, col]` and `state.AC[row, col]` behave exactly as they would for `state[row]::InfiniteMPS`.

The row direction is the direction along which a transfer matrix, represented as a [`MultilineMPO`](@ref), is applied.
By convention, row `i` of a `MultilineMPO` maps row `i` of the network onto row `i + 1`, so applying a single row shifts the boundary up by one.

Note that the bra and the ket are therefore different lines.
A `MultilineMPS` is consequently not a state whose expectation value one takes.
This is why [`expectation_value`](@ref) has no method for a `MultilineMPS`/`MultilineMPO` pair, and why the quantity such an iteration converges to is a [`dominant_eigenvalue`](@ref).

### One fixed point, many lines

It is worth being explicit about what the extra rows are doing, because it is easy to read a `MultilineMPS` as a stack of independent states.
It is not.
However many rows the operator has, the network has exactly **one** boundary fixed point: the MPS that comes back to itself after being pushed through all of the rows.

The lines of a `MultilineMPS` are bookkeeping for that single problem.
Line `i + 1` holds the boundary after row `i` has been applied to line `i`, so the lines are successive stages of one boundary travelling through the network rather than separate solutions.
What the circular row shifting buys is that `leading_boundary` can cut the one eigenvalue problem into `nrows` smaller coupled subproblems (one per row) and solve them together, instead of contracting all rows into a single operator with a bond dimension that is the product of theirs.

Each subproblem contributes its own partial factor, and the dominant eigenvalue of the actual fixed point is the **product** of all of them.
Independently of that, the boundary may have a non-trivial unit cell along the chain, in which case each of the `ncols` sites carries its own factor too, and those multiply as well.
[`dominant_eigenvalue`](@ref) accumulates both directions: it returns the product over every site `(i, j)` of the `(nrows, ncols)` unit cell, which is the eigenvalue of the true fixed point for one unit cell of the network.

Applying an operator therefore takes an ordinary [`InfiniteMPS`](@ref) and pushes it through every row in turn, advancing it by one full period of the network:

```julia
O * ψ == O[end] * (… * (O[2] * (O[1] * ψ)))
```

### What is currently supported

Algorithms only support lines that are themselves infinite: a `MultilineMPS` used with `leading_boundary` and friends is built out of [`InfiniteMPS`](@ref) lines.
[`FiniteMPS`](@ref) lines are accepted by the type and by the vector constructor, so that finite multiline boundaries can be built and inspected.
However, no algorithm handles them yet and they will fail somewhere further down.

### Subtleties

- **`size` vs. iteration:** `size(state)` returns `(nrows, ncols)`, describing the lattice shape.
Iterating over a `MultilineMPS` (or indexing it with a single integer, `state[i]`) instead yields the individual MPS *lines*, so `length(state) == nrows`, not `nrows * ncols`.
Code that wants to operate line-by-line should use `state[i]`/`parent(state)`, while code that wants the lattice shape should use `size`.
- **Norms and inner products:** `dot`/`norm` sum the contribution of every row, so a `MultilineMPS` whose individual rows are each normalized to 1 does *not* itself have norm 1: `norm(state) == sqrt(nrows)` for `nrows` identical normalized rows.

These objects are also used extensively in the context of [PEPSKit.jl](https://github.com/QuantumKitHub/PEPSKit.jl).

