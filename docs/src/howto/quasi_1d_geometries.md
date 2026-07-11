# [Quasi-1D geometries](@id howto_quasi_1d_geometries)

```@meta
DocTestSetup = quote
    using MPSKit, MPSKitModels, TensorKit
end
```

MPSKit works with matrix product states, which are intrinsically one-dimensional objects.
A two-dimensional lattice can still be studied by winding it onto a single chain: the sites of the 2D lattice are placed in a linear order, and a 2D coupling becomes a (possibly long-ranged) coupling between two positions on that chain.
This is the standard "quasi-1D" or "cylinder" approach to 2D systems with tensor networks.

This page collects recipes for building such geometries.
The lattice types and the `@mpoham` helper used here come from [MPSKitModels.jl](https://quantumkithub.github.io/MPSKitModels.jl/dev/), a companion package that supplies lattices, local operators, and ready-made model Hamiltonians.
For the underlying one-dimensional Hamiltonian construction see [Building Hamiltonians](@ref howto_hamiltonians); for the algorithms that consume the resulting operator see [Ground-state algorithms](@ref lib_groundstate).

```@example quasi1d
using MPSKit, MPSKitModels, TensorKit
```

---

## Available lattice geometries

MPSKitModels exposes a small family of lattice types, all subtypes of `AbstractLattice`.
The one-dimensional lattices are `FiniteChain` and `InfiniteChain`.
The genuinely two-dimensional geometries, wrapped for use on a 1D chain, are:

- `FiniteCylinder(L, N)` and `InfiniteCylinder(L, N)` — a strip of circumference `L` rolled into a tube, so the two edges in the transverse direction are identified (periodic around the circumference).
- `FiniteStrip(L, N)` and `InfiniteStrip(L, N)` — the same rectangular patch but with *open* boundaries in the transverse direction (no wrap-around).
- `FiniteLadder(N)` and `InfiniteLadder(N)` — convenience constructors for the width-2 strip, i.e. `FiniteStrip(2, N)` / `InfiniteStrip(2, N)`.
- `FiniteHelix(L, N)` and `InfiniteHelix(L, N)` — a helical winding of the cylinder.
- `HoneycombYC(L, N)` — a honeycomb lattice on an infinite cylinder.

For the square-lattice geometries the two integer arguments are the circumference `L` (number of sites per rung) and the total number of sites `N`; `N` must be a multiple of `L`, and it defaults to `L` (a single rung).
Constructing a lattice does not build any operator; it only fixes the geometry and the site ordering.

```@example quasi1d
InfiniteCylinder(3)     # circumference 3, one rung per unit cell
```

```@example quasi1d
InfiniteLadder(4)       # a two-leg ladder, four sites per unit cell
```

!!! note
    The `Finite*` variants describe a finite patch and produce a [`FiniteMPOHamiltonian`](@ref); the `Infinite*` variants describe a unit cell that repeats along the chain axis and produce an [`InfiniteMPOHamiltonian`](@ref).
    Choose the pair that matches the state you intend to optimize.

---

## Building a Hamiltonian on a cylinder

The model builders in MPSKitModels accept a lattice as an optional positional argument, so switching from a chain to a cylinder is a one-word change.
Here is the transverse-field Ising model on an infinite cylinder of circumference 3:

```@example quasi1d
H_cyl = transverse_field_ising(InfiniteCylinder(3); g = 3.0)
```

The result is an ordinary [`InfiniteMPOHamiltonian`](@ref) with one MPO tensor per site of the unit cell.
It is used exactly like a chain Hamiltonian: pair it with an [`InfiniteMPS`](@ref) whose unit cell has the same length and feed it to `find_groundstate` (see [Ground-state algorithms](@ref lib_groundstate)).
Because the cylinder wraps a 2D coupling onto the chain, the bond dimension required for a converged result grows quickly with the circumference; see the note at the end of this page.

---

## Building a Hamiltonian on a ladder

The same model builders work for a ladder:

```@example quasi1d
H_ladder = heisenberg_XXX(InfiniteLadder(4); spin = 1 // 2)
```

For couplings that are not covered by a ready-made model, assemble the Hamiltonian directly with the `@mpoham` macro.
It sums single-site and two-site local operators over the vertices and bonds that the lattice reports.
The building blocks are `vertices(lattice)` (all sites) and `nearest_neighbours(lattice)` (all nearest-neighbour bonds, including the transverse rung and wrap-around bonds):

```@example quasi1d
lat = InfiniteLadder(4)
H_manual = @mpoham sum(σᶻᶻ(){i, j} for (i, j) in nearest_neighbours(lat)) +
    sum(2.0 * σˣ(){i} for i in vertices(lat))
```

The `O{i, j}` syntax marks `O` as a local operator acting on sites `i` and `j`, where the indices are lattice points; `@mpoham` handles their placement on the 1D chain.
This is the two-dimensional counterpart of the manual chain construction in [Building Hamiltonians](@ref howto_hamiltonians).

---

## How the 2D lattice maps onto the 1D chain

Each 2D lattice defines a linear order on its sites through `linearize_index`, which turns a `(row, column)` coordinate into a single position along the MPS chain.
For a cylinder the sites of one rung are numbered first, then the next rung, and so on.
The rung index runs fastest:

```@example quasi1d
cyl = InfiniteCylinder(3)
linearize_index.(collect(vertices(cyl)))
```

The bonds that `nearest_neighbours` returns reveal how far apart coupled sites end up on the chain.
Listing them as linear-index pairs for the same circumference-3 cylinder:

```@example quasi1d
[linearize_index(i) => linearize_index(j) for (i, j) in nearest_neighbours(cyl)]
```

Two kinds of bonds appear.
Bonds along the chain axis connect a site to the corresponding site one rung over, at chain distance `L` (here `1 => 4`, `2 => 5`, `3 => 6`).
Bonds around the circumference connect neighbours within a rung; the bond that closes the ring connects the first and last site of a rung (here `3 => 1`), spanning `L - 1` sites on the chain.
This wrap-around bond is the longest-ranged coupling in the problem, and it is what makes a wider cylinder more expensive: the MPO must carry that coupling across `L - 1` sites, and the entanglement cut through the chain now spans the whole circumference.

!!! warning "Circumference controls the cost"
    The bond dimension needed for a given accuracy grows rapidly with the circumference `L`, because the entanglement across a cut scales with the length of the boundary it severs (the circumference).
    Keep `L` small in exploratory runs and increase it while watching convergence.

---

## Choosing the site ordering

The default ordering above is not the only option: any permutation of the sites is a valid 1D chain, and a better ordering can shorten the longest-ranged bonds.
`SnakePattern` wraps a lattice together with a permutation function that maps the lattice's natural linear index to its position on the chain.
The wrapped lattice reports the same vertices and bonds, but re-indexed through the pattern.

The example below reverses the site order within every second rung of a finite circumference-3 cylinder — a "boustrophedon" (back-and-forth) snake that keeps successive rungs adjacent:

```@example quasi1d
finite_cyl = FiniteCylinder(3, 6)
pattern = i -> [1, 2, 3, 6, 5, 4][i]      # reverse the second rung
snake = SnakePattern(finite_cyl, pattern)

before = [linearize_index(i) => linearize_index(j) for (i, j) in nearest_neighbours(finite_cyl)]
after = [linearize_index(i) => linearize_index(j) for (i, j) in nearest_neighbours(snake)]
(before, after)
```

Passing `snake` to `@mpoham` (in place of `finite_cyl`) then builds the Hamiltonian in this reordered basis.
A `SnakePattern` built without a pattern, `SnakePattern(lattice)`, uses the identity ordering.

!!! warning "Ordering helpers are broken at v0.4.7"
    MPSKitModels also exports `backandforth_pattern` and `frontandback_pattern` as pre-built cylinder orderings, but at the pinned version (v0.4.7) they error: the returned closure indexes a lazy `Iterators.flatten` object, which has no `getindex` method.
    This is a known upstream bug, so this page shows an explicit permutation instead.

    Any custom permutation must be defined for every linear index that the lattice's bonds reference.
    On an *infinite* lattice the nearest-neighbour bonds reach into the next unit cell, so a permutation defined only on `1:N` errors there with a `BoundsError`; a pattern for an infinite cylinder has to wrap periodically, e.g. `pattern(i) = ((i - 1) ÷ N) * N + perm[mod1(i, N)]`.
    The finite cylinder above sidesteps this.

```@meta
DocTestSetup = nothing
```
