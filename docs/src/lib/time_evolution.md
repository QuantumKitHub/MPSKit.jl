# [Time evolution](@id lib_time_evolution)

Reference for the time-evolution drivers and algorithms.
For a task-oriented walkthrough see the how-to guides.
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Drivers

```@docs; canonical=false
time_evolve
timestep
timestep!
```

## MPS time-evolution algorithms

```@docs; canonical=false
TDVP
TDVP2
BUG
```

## Time-evolution MPOs

For evolving with an explicitly constructed propagator MPO, e.g. for an [`InfiniteMPS`](@ref), use [`make_time_mpo`](@ref) with one of the expansion algorithms below.

```@docs; canonical=false
make_time_mpo
TaylorCluster
WI
WII
```

## MPO–MPS products

Applying an MPO to a state — a propagator MPO among others — goes through [`approximate`](@ref).
The variational algorithms ([`DMRG2`](@ref) and friends) treat the destination as an initial guess, whereas [`Zipup`](@ref) sweeps the product out in one pass and needs none.

```@docs; canonical=false
approximate
approximate!
Zipup
```
