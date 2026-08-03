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
```

## Time-evolution MPOs

For evolving with an explicitly constructed propagator MPO, e.g. for an [`InfiniteMPS`](@ref), use [`make_time_mpo`](@ref) with one of the expansion algorithms below.

```@docs; canonical=false
make_time_mpo
TaylorCluster
WI
WII
```
