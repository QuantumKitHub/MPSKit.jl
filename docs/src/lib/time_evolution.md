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

<!--
Maintainer footer.
Symbols included: time_evolve, timestep, timestep!, TDVP, TDVP2, make_time_mpo, TaylorCluster, WI, WII.
Caveats:
- `time_evolve!` exists in src but is NOT exported, so it is intentionally omitted from this page.
- REVIEW: the `time_evolve` docstring claims a default `alg=TDVP`, but this is not an actual default
  argument in the `src` method signature. This is a docstring/signature mismatch that should be
  reconciled by the maintainer (either add the default in code or correct the docstring).
-->
