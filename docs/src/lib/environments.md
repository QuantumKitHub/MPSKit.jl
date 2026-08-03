# [Environments](@id lib_environments)

Reference for MPSKit's environment machinery — the caches that store the partially contracted tensor networks reused throughout the algorithms.
For an explanation of what environments are and why they exist see the concept page on [Environments](@ref concept_environments); this page only lists the API.
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

```@meta
CurrentModule = MPSKit
```

## Constructing environments

```@docs; canonical=false
environments
```

## Querying environments

```@docs; canonical=false
leftenv
rightenv
```

## Environment types

The concrete environment types below are returned by [`environments`](@ref) and are managed automatically by the algorithms.
They are implementation details — you normally obtain them from `environments` rather than constructing them directly — and are not part of the public API.

```@docs; canonical=false
AbstractMPSEnvironments
FiniteEnvironments
InfiniteEnvironments
InfiniteQPEnvironments
```
