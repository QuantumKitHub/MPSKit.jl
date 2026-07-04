# [Observables and analysis](@id lib_observables)

Reference for extracting physical quantities and analysis diagnostics from an MPS.
For a task-oriented walkthrough see the how-to guide [Computing observables](@ref howto_observables).
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Expectation values

```@docs; canonical=false
expectation_value
```

!!! warning "Multiline environments"
    The `expectation_value(::MultilineMPS, ::MultilineMPO, envs...)` method reuses the
    passed environments without recomputing them for the operator (see the `# TODO: fix
    environments` note in `src/algorithms/expval.jl`).
    Results along this code path should be cross-checked against an independent calculation
    until the environment handling is finalized.

## Correlators

```@docs; canonical=false
correlator
```

## Convergence diagnostics

```@docs; canonical=false
variance
```

!!! warning "Variance of infinite quasiparticle states"
    The `variance(::InfiniteQP, ::InfiniteMPOHamiltonian, envs)` method carries an
    unresolved implementation note in `src/algorithms/toolbox.jl` and may be unreliable.
    Verify its output before using it as a convergence diagnostic for quasiparticle states.

## Transfer matrix and correlation length

```@docs; canonical=false
correlation_length
marek_gap
transfer_spectrum
transferplot
```

## Entanglement

Entropy and entanglement spectrum are computed from the state's bond/gauge tensors.
See the how-to [Entanglement entropy and spectrum](@ref howto_entanglement) for worked recipes.

```@docs; canonical=false
entropy
entanglement_spectrum
entanglementplot
```

<!--
Maintainer notes:
- Symbols included: `expectation_value`, `correlator`, `variance`, `correlation_length`,
  `marek_gap`, `transfer_spectrum`, `transferplot`, `entropy`, `entanglement_spectrum`,
  `entanglementplot`.
- The `# TODO: fix environments` concern on `expectation_value(::MultilineMPS, ::MultilineMPO)`
  (`src/algorithms/expval.jl`) and the unresolved-issue comment on
  `variance(::InfiniteQP, ::InfiniteMPOHamiltonian)` (`src/algorithms/toolbox.jl`) are now
  surfaced to readers via `!!! warning` admonitions above. Remove those warnings once the
  underlying source issues are resolved.
-->
