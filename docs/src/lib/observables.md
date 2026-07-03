# [Observables and analysis](@id lib_observables)

Reference for extracting physical quantities and analysis diagnostics from an MPS.
For a task-oriented walkthrough see the how-to guide [Computing observables](@ref howto_observables).
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Expectation values

```@docs; canonical=false
expectation_value
```

## Correlators

```@docs; canonical=false
correlator
```

## Convergence diagnostics

```@docs; canonical=false
variance
```

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
- REVIEW: `expectation_value(::MultilineMPS, ::MultilineMPO, envs...)` in
  `src/algorithms/expval.jl` carries a `# TODO: fix environments` comment in the source;
  the docstring surfaces as-is here, but the method's correctness/status should be
  confirmed before calling this reference page complete.
- REVIEW: `variance(::InfiniteQP, ::InfiniteMPOHamiltonian, envs)` in
  `src/algorithms/toolbox.jl` carries a `# I remember there being an issue here @gertian?`
  comment, suggesting the method may be unreliable; flagging for maintainer judgment on
  whether it needs a caveat note or an `!!! warning` admonition on this page.
-->
