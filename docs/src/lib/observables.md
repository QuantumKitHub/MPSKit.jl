# [Observables and analysis](@id lib_observables)

Reference for extracting physical quantities and analysis diagnostics from an MPS.
For a task-oriented walkthrough see the how-to guide [Computing observables](@ref howto_observables).
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Expectation values

```@docs; canonical=false
expectation_value
```

!!! note "Environments are ignored in the multiline varargs method"
    `expectation_value(::MultilineMPS, ::MultilineMPO, envs...)` accepts environments but does
    not use them: it evaluates the expectation value line by line, and each line recomputes its
    own. Passing environments here therefore saves no work — the result is correct either way.

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
