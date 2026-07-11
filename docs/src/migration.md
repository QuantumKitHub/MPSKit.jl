# Migration guide

This page collects notes for upgrading between MPSKit releases when a change requires action
on your side.
It summarizes the breaking changes recorded in the [Changelog](@ref); consult that page for
the complete, per-version list of changes.

## Unreleased

### `environments` now uses a single positional contract

[`environments`](@ref) now follows one positional contract for every state and operator kind:

```
environments(below, operator, above, alg)
```

where `alg` (slot 4) is the environment algorithm.
The operator form requires an explicit `above` argument.
Auxiliary inputs are now keyword-only: `leftstart`/`rightstart` for finite and window
environments, and `lenvs`/`renvs` for window and quasiparticle environments.

The two-argument form `environments(below, above)` (two states) is reserved for the
operator-free overlap environments.
There is no two-argument `environments(below, operator)` shorthand: a two-argument call is
always interpreted as an overlap, because the second argument cannot be disambiguated between
a ket and an operator (this is undecidable for density matrices, where states and operators
share a representation).
If you previously relied on a two-argument operator call, pass the operator explicitly with an
`above` state.
See [#436](https://github.com/QuantumKitHub/MPSKit.jl/pull/436).

## Earlier releases

The [Changelog](@ref) records further changes across the 0.13.x series, including a refactor
of the `IDMRG` implementation to the `IterativeSolver` interface
([#348](https://github.com/QuantumKitHub/MPSKit.jl/pull/348)), fixes to `changebonds`
consistency ([#415](https://github.com/QuantumKitHub/MPSKit.jl/pull/415)), and a refactor of
time-evolution MPO construction
([#422](https://github.com/QuantumKitHub/MPSKit.jl/pull/422)).
