# [Excitations](@id lib_excitations)

Reference for the excitation interface, its algorithms, and the quasiparticle state types it produces.
For a task-oriented walkthrough see the how-to guides.
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Interface

```@docs; canonical=false
excitations
```

## Algorithms

```@docs; canonical=false
QuasiparticleAnsatz
FiniteExcited
ChepigaAnsatz
ChepigaAnsatz2
```

## Quasiparticle states

These are the ansatz states produced by, and passed to, `excitations` on top of a ground state.

```@docs; canonical=false
QP
LeftGaugedQP
RightGaugedQP
```

<!--
Maintainer notes:

Symbols documented on this page: `excitations`, `QuasiparticleAnsatz`, `FiniteExcited`,
`ChepigaAnsatz`, `ChepigaAnsatz2`, `QP`, `LeftGaugedQP`, `RightGaugedQP`.
-->
