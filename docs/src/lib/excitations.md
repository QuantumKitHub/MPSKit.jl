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
Maintainer notes (REVIEW):

Symbols documented on this page: `excitations`, `QuasiparticleAnsatz`, `FiniteExcited`,
`ChepigaAnsatz`, `ChepigaAnsatz2`, `QP`, `LeftGaugedQP`, `RightGaugedQP`.

- REVIEW: the `excitations(H::InfiniteMPO, alg::QuasiparticleAnsatz, ...)` method at
  src/algorithms/excitation/quasiparticleexcitation.jl:212 appears to reference `H_eff`
  before it is assigned. This looks like a source bug rather than a docs issue; this page
  does not present that call path as reliable pending a fix.
- REVIEW: `FiniteExcited`'s umbrella docstring gives a simplified default for `init`
  that does not match the algorithm's actual default in the source. Worth reconciling
  the docstring with the real default rather than relying on this page to paper over it.
-->
