# [States](@id lib_states)

Reference for the matrix product state types.
The full, canonical docstrings for the whole package live in the [Library](@ref lib_index) index.

## Matrix product states

```@docs; canonical=false
FiniteMPS
InfiniteMPS
WindowMPS
MultilineMPS
```

<!-- REVIEW: the exported quasiparticle-state types `QP`, `LeftGaugedQP`, and
`RightGaugedQP` are part of the public API but currently have NO docstrings, so
they cannot be listed here. Add docstrings in src/states/quasiparticle_state.jl,
then add a "Quasiparticle states" @docs block here. -->

