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

## Quasiparticle states

Excitation ansätze produced by [`excitations`](@ref).
These behave as vectors and are normally obtained from `excitations` rather than constructed directly.

```@docs; canonical=false
QP
LeftGaugedQP
RightGaugedQP
```

