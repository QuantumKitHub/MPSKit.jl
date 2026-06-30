# [Operators](@id lib_operators)

Reference for matrix product operators and Hamiltonians. The full, canonical
docstrings for the whole package live in the [Library](@ref lib_index) index.

## Matrix product operators

```@docs; canonical=false
AbstractMPO
MPO
FiniteMPO
InfiniteMPO
MultilineMPO
```

## Hamiltonians

```@docs; canonical=false
MPOHamiltonian
FiniteMPOHamiltonian
InfiniteMPOHamiltonian
```

## Jordan-block MPO tensors

```@docs; canonical=false
JordanMPOTensor
```

<!-- REVIEW: the exported `JordanMPOTensorMap` is public API but has no docstring;
add one in src/operators/jordanmpotensor.jl, then list it above. -->


## Operator algebra

```@docs; canonical=false
MultipliedOperator
TimedOperator
UntimedOperator
LazySum
```
