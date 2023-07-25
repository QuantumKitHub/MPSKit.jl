# Prerequisites

The following sections describe the prerequisites for using MPSKit. If you are already
familiar with the concepts of MPSKit and TensorKit, you can skip to the [Overview](@ref)
section.

## TensorKit

```@example tensorkit
using TensorKit
```

MPSKit uses the tensors defined in [TensorKit.jl](https://github.com/Jutho/TensorKit.jl) as
its underlying data structure. This is what allows the library to be generic with respect to
the symmetry of the tensors. The main difference with regular multi-dimensional arrays is
the notion of a partition of the dimensions in **incoming** and **outgoing**, which are
respectively called **domain** and **codomain**. In other words, a `TensorMap` can be
interpreted as a linear map from its domain to its codomain. Additionally, as generic
symmetries are supported, in general the structure of the indices are not just integers, but
are given by spaces.

The general syntax for creating a tensor is one of the following equivalent forms:
```julia
TensorMap(initializer, scalartype, codomain, domain)
TensorMap(initializer, scalartype, codomain ← domain) # ← is the `\leftarrow` operator
```

For example, the following creates a random tensor with three legs, each of which has
dimension two, however with different partitions.

```@example tensorkit
V1 = ℂ^2 # ℂ is the `\bbC` operator, equivalent to ComplexSpace(10)
t1 = Tensor(rand, Float64, V1 ⊗ V1 ⊗ V1) # all spaces in codomain
t2 = TensorMap(rand, Float64, V1, V1 ⊗ V1) # one space in codomain, two in domain

try
    t1 + t2 # incompatible partition
catch err
    println(err)
end

try
    t1 + permute(t2, (1, 2, 3), ()) # incompatible arrows
catch err
    println(err)
end
```

These abstract objects can represent not only plain arrays but also symmetric tensors. The
following creates a symmetric tensor with ℤ₂ symmetry, again with three legs of dimension
two. However, now the dimension two is now split over even and odd sectors of ℤ₂.

```@example tensorkit
V2 = Z2Space(0 => 1, 1 => 1)
t3 = TensorMap(rand, Float64, V2 ⊗ V2, V2)
```

For more information, check out the [TensorKit documentation](https://jutho.github.io/TensorKit.jl/stable/)!

## Conventions



Within MPSKit we defined a set of [states](@ref um_states), a number of [operators](@ref um_operators) and some [algorithms](@ref um_algorithms) which combine the two in a nontrivial way.

As a simple example we can define a FiniteMPS
```julia
state = FiniteMPS(rand,ComplexF64,10,ℂ^2,ℂ^10);
```

A hamiltonian operator
```julia
opp = nonsym_ising_ham();
```

And use this to find the groundstate
```julia
(groundstate,_) = find_groundstate(state,opp,DMRG());
```

## Tips & tricks

- There is an examples folder
- Julia inference is taxed a lot; the first time any function is run takes a really long time
- There are predefined hamiltonians in [MPSKitModels.jl](https://github.com/maartenvd/MPSKitModels.jl)
