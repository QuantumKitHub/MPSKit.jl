# Basics

The following few sections should help you on your way to setting up and running simulations.

## TensorKit

MPSKit works on "TensorMap" objects defined in [TensorKit.jl](https://github.com/Jutho/TensorKit.jl). These abstract objects can represent not only plain arrays but also symmetric tensors. A TensorMap is a linear map from its domain to its codomain.

Initializing a TensorMap can be done using
```julia
TensorMap(initializer,eltype,codomain,domain);
TensorMap(inputdat,codomain,domain);
```

As an example, the following creates a random map from ℂ^10 to ℂ^10 (which is equivalent to a random matrix)
```julia
TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
```
Similarly, the following creates a symmetric tensor
```julia
TensorMap(rand,ComplexF64,Rep[U₁](0=>1)*Rep[U₁](1//2=>3),Rep[U₁](1//2=>1,-1//2=>2))
```

TensorKit defines a number of operations on TensorMap objects
```julia
a = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);

3*a; # multiply by a scalar

a+a; #addition of tensormaps

a*a; #multiplication of tensormaps

a*adjoint(a); #taking the adjoint

dot(a,a); #inner product

permute(a,(1,2),()); # create a new tensormap ℂ^10 * (ℂ^10)' ← nothing

...
```

for more complicated operations, we can use the same @tensor macro defined in [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)
```julia
a = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
b = TensorMap(rand,ComplexF64,ℂ^10,ℂ^10);
@tensor c[-1;-2]:=a[-1,1]*b[1,-2];
```
creates a new TensorMap c equal to a*b.

For more information, check out the [TensorKit documentation](https://jutho.github.io/TensorKit.jl/stable/)!

## Overview

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
