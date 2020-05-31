# [Operators](@id um_operators)

## MPOHamiltonian

We represent all quantum hamiltonians in their mpo form. As an example, the following bit of code constructs the ising hamiltonian.

```julia
(sx,sy,sz,id) = nonsym_spintensors(1//2);
data = Array{Any,3}(missing,1,3,3);
data[1,1,1] = id;
data[1,1,2] = -sz;
data[1,2,3] = sz;
data[1,1,3] = 3*sx;
ham = MPOHamiltonian(data);
```

When we work with symmetries, it is often not possible to represent the entire hamiltonian as a sum of a product of one-body operators.
In xxz for examples, only the sum (sx sx + sy sy + sz sz) is su(2) symmetric, but individually none of the terms are.
It is for this reason that we use 4 leg mpo tensors in this hamiltonian object. The following bit of code

```julia
ham[1,1,1]
```

Will print out a tensormap mapping (virtual space x physical space) to (virtual space x physical space).
The conversion to mpo tensors was done automagically behind the scenes!

MPOHamiltonian are always assumed to be periodic in the first index (position).
In this way, we can both represent periodic infinite hamiltonians and place dependent finite hamiltonians.

## PeriodicMPO

This operator is used for statistical physics problems. It is simply a 2 dimensional periodic array of mpo tensors.

Can be created using
```julia
PeriodicMPO(t::AbstractArray{T,2}) where T<:MPOTensor
```

## ComAct

We can deal with thermal density matrices by mapping them back to a state with hilbert space p*p'.
Time evolution is then done using the (anti)commutator of the original hamiltonian.
ComAct represents this object and can be created by calling

```julia
anticommutator(ham)
commutator(ham)
```

Only finite density matrices are supported at the moment. If you want to do finite temperature stuff in the thermodynamic limit, then you should manually fuse p*p' and construct the the commutator in this space.
