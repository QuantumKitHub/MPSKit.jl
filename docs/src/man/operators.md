# [Operators](@id um_operators)

## MPOHamiltonian

### usage

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

Will print out a tensormap mapping (virtual space x physical space) to (physical space x virtual space).
The conversion to mpo tensors was done automagically behind the scenes!

MPOHamiltonian are always assumed to be periodic in the first index (position).
In this way, we can both represent periodic infinite hamiltonians and place dependent finite hamiltonians.

### implementation details

The mpohamiltonian has 3 fields:

```julia
Os::PeriodicArray{Union{E,T},3}
domspaces::PeriodicArray{S,2}
pspaces::PeriodicArray{S,1}

Where T<:MPOTensor, E<:Number
```

When indexing the hamiltonian at index [i,j,k], the code looks up the corresponding field in Os[i,j,k]. Either that element is a tensormap, in which case it gets returned. If it equals zero(E), then we return a tensormap
```julia
domspaces[i,j]*pspaces[i] ← pspaces[i]*domspaces[i+1,k]
```
wither norm zero. If the element is a nonzero number, then implicitly we have the identity operator there (multiplied by that element). Of course in that case,
```julia
domspaces[i,j] == domspaces[i+1,k]
```
Otherwise the identity operator can't be uniquely defined.

The overarching idea is that the user never has to worry about these inner fields, and can act as if the mpohamiltonian is a 3 dimensional array of tensormaps, while we optimize behind the scenes for the special cases where the operator is zero or the identity.

## InfiniteMPO

This operator is used for statistical physics problems. It is simply a periodic array of mpo tensors.

Can be created using
```julia
InfiniteMPO(t::AbstractArray{T,1}) where T<:MPOTensor
```
