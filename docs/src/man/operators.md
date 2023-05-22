# [Operators](@id um_operators)

## DenseMPO

This operator is used for statistical physics problems. It is simply a periodic array of mpo tensors.

Can be created using
```julia
DenseMPO(t::AbstractArray{T,1}) where T<:MPOTensor
```

## SparseMPO

`SparseMPO` is similar to a `DenseMPO`, in that it again represents an mpo tensor, periodically repeated. However this type keeps track of all internal zero blocks, allowing for a more efficient representation of certain operators (such as time evolution operators and quantum hamiltonians). You can convert a sparse mpo to a densempo, but the converse does not hold.


Indexing a `SparseMPO` returns a `SparseMPOSlice` object, which has 3 fields

```@docs
SparseMPOSlice
```

When indexing a SparseMPOSlice at index `[j, k]` (or equivalently `SparseMPO[i][j, k]`), the code looks up the corresponding field in `Os[j, k]`. Either that element is a tensormap, in which case it gets returned. If it equals `zero(E)`, then we return a tensormap
```julia
domspaces[j] * pspace ← pspace * imspaces[k]
```
with norm zero. If the element is a nonzero number, then implicitly we have the identity operator there (multiplied by that element).

The idea here is that you don't have to worry about the underlying structure, you can just index into a sparsempo as if it is a vector of matrices. Behind the scenes we then optimize certain contractions by using the sparsity structure.

SparseMPO are always assumed to be periodic in the first index (position).
In this way, we can both represent periodic infinite mpos and place dependent finite mpos.

## MPOHamiltonian

We represent all quantum hamiltonians in their mpo form. As an example, the following bit of code constructs the ising hamiltonian.

```julia
sx, sy, sz, id = nonsym_spintensors(1 // 2)
data = Array{Any,3}(missing, 1, 3, 3)
data[1, 1, 1] = id
data[1, 1, 2] = -sz
data[1, 2, 3] = sz
data[1, 1, 3] = 3 * sx
ham = MPOHamiltonian(data);
```

When we work with symmetries, it is often not possible to represent the entire hamiltonian as a sum of a product of one-body operators.
For example, in the XXZ Heisenberg model only the sum ``sx * sx + sy * sy + sz * sz`` is su(2) symmetric, but individually none of the terms are.
It is for this reason that we use 4 leg mpo tensors in this hamiltonian object. The following bit of code

```julia
ham[1][1, 1]
```

Will print out a tensormap mapping `virtual_space ⊗ physical_space` to `physical_space ⊗ virtual_space`.
The conversion to mpo tensors was done automagically behind the scenes!


An `MPOHamiltonian` is really just a `SparseMPO`, but with the garantuee that the sub-blocks are upper triangular. This effectively means that they are finite state machines, which are general enough to encode any hamiltonian but are efficient to construct environments for.
