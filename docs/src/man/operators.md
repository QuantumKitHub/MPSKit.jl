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
MPSKit.SparseMPOSlice
```

When indexing a `SparseMPOSlice` at index `[j, k]` (or equivalently `SparseMPO[i][j, k]`), the code looks up the corresponding field in `Os[j, k]`. Either that element is a tensormap, in which case it gets returned. If it equals `zero(E)`, then we return a tensormap
```julia
domspaces[j] * pspace ← pspace * imspaces[k]
```
with norm zero. If the element is a nonzero number, then implicitly we have the identity operator there (multiplied by that element).

The idea here is that you don't have to worry about the underlying structure, you can just index into a sparsempo as if it is a vector of matrices. Behind the scenes we then optimize certain contractions by using the sparsity structure.

SparseMPO are always assumed to be periodic in the first index (position).
In this way, we can both represent periodic infinite mpos and place dependent finite mpos.

## MPOHamiltonian

We represent all quantum hamiltonians in their MPO form. This consists of an upper
triangular (sparse) matrix of regular MPO tensors per site, resulting in an array with 3
dimensions. Thus, an `MPOHamiltonian` is just a wrapper around `SparseMPO`, but with some
guarantees about its structure. As an example, the following bit of code constructs the
Ising hamiltonian.

```@setup mpohamiltonian
using TensorKit
using MPSKit
```

```@example mpohamiltonian
T = ComplexF64
X = TensorMap(T[0 1; 1 0], ℂ^2 ← ℂ^2)
Z = TensorMap(T[1 0; 0 -1], ℂ^2 ← ℂ^2)

data = Array{Any,3}(missing, 1, 3, 3)
data[1, 1, 1] = identity(ℂ^2)
data[1, 1, 1] = one(T) # regular numbers are interpreted as identity operators
data[1, 1, 2] = -Z
data[1, 2, 3] = Z
data[1, 1, 3] = 3 * X
H_Ising = MPOHamiltonian(data)
nothing # hide
```

When working with symmetries, it is often not possible to represent the entire Hamiltonian
as a sum of a product of one-body operators. This means that there will be auxiliary virtual
spaces connecting the different MPO tensors. For example, when constructing the XXX
Heisenberg model without symmetries, the following code suffices:

```@example mpohamiltonian
Y = TensorMap(T[0 -im; im 0], ℂ^2 ← ℂ^2)
data = Array{Any,3}(missing, 1, 5, 5)
data[1, 1, 1] = one(T)
data[1, end, end] = one(T)
data[1, 1, 2] = X
data[1, 2, end] = X
data[1, 1, 3] = Y
data[1, 3, end] = Y
data[1, 1, 4] = Z
data[1, 4, end] = Z
H_Heisenberg = MPOHamiltonian(data)
nothing # hide
```

However, none of the operators above are SU(2) symmetric, only the total hamiltonian is. The
solution is found by combining `XX + YY + ZZ` into a single tensor, and then decomposing
that tensor into a product of local MPO tensors.

```@example mpohamiltonian
using MPSKit: decompose_localmpo, add_util_leg
SS = TensorMap(zeros, T, SU2Space(1//2 => 1)^2 ← SU2Space(1//2 => 1)^2)
blocks(SS)[SU2Irrep(0)] .= -3/4
blocks(SS)[SU2Irrep(1)] .= 1/4
S_left, S_right = MPSKit.decompose_localmpo(add_util_leg(SS))
@show space(S_right, 1) # this is the virtual space connecting the two mpo tensors
data = Array{Any,3}(missing, 1, 3, 3)
data[1, 1, 1] = one(T)
data[1, end, end] = one(T)
data[1, 1, 2] = S_left
data[1, 2, end] = S_right
H_Heisenberg_SU2 = MPOHamiltonian(data)
nothing # hide
```

because of this, when indexing a Hamiltonian, a 2,2-tensormap is returned:

```@example mpohamiltonian
H_Heisenberg[1][1, 2]
```
