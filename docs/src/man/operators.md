# [Operators](@id um_operators)

In analogy to how we can define matrix product states as a contraction of local tensors, a
similar construction exist for operators. To that end, a Matrix Product Operator (MPO) is
nothing more than a collection of local [`MPOTensor`](@ref) objects, contracted along a
line. Again, we can distinguish between finite and infinite operators, with the latter being
represented by a periodic array of MPO tensors.

## FiniteMPO

Starting off with the simplest case, a basic finite MPO is a vector of `MPOTensor` objects.
These objects can be created either directly from a vector of `MPOTensor`s, or starting from
a dense operator (a subtype of `AbstractTensorMap{S,N,N}`), which is then decomposed into a
product of local tensors.

<!-- TODO: insert picture -->

```@setup operators
using TensorKit, MPSKit
```

```@example operators
S_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
S_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
O_xzx = FiniteMPO(S_x ⊗ S_x ⊗ S_x);
```

The individual tensors are accessible via regular indexing. Note that the tensors are
internally converted to the `MPOTensor` objects, thus having 4 indices. In this specific
case, the left- and right virtual spaces are trivial, but this is not a requirement.

```@example operators
O_xzx[1]
```

For convenience, a number of utility functions are defined for probing the structure of the
constructed MPO. For example, the spaces can be queried as follows:

```@example operators
left_virtualspace(O_xzx, 2)
right_virtualspace(O_xzx, 2)
physicalspace(O_xzx, 2)
```

MPOs also support a range of linear algebra operations, such as addition, subtraction and
multiplication, either among themselves or with a finite MPS. Here, it is important to note
that these operations will increase the virtual dimension of the resulting MPO or MPS, and
this naive application is thus typically not optimal. For approximate operations that do not
increase the virtual dimension, the more advanced algorithms in the [um_algorithms](@ref)
sections should be used.

```@example operators
O_xzx² = O_xzx * O_xzx
println("Virtual dimension of O_xzx²: ", left_virtualspace(O_xzx², 2))
O_xzx_sum = 0.1 * O_xzx + O_xzx²
println("Virtual dimension of O_xzx_sum: ", left_virtualspace(O_xzx_sum, 2))
```

```@example operators
O_xzx * FiniteMPS(3, ℂ^2, ℂ^4)
```

!!! note
    The virtual spaces of the resulting MPOs typically grow exponentially with the
    number of multiplications. Nevertheless, a number of optimizations are in place that
    make sure that the virtual spaces do not increase past the maximal virtual space that
    is dictated by the requirement of being full-rank tensors.

## MPOHamiltonian

We can also represent quantum Hamiltonians in the same form. This is done by converting a
sum of local operators into a single MPO operator. The resulting operator has a very
specific structure, and is often referred to as a *Jordan block MPO*.

This object can be constructed as an MPO by using the [`MPOHamiltonian`](@ref) constructor,
which takes two crucial pieces of information:

1. An array of `VectorSpace` objects, which determines the local Hilbert spaces of the
   system. The resulting MPO will snake through the array in linear indexing order.

2. A set of local operators, which are characterised by a number of indices that specify on
   which sites the operator acts, along with an operator to define the action. These are
   specified as a `inds => operator` pairs, or any other iterable collection thereof. The
   `inds` should be tuples of valid indices for the array of `VectorSpace` objects, or a
   single integer for single-site operators.

As a concrete example, we consider the
[Transverse-field Ising model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model)
defined by the Hamiltonian

```math
H = -J \sum_{\langle i, j \rangle} X_i X_j - h \sum_j Z_j
```

```@example operators
J = 1.0
h = 0.5
chain = fill(ℂ^2, 4) # a finite chain of 4 sites, each with a 2-dimensional Hilbert space
single_site_operators = [1 => -h * S_z, 2 => -h * S_z, 3 => -h * S_z]
two_site_operators = [(1, 2) => -J * S_x ⊗ S_x, (2, 3) => -J * S_x ⊗ S_x]
H_ising = MPOHamiltonian(chain, single_site_operators..., two_site_operators...)
```

Various alternative constructions are possible, such as using a `Dict` with key-value pairs
that specify the operators, or using generator expressions to simplify the construction.

```@example operators
H_ising′ = -J * MPOHamiltonian(chain,
                               (i, i + 1) => S_x ⊗ S_x for i in 1:(length(chain) - 1)) -
            h * MPOHamiltonian(chain, i => S_z for i in 1:length(chain))
H_ising ≈ H_ising′
```

Note that this construction is not limited to nearest-neighbour interactions, or 1D systems.
In particular, it is possible to construct quasi-1D realisations of 2D systems, by using
different arrays of `VectorSpace` objects. For example, the 2D Ising model on a square
lattice can be constructed as follows:

```@example operators
square = fill(ℂ^2, 4, 4) # a 4x4 square lattice
operators = Dict()

local_operators = Dict()
for I in eachindex(square)
    local_operators[(I,)] = -h * S_z # single site operators still require tuples of indices
end

horizontal_operators = Dict()
for I in eachindex(square)
    if I[1] < size(square, 1)
        horizontal_operators[(I, I + (1, 0))] = -J * S_x ⊗ S_x
    end
end

vertical_operators = Dict()
for I in eachindex(square)
    if I[2] < size(square, 2)
        vertical_operators[(I, I + (0, 1))] = -J * S_x ⊗ S_x
    end
end

H_ising_2d = MPOHamiltonian(square, local_operators) +
    MPOHamiltonian(square, horizontal_operators) +
    MPOHamiltonian(square, vertical_operators)
```

There are various utility functions available for constructing more advanced lattices, for
which the [lattices](@ref) section should be consulted.

### Expert mode

The `MPOHamiltonian` constructor is in fact an automated way of constructing the
aforementioned *Jordan block MPO*. In its most general form, the matrix $W$ takes on the
form of the following block matrix:

```math
\begin{pmatrix}
1 & C & D \\
0 & A & B \\
0 & 0 & 1
\end{pmatrix}
```

which generates all single-site local operators $D$, all two-site operators $CB$, three-site
operators $CAB$, and so on. Additionally, this machinery can also be used to construct
interaction that are of (exponentially decaying) infinite range, and to approximate
power-law interactions.

In order to illustrate this, consider the following explicit example of the Transverse-field
Ising model:

```math
W = \begin{pmatrix}
1 & X & -hZ \\ 
0 & 0 & -JX \\
0 & 0 & 1
\end{pmatrix}
```

If we add in the left and right boundary vectors

```math
v_L = \begin{pmatrix}
1 & 0 & 0
\end{pmatrix}
, \qquad 
v_R = \begin{pmatrix}
0 \\ 0 \\ 1
\end{pmatrix}
```

One can easily check that the Hamiltonian on $N$ sites is given by the contraction

```math
H = V_L W^{\otimes N} V_R
```

We can even verify this symbolically:

```@example operators
using Symbolics
L = 4
# generate W matrices
@variables A[1:L] B[1:L] C[1:L] D[1:L]
Ws = map(1:L) do l
    return [1 C[l] D[l]
            0 A[l] B[l]
            0 0    1]
end

# generate boundary vectors
Vₗ = [1, 0, 0]'
Vᵣ = [0, 0, 1]

# expand the MPO
expand(Vₗ * prod(Ws) * Vᵣ)
```

The `MPOHamiltonian` constructor can also be used to construct the operator from this most
general form, by supplying a 3-dimensional array $W$ to the constructor. Here, the first
dimension specifies the site in the unit cell, the second dimension specifies the row of the
matrix, and the third dimension specifies the column of the matrix.

!!! note
    While the above example can be constructed from building blocks that are strictly local
    operators, this is not always the case, especially when symmetries are involved. In
    those cases, the elements of the matrix $W$ have additional virtual legs that are
    contracted between different sites.

### Working with `MPOHamiltonian` objects

!!! warning
    This part is still a work in progress

Because of the discussion above, the `MPOHamiltonian` object is in fact just a `FiniteMPO`,
with some additional structure. This means that similar operations and properties are
available, such as the virtual spaces, or the individual tensors. However, the block
structure of the operator means that now the virtual spaces are not just a single space, but
a collection (direct sum) of spaces, one for each row/column.

<!-- TODO: add examples virtualspace once blocktensors are in place -->

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
