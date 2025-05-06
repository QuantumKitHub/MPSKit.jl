# [Operators](@id um_operators)

In analogy to how we can define matrix product states as a contraction of local tensors, a
similar construction exist for operators. To that end, a Matrix Product Operator (MPO) is
nothing more than a collection of local [`MPOTensor`](@ref MPSKit.MPOTensor) objects, contracted along a
line. Again, we can distinguish between finite and infinite operators, with the latter being
represented by a periodic array of MPO tensors.

## FiniteMPO

Starting off with the simplest case, a basic [`FiniteMPO`](@ref) is a vector of `MPOTensor` objects.
These objects can be created either directly from a vector of `MPOTensor`s, or starting from
a dense operator (a subtype of `AbstractTensorMap`), which is then decomposed into a
product of local tensors.

![](../assets/mpo.svg)

```@setup operators
using TensorKit, MPSKit, MPSKitModels
```

```@example operators
S_x = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
S_z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)
O_xzx = FiniteMPO(S_x ⊗ S_z ⊗ S_x);
```

The individual tensors are accessible via regular indexing. Note that the tensors are
internally converted to the `MPOTensor` objects, thus having four indices. In this specific
case, the left- and right virtual spaces are trivial, but this is not a requirement.

```@example operators
O_xzx[1]
```

!!! warning
    The local tensors are defined only up to a gauge transformation of the virtual spaces.
    This means that the tensors are not uniquely defined, and special care must be taken
    when comparing MPOs on an element-wise basis.

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
O_xzx_sum * FiniteMPS(3, ℂ^2, ℂ^4)
```

!!! note
    The virtual spaces of the resulting MPOs typically grow exponentially with the
    number of multiplications. Nevertheless, a number of optimizations are in place that
    make sure that the virtual spaces do not increase past the maximal virtual space that
    is dictated by the requirement of being full-rank tensors.

## InfiniteMPO

This construction can again be extended to the infinite case, where the tensors are repeated periodically.
Therefore, an [`InfiniteMPO`](@ref) is simply a `PeriodicVector` of `MPOTensor` objects.
These can only be constructed from vectors of `MPOTensor`s, since it is impossible to create the infinite operators directly.

```@example operators
mpo = InfiniteMPO(O_xzx[1:2])
```

Otherwise, their behavior is mostly similar to that of their finite counterparts.

## FiniteMPOHamiltonian

We can also represent quantum Hamiltonians in the same form. This is done by converting a
sum of local operators into a single MPO operator. The resulting operator has a very
specific structure, and is often referred to as a *Jordan block MPO*.

This object can be constructed as an MPO by using the [`FiniteMPOHamiltonian`](@ref) constructor,
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
chain = fill(ℂ^2, 3) # a finite chain of 4 sites, each with a 2-dimensional Hilbert space
single_site_operators = [1 => -h * S_z, 2 => -h * S_z, 3 => -h * S_z]
two_site_operators = [(1, 2) => -J * S_x ⊗ S_x, (2, 3) => -J * S_x ⊗ S_x]
H_ising = FiniteMPOHamiltonian(chain, single_site_operators..., two_site_operators...)
```

Various alternative constructions are possible, such as using a `Dict` with key-value pairs
that specify the operators, or using generator expressions to simplify the construction.

```@example operators
H_ising′ = -J * FiniteMPOHamiltonian(chain,
                               (i, i + 1) => S_x ⊗ S_x for i in 1:(length(chain) - 1)) -
            h * FiniteMPOHamiltonian(chain, i => S_z for i in 1:length(chain))
isapprox(H_ising, H_ising′; atol=1e-6)
```

Note that this construction is not limited to nearest-neighbour interactions, or 1D systems.
In particular, it is possible to construct quasi-1D realisations of 2D systems, by using
different arrays of [`VectorSpace`](@extref TensorKit.VectorSpace) objects.
For example, the 2D Ising model on a square lattice can be constructed as follows:

```@example operators
square = fill(ℂ^2, 3, 3) # a 3x3 square lattice
operators = Dict()

local_operators = Dict()
for I in eachindex(square)
    local_operators[(I,)] = -h * S_z # single site operators still require tuples of indices
end

# horizontal and vertical interactions are easier using Cartesian indices
horizontal_operators = Dict()
I_horizontal = CartesianIndex(0, 1)
for I in eachindex(IndexCartesian(), square)
    if I[2] < size(square, 2)
        horizontal_operators[(I, I + I_horizontal)] = -J * S_x ⊗ S_x
    end
end

vertical_operators = Dict()
I_vertical = CartesianIndex(1, 0)
for I in eachindex(IndexCartesian(), square)
    if I[1] < size(square, 1)
        vertical_operators[(I, I + I_vertical)] = -J * S_x ⊗ S_x
    end
end

H_ising_2d = FiniteMPOHamiltonian(square, local_operators) +
    FiniteMPOHamiltonian(square, horizontal_operators) +
    FiniteMPOHamiltonian(square, vertical_operators);
```

There are various utility functions available for constructing more advanced lattices, for
which the [lattices](@ref) section should be consulted.

## InfiniteMPOHamiltonian

Again, this construction can be extended straightforwardly to the infinite case.
To that end, we simply need to specify all interactions per unit cell.
In particular, an [`InfiniteMPOHamiltonian`](@ref) for the Ising model is obtained via

```@example operators
J = 1.0
h = 0.5
infinite_chain = PeriodicVector([ℂ^2]) # an infinite chain of a local 2-dimensional Hilbert space
H_ising_infinite = InfiniteMPOHamiltonian(infinite_chain, 1 => -h * S_z, (1, 2) => -J * S_x ⊗ S_x)
```

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

The [`FiniteMPOHamiltonian`](@ref) constructor can also be used to construct the operator from this most
general form, by supplying a vector of [`BlockTensorMap`](@extref BlockTensorKit.BlockTensorMap) objects
to the constructor. Here, the vector specifies the sites in the unit cell, while the blocktensors contain
the rows and columns of the matrix. We can verify this explicitly:

```@example operators
H_ising[2] # print the blocktensor
```

### Working with `MPOHamiltonian` objects

!!! warning
    This part is still a work in progress

Because of the discussion above, the `FiniteMPOHamiltonian` object is in fact just an `AbstractMPO`,
with some additional structure. This means that similar operations and properties are
available, such as the virtual spaces, or the individual tensors. However, the block
structure of the operator means that now the virtual spaces are not just a single space, but
a collection (direct sum) of spaces, one for each row/column.

```@example operators
left_virtualspace(H_ising, 1), right_virtualspace(H_ising, 1), physicalspace(H_ising, 1)
```
