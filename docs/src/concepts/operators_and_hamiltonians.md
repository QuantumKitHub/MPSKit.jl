# [Operators and Hamiltonians](@id concept_operators_and_hamiltonians)

Just as a matrix product state factorises a wavefunction into a chain of local tensors, an operator on a one-dimensional system can be factorised in exactly the same way.
The result is a *matrix product operator* (MPO): the operator analogue of an MPS.
This page explains what that factorisation is, why the local tensors are not unique, and — above all — the particular upper-triangular *Jordan-block* structure that lets a sum of local terms be written as a single MPO.
It is about understanding rather than construction: for how to *build* a Hamiltonian see [Building Hamiltonians](@ref howto_hamiltonians), and for the full type signatures see the [Operators](@ref lib_operators) reference.

## What an MPO is

An MPO is a collection of local [`MPOTensor`](@ref MPSKit.MPOTensor) objects contracted along a line.
Where an MPS site tensor has one physical leg and two virtual legs, an MPO site tensor has *two* physical legs — one incoming and one outgoing — because it maps states to states, and again two virtual legs that thread the operator together along the chain.

```@raw html
<img src="../assets/mpo.svg" alt="An MPO drawn as a chain of tensors, each with an incoming and an outgoing physical leg and virtual legs joining it to its neighbours." width="50%" class="color-invertible"/>
```

*The diagram shows an MPO as a row of site tensors, each carrying a pair of physical indices (one in, one out) and linked to its neighbours through virtual bonds.*

As with states, the construction comes in a finite and an infinite flavour.
A [`FiniteMPO`](@ref) is a plain vector of `MPOTensor` objects with trivial (dimension-one) virtual spaces at the two ends, so that the network describes a genuine operator on a finite chain.
An [`InfiniteMPO`](@ref) instead repeats a finite unit cell periodically, and is therefore stored as a periodic array of `MPOTensor` objects rather than an ordinary vector.

### Gauge non-uniqueness

The local tensors of an MPO are not uniquely determined by the operator they encode.
Exactly as for an MPS, an invertible gauge transformation can be inserted on any virtual bond and reabsorbed into the two neighbouring tensors without changing the contracted network.
The individual site tensors are therefore defined only up to this virtual-space gauge freedom.

!!! warning "Element-wise comparison is unsafe"
    Because two different sets of local tensors can represent the very same operator, comparing MPOs tensor-by-tensor is not meaningful.
    Test for equality through gauge-invariant quantities instead.

### Products and sums grow the virtual dimension

MPOs support the usual linear-algebra operations — addition, subtraction, and multiplication, either among themselves or acting on an MPS.
Each such operation combines the virtual spaces of its operands, so the virtual dimension of the result is (generically) the *product* or *sum* of the input dimensions rather than staying fixed.
Composing operators naively therefore makes the representation grow, and the growth compounds under repeated multiplication.This growth is precisely what motivates the *approximate* algorithms that re-express a product or sum within a bounded virtual dimension; see the [algorithm landscape](@ref concept_algorithm_landscape) for where those methods fit.

## MPO Hamiltonians and the Jordan-block form

A quantum Hamiltonian is a *sum* of local terms rather than a single dense operator, yet it too can be written as one MPO.
The trick is a characteristic upper-triangular block structure, so distinctive that the resulting object is usually called a *Jordan-block MPO*.
In MPSKit this is the [`MPOHamiltonian`](@ref) family — [`FiniteMPOHamiltonian`](@ref) and [`InfiniteMPOHamiltonian`](@ref) — and it is what the Hamiltonian constructors assemble under the hood.

In its most general form, the per-site block matrix ``W`` reads

```math
W = \begin{pmatrix}
1 & C & D \\
0 & A & B \\
0 & 0 & 1
\end{pmatrix}
```

where the corner entries `1` are identity operators and ``A``, ``B``, ``C``, ``D`` are (blocks of) local operators.
The Hamiltonian on ``N`` sites is recovered by contracting one copy of ``W`` per site between two boundary vectors,

```math
v_L = \begin{pmatrix} 1 & 0 & 0 \end{pmatrix},
\qquad
v_R = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix},
\qquad
H = V_L\, W^{\otimes N}\, V_R .
```

### A finite-state automaton

The upper-triangular shape makes ``W`` behave like a finite-state automaton that reads the chain from left to right.
The boundary vector ``v_L`` starts in the top-left "identity" state; the automaton may stay there (the leading `1`), it may *finish* immediately by placing a single-site term through ``D``, or it may *start* an interaction through ``C``, propagate it across intermediate sites through ``A``, and *close* it through ``B`` into the bottom-right "identity" state selected by ``v_R``.
Every complete left-to-right path through the block matrix contributes one term of the Hamiltonian:

- ``D`` alone generates the single-site terms,
- ``C \cdot B`` generates the two-site terms,
- ``C \cdot A \cdot B`` generates the three-site terms,
- and in general ``C \cdot A^{k} \cdot B`` generates a term spanning ``k+2`` sites.

The repeated ``A`` block is what makes longer-range interactions possible at fixed virtual dimension: choosing ``A`` to be (a multiple of) the identity gives every additional site the same weight, while a decaying ``A`` gives geometrically decaying couplings.
A sum of such geometric series can approximate a power-law interaction to any desired accuracy, which is how (exponentially decaying) infinite-range and approximate power-law couplings are represented.
<!-- REVIEW: the statement that A = c·(identity) yields exponentially/geometrically decaying couplings and that sums of geometric decays approximate power laws is a physics claim taken from the legacy page; the maintainer should confirm the precise decay law and the power-law fitting scheme. -->

### The transverse-field Ising Hamiltonian

For the [transverse-field Ising model](https://en.wikipedia.org/wiki/Transverse-field_Ising_model),

```math
H = -J \sum_{\langle i, j \rangle} X_i X_j - h \sum_j Z_j ,
```

the block matrix specialises to

```math
W = \begin{pmatrix}
1 & X & -hZ \\
0 & 0 & -JX \\
0 & 0 & 1
\end{pmatrix} .
```

Here ``D = -hZ`` is the single-site field term, and the nearest-neighbour coupling ``-J X_i X_{i+1}`` is produced by ``C = X`` on one site meeting ``B = -JX`` on the next.
The middle block ``A = 0`` truncates the automaton after two sites, which is exactly what a nearest-neighbour model needs — there are no longer-range paths.

### Verifying the expansion symbolically

Because ``H = V_L\, W^{\otimes N}\, V_R`` is just repeated matrix multiplication, the term-generation rule above can be checked with a symbolic algebra system.
Filling ``W`` with abstract symbols ``A``, ``B``, ``C``, ``D`` on each site and expanding the product exposes exactly which combinations survive.

```@example operators
using Symbolics
L = 4
# generate W matrices, one per site
@variables A[1:L] B[1:L] C[1:L] D[1:L]
Ws = map(1:L) do l
    return [1 C[l] D[l]
            0 A[l] B[l]
            0 0    1]
end

# left and right boundary vectors
Vₗ = [1, 0, 0]'
Vᵣ = [0, 0, 1]

# expand the contraction H = V_L W^{⊗L} V_R
expand(Vₗ * prod(Ws) * Vᵣ)
```

Reading off the result, the lone ``D`` terms are the single-site contributions, the ``C \cdot B`` products are the two-site terms, the ``C \cdot A \cdot B`` products are the three-site terms, and so on — precisely the automaton paths described above.

## Sparse and block structure

Because an [`MPOHamiltonian`](@ref) is an MPO with the extra Jordan-block structure, its virtual space is not a single space but a *direct sum* of spaces, one for each row (or column) of the block matrix ``W``.
The site tensors are therefore stored as [`BlockTensorMap`](@extref BlockTensorKit.BlockTensorMap) objects rather than ordinary dense tensor maps, with each block occupying one cell of the ``W`` matrix.
MPSKit exposes the specialised [`JordanMPOTensor`](@ref) for exactly this layout.

!!! note "Sparsity is what keeps it efficient"
    Most cells of ``W`` are zero — the whole lower-left triangle, and typically much of the interior ``A`` block.
    Storing only the non-zero blocks is what makes the Jordan-block representation compact, so the cost tracks the number of distinct interaction terms rather than the nominal size of ``W``.

The [`JordanMPOTensor`](@ref) type and its internal accessors are implementation detail and may change; treat them as unstable and prefer the public constructors and `@ref`-documented interface.

## Beyond nearest-neighbour, 1D chains

The same machinery is not limited to nearest-neighbour couplings or to strictly one-dimensional systems: quasi-1D cylinders and 2D lattices are obtained by snaking the MPO through a multi-dimensional array of physical spaces, and longer-range interactions slot into the ``A`` block as described above.
See [Building Hamiltonians](@ref howto_hamiltonians) for the construction recipes and [MPSKitModels.jl](https://quantumkithub.github.io/MPSKitModels.jl/dev/) for ready-made lattices and models.

## Where to go next

- To build Hamiltonians from local terms, in 1D or on lattices, see [Building Hamiltonians](@ref howto_hamiltonians).
- For the full type signatures and docstrings, see the [Operators](@ref lib_operators) reference.
- For the approximate algorithms that keep MPO products and sums bounded, see the [algorithm landscape](@ref concept_algorithm_landscape).
