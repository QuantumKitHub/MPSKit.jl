```@meta
DocTestSetup = quote
    using MPSKit, TensorKit
end
```

# [TensorKit for MPS users](@id concept_vector_spaces)

Every tensor in MPSKit is a TensorKit [`TensorMap`](https://quantumkithub.github.io/TensorKit.jl/stable/), and this single choice is what makes the library generic over symmetry.
The same MPS and MPO code runs for a plain complex vector space, an abelian symmetry such as ℤ₂ or U(1), a non-abelian symmetry such as SU(2), and for fermionic or anyonic systems, because the symmetry lives inside the tensor rather than in the algorithms.
This page builds the mental model you need to read that API comfortably: what a `TensorMap` is, how its indices are typed by vector *spaces*, and the index conventions MPSKit adopts for its state and operator tensors.
It is about understanding rather than construction: once the tensors introduced here feel familiar they are put to work in [Matrix product states](@ref concept_matrix_product_states) and [Operators and Hamiltonians](@ref concept_operators_and_hamiltonians).

## Tensors as linear maps

The mental shift from a multi-dimensional array to a `TensorMap` is small but important.
An array is a bag of numbers indexed by integer sizes; a `TensorMap` is a **linear map** from one space to another, and its indices carry *types* — vector spaces — rather than bare sizes.
The legs are partitioned into a **codomain** (the outputs of the map) and a **domain** (its inputs), so that a tensor with codomain `W` and domain `V` is read as a map `W ← V`.

Throughout this page we use a single running example: the two-dimensional complex space of a spin-1/2 degree of freedom.
It is written `ℂ^2` (the `ℂ` is typed `\bbC`), which constructs a [`ComplexSpace`](https://quantumkithub.github.io/TensorKit.jl/stable/) of dimension two.

```jldoctest vspace
julia> V = ℂ^2
ℂ^2

julia> dim(V)
2
```

A space knows its dimension, and that dimension — not a Julia integer — is what a tensor's legs are built from.

## Building tensors

Constructing a tensor mirrors constructing an array, with the `axes`/`size` specifiers replaced by spaces.
The two most common constructors are `rand` and `zeros`, which take a scalar type followed by the codomain and domain.
The domain and codomain can be passed as two separate arguments, or joined with the `←` arrow (typed `\leftarrow`):

```jldoctest vspace
julia> t = rand(Float64, V ⊗ V, V);

julia> space(t)
(ℂ^2 ⊗ ℂ^2) ← ℂ^2

julia> codomain(t)
(ℂ^2 ⊗ ℂ^2)
```

Here `t` is a map from one spin-1/2 space to two of them, built with `⊗` (typed `\otimes`) to combine spaces.
Querying [`space`](https://quantumkithub.github.io/TensorKit.jl/stable/), [`codomain`](https://quantumkithub.github.io/TensorKit.jl/stable/), and [`domain`](https://quantumkithub.github.io/TensorKit.jl/stable/) always prints deterministically, even though the tensor's entries are random.
The `zeros` form takes the codomain and domain as separate positional arguments:

```jldoctest vspace
julia> z = zeros(ComplexF64, V, V);

julia> space(z)
ℂ^2 ← ℂ^2
```

## Symmetric tensors: the payoff

The reason for all of this typing of indices is that the very same interface represents *symmetric* tensors, at no extra cost to the code that uses them.
Instead of `ℂ^2` we hand the constructor a space that has been split into charge **sectors**.
For a ℤ₂ symmetry, `Z2Space(0 => 1, 1 => 1)` is a two-dimensional space whose dimension is distributed as one dimension in the even (charge `0`) sector and one in the odd (charge `1`) sector:

```jldoctest vspace
julia> V2 = Z2Space(0 => 1, 1 => 1)
Rep[ℤ₂](…) of dim 2:
 0 => 1
 1 => 1

julia> dim(V2)
2
```

A tensor built on this space is a genuinely block-sparse, symmetry-respecting object, yet it is constructed and queried exactly like the plain one above:

```jldoctest vspace
julia> t3 = rand(Float64, V2 ⊗ V2, V2);

julia> space(t3)
(Rep[ℤ₂](0 => 1, 1 => 1) ⊗ Rep[ℤ₂](0 => 1, 1 => 1)) ← Rep[ℤ₂](0 => 1, 1 => 1)
```

Only the space changed; the tensor stores just the symmetry-allowed blocks and enforces charge conservation for you.
Swapping `Z2Space` for a U(1), SU(2), or fermionic space would be an equally local change, and this is exactly why MPSKit's algorithms never mention a symmetry: it is carried entirely by the spaces.
See [Using symmetries](@ref tutorial_using_symmetries) for the full progression of symmetry types.

## Reading a partition error

One feature of `TensorMap`s has no counterpart in plain arrays and is worth meeting deliberately, because it produces an error message that is puzzling the first time.
The partition of legs into codomain and domain is *part of a tensor's type*: two tensors are compatible for addition only when their codomains and domains match, arrows included.
Take a tensor and re-partition it so that every leg sits in the codomain (moving a leg across the `←` also flips its arrow to the dual space):

```@example vspace
using MPSKit, TensorKit # hide
V = ℂ^2 # hide
t = rand(Float64, V ⊗ V, V)
t2 = permute(t, ((1, 2, 3), ()))
space(t), space(t2)
```

`t` and `t2` describe the same legs but with different partitions, so adding them directly fails:

```@example vspace
try #hide
t + t2 # partitions do not match
catch err; Base.showerror(stderr, err); end #hide
```

<!-- REVIEW: the exact wording of this error message is TensorKit-version-dependent and shown only for illustration; the point is that the mismatch is reported in terms of the domain/codomain, not the shape. -->

The fix is [`permute`](https://quantumkithub.github.io/TensorKit.jl/stable/), which regroups the legs into a chosen partition.
Bringing `t2` back to the partition of `t` makes the addition well-defined again:

```@example vspace
space(t + permute(t2, ((1, 2), (3,))))
```

The lesson is not to avoid re-partitioning but to read such an error as "the same legs, grouped differently" and reach for `permute`.

## MPSKit's index conventions

With the `TensorMap` model in hand, we can state the leg conventions MPSKit uses for its own tensors — and, more importantly, *why* it uses them.

An MPS site tensor has a left virtual space `Vₗ`, one or more physical spaces `P`, and a right virtual space `Vᵣ`, and MPSKit orders them so that the left virtual and physical legs form the codomain while the right virtual leg forms the domain, i.e. `Vₗ ⊗ P ← Vᵣ` (an MPO tensor, with an incoming and an outgoing physical leg, reads `Vₗ ⊗ P ← P ⊗ Vᵣ`).
At first glance this ordering looks arbitrary, but it is chosen to keep the tensor networks **planar**: the legs run left-to-right without any lines having to cross.
Planarity is what lets the algorithms be written without spurious crossings, and this matters most for **fermionic systems**, where every extra line crossing carries a sign and unnecessary crossings would reintroduce a sign problem.

<!-- REVIEW: the claim that this leg ordering keeps the networks planar, and that avoiding line crossings is what prevents fermionic sign problems, is a physics/design rationale carried over from the legacy man/intro.md — please confirm the reasoning is stated correctly. -->

### The MPS tensor

```@raw html
<img src="../assets/mps_tensor_definition.png" alt="An MPS tensor drawn as a box: a left virtual leg and one or more physical legs on the left, a right virtual leg on the right." class="color-invertible"/>
```

The diagram encodes the ordering

```math
V_\ell \otimes P_1 \otimes \cdots \otimes P_{k} \leftarrow V_r,
```

i.e. leg 1 is the left virtual space, the physical spaces come next (the picture labels them `physical (2:N-1)`), and the final leg is the right virtual space.
<!-- REVIEW: leg-order fallback read from mps_tensor_definition.png (left virtual as leg 1 in the codomain, physical legs 2..N-1, right virtual as the last leg in the domain); confirm against the intended convention. -->

Crucially, an MPS tensor may carry an **arbitrary number of physical legs**, and both [`FiniteMPS`](@ref) and [`InfiniteMPS`](@ref) handle the resulting objects.
This is what allows, for example, boundary tensors in PEPS code, which carry two physical legs.

### The bond tensor

```@raw html
<img src="../assets/bond_tensor_definition.png" alt="A bond tensor drawn as a box with one virtual leg on the left and one virtual leg on the right." class="color-invertible"/>
```

A bond tensor sits between two MPS site tensors and has only the two virtual legs, ordered

```math
V_\ell \leftarrow V_r,
```

i.e. the left virtual space is the codomain and the right virtual space is the domain.
<!-- REVIEW: leg-order fallback read from bond_tensor_definition.png (virtual leg 1 on the left in the codomain, virtual leg 2 on the right in the domain); confirm against the intended convention. -->

### The MPO tensor

```@raw html
<img src="../assets/mpo_tensor_definition.png" alt="An MPO tensor drawn as a box: a left virtual leg and an outgoing physical leg on the left, an incoming physical leg and a right virtual leg on the right." class="color-invertible"/>
```

An MPO tensor, used to represent both quantum Hamiltonians and classical statistical-mechanics problems, carries two physical legs (one outgoing, one incoming) and two virtual legs, ordered

```math
V_\ell \otimes P \leftarrow P \otimes V_r.
```

The picture labels these `virtual (1)` and `physical (2)` in the codomain, and `physical (3)` and `virtual (4)` in the domain.
<!-- REVIEW: leg-order fallback read from mpo_tensor_definition.png (leg 1 left virtual, leg 2 outgoing physical, leg 3 incoming physical, leg 4 right virtual); confirm against the intended convention. -->

## Where to go next

- To see these tensors assembled into states and gauged into canonical form, read [Matrix product states](@ref concept_matrix_product_states).
- For the operator side and the Jordan-block structure of Hamiltonians, read [Operators and Hamiltonians](@ref concept_operators_and_hamiltonians).
- For the full range of symmetry types and when each pays off, see [Using symmetries](@ref tutorial_using_symmetries).
- For the underlying tensor library, consult the [TensorKit documentation](https://quantumkithub.github.io/TensorKit.jl/stable/).
```
