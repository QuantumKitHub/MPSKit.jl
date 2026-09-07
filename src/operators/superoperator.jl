# Superoperators acting on Matrix Product Density Operators
# =========================================================
# An MPDO is an MPS whose site tensors carry two physical legs, `A[vl p p̄; vr]`, with `p`
# the ket leg and `p̄` the bra leg, so that an operator can act on either side of `ρ`:
#
#   left multiplication  `ρ ↦ Oρ` acts on `p`  -- already supported throughout MPSKit
#   right multiplication `ρ ↦ ρO` acts on `p̄`  -- provided here, selected by `BraSide`
#
# The two differ in where the MPO virtual leg is braided through the other physical leg,
# and in that the bra leg is dual, so that contracting an operator with it transposes that
# operator. Both are handled by the `BraSide` methods of `transfer_left`/`transfer_right`
# and of the derivative operators.

"""
    struct BraSide{O}

Wrapper marking an MPO tensor `O` as acting on the *second* (bra) physical leg of an MPDO
tensor `A[vl p p̄; vr]`, rather than on the first (ket) leg. Since that leg is dual, this
transposes `O`, so a `BraSide` tensor implements right multiplication `ρ ↦ ρO`.

This is purely a dispatch marker: it carries no data beyond the wrapped tensor, and exists
so that [`transfer_left`](@ref), [`transfer_right`](@ref) and the derivative operators can
select the bra-side contraction. Indexing forwards to the wrapped tensor and re-wraps the
result, so slicing the Jordan structure of an [`MPOHamiltonian`](@ref) preserves the mark.

See also: [`BraMPO`](@ref), [`SuperOperator`](@ref)
"""
struct BraSide{O}
    O::O
end

@inline Base.getindex(W::BraSide, inds...) = BraSide(getindex(W.O, inds...))
physicalspace(W::BraSide) = physicalspace(W.O)

"""
    _bra_transpose(O::BraSide)

The wrapped tensor transposed onto the dual space that the bra leg of an MPDO lives in.

Note that `conj` is not a substitute: it agrees with the transpose only for Hermitian `O`,
and it reverses the tensor's cyclic leg orientation, so that no placement of the braiding
tensors closes the surrounding contraction back into a planar diagram.
"""
_bra_transpose(O::BraSide) = transpose(O.O, ((1, 3), (2, 4)))

"""
    struct BraMPO{T}
    BraMPO(O::AbstractMPO)

An MPO reinterpreted as acting on the bra (second physical) leg of an MPDO, i.e. by right
multiplication `ρ ↦ ρO`. Indexing yields [`BraSide`](@ref)-wrapped tensors; the remainder
of the `AbstractMPO` interface is forwarded to the parent.

Deliberately not an `AbstractMPO` subtype: the generic `AbstractMPO` linear algebra (`+`,
`*`, `fuse_mul_mpo`) is not meaningful for a marker type, and subtyping gains no dispatch
that is needed here.

See also: [`BraSide`](@ref), [`SuperOperator`](@ref)
"""
struct BraMPO{T}
    op::T
end

Base.parent(H::BraMPO) = H.op
Base.length(H::BraMPO) = length(H.op)
@inline Base.getindex(H::BraMPO, i::Int) = BraSide(H.op[i])
@inline Base.getindex(H::BraMPO, inds) = map(BraSide, H.op[inds])
Base.iterate(H::BraMPO, args...) = _wrap_braside(iterate(H.op, args...))
Base.broadcastable(H::BraMPO) = map(BraSide, H.op)
_wrap_braside(::Nothing) = nothing
_wrap_braside((W, state)::Tuple) = (BraSide(W), state)

left_virtualspace(H::BraMPO, site::Int) = left_virtualspace(H.op, site)
right_virtualspace(H::BraMPO, site::Int) = right_virtualspace(H.op, site)
physicalspace(H::BraMPO, site::Int) = physicalspace(H.op, site)

Base.eltype(::Type{BraMPO{T}}) where {T} = eltype(T)
VectorInterface.scalartype(::Type{BraMPO{T}}) where {T} = scalartype(T)
TensorKit.spacetype(::Type{BraMPO{T}}) where {T} = spacetype(T)
TensorKit.spacetype(H::BraMPO) = spacetype(typeof(H))

isidentitylevel(H::BraMPO, i::Int) = isidentitylevel(H.op, i)
isemptylevel(H::BraMPO, i::Int) = isemptylevel(H.op, i)

Base.:-(H::BraMPO) = BraMPO(-H.op)

# A `BraMPO` is structurally identical to its parent -- it differs only in which physical
# leg it contracts with -- so it reaches exactly the same environment algorithms. These
# unions widen the relevant signatures in one place.
const AbstractMPOLike = Union{AbstractMPO, BraMPO}
const FiniteMPOLike = Union{FiniteMPO, BraMPO{<:FiniteMPO}}
const FiniteMPOHamiltonianLike = Union{FiniteMPOHamiltonian, BraMPO{<:FiniteMPOHamiltonian}}
const FiniteOperatorLike = Union{FiniteMPOLike, FiniteMPOHamiltonianLike}
const InfiniteMPOLike = Union{InfiniteMPO, BraMPO{<:InfiniteMPO}}
const InfiniteMPOHamiltonianLike = Union{InfiniteMPOHamiltonian, BraMPO{<:InfiniteMPOHamiltonian}}
const InfiniteOperatorLike = Union{InfiniteMPOLike, InfiniteMPOHamiltonianLike}

"""
    struct SuperOperator{TL, TR}
    SuperOperator(left, right)

Superoperator `ρ ↦ left * ρ + ρ * right` acting on a Matrix Product Density Operator, i.e.
on an MPS whose site tensors carry two physical legs `A[vl p p̄; vr]`. `left` acts on the
ket leg and `right`, wrapped in a [`BraMPO`](@ref), on the bra leg.

The two terms are kept as an independent pair of MPO sandwiches, so that environments,
effective Hamiltonians and time evolution all reduce to the existing [`LazySum`](@ref)
machinery.

For Hamiltonian dynamics take `right = -left`: `SuperOperator(H, -H)` is the commutator
`ρ ↦ [H, ρ]`, so that `timestep(ρ, SuperOperator(H, -H), t, dt, TDVP())` realises
`ρ ↦ exp(-iHdt) ρ exp(+iHdt)`. No transpose is needed on `right`; the bra leg is dual, so
contracting an operator with it transposes that operator already.

# Fields

$(TYPEDFIELDS)

See also: [`BraSide`](@ref), [`BraMPO`](@ref)
"""
struct SuperOperator{TL, TR}
    "operator acting on the ket (first physical) leg, by left multiplication"
    left::TL
    "operator acting on the bra (second physical) leg, by right multiplication"
    right::BraMPO{TR}
end

# the `(left, ::BraMPO)` method is the constructor generated for the struct itself
SuperOperator(left, right) = SuperOperator(left, BraMPO(right))

# the lazy-sum view that every downstream interface is defined in terms of
_lazysum(O::SuperOperator) = LazySum([O.left, O.right])
