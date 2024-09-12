# Matrix Product Operators
# ========================
"""
    abstract type AbstractMPO{O<:MPOTensor} <: AbstractVector{O} end

Abstract supertype for Matrix Product Operators (MPOs).
"""
abstract type AbstractMPO{O<:MPOTensor} <: AbstractVector{O} end

# Hamiltonian Matrix Product Operators
# ====================================
"""
    abstract type AbstractHMPO{O<:MPOTensor} <: AbstractMPO{O}

Abstract supertype for Hamiltonian MPOs.
"""
abstract type AbstractHMPO{O<:MPOTensor} <: AbstractMPO{O} end

function HMPO(lattice::AbstractVector{S}, terms...) where {S<:VectorSpace}
    if lattice isa PeriodicArray
        return InfiniteHamiltonianMPO(lattice, terms...)
    else
        return FiniteHamiltonianMPO(lattice, terms...)
    end
end
function HMPO(operator::AbstractTensorMap{E,S,N,N}; L=Inf) where {E,S,N}
    @assert domain(operator) == codomain(operator) "Not a valid Hamiltonian operator."
    @assert allequal(collect(domain(operator))) "The operator must have the same local spaces."

    if isfinite(L)
        lattice = repeat([space(operator, 1)], L)
        return HMPO(lattice, ntuple(x -> x + i - 1, N) => operator for i in 1:(L - (N - 1)))
    else
        lattice = PeriodicArray([space(operator, 1)])
        return HMPO(lattice, ntuple(identity, N) => operator)
    end
end

# useful union types
const SparseMPO{O<:SparseBlockTensorMap} = AbstractMPO{O}

# By default, define things in terms of parent
Base.size(mpo::AbstractMPO, args...) = size(parent(mpo), args...)
Base.length(mpo::AbstractMPO) = length(parent(mpo))

@inline Base.getindex(mpo::AbstractMPO, args...) = getindex(parent(mpo), args...)
@inline function Base.setindex!(mpo::AbstractMPO, value::MPOTensor, i::Int)
    setindex!(parent(mpo), value, i)
    return mpo
end

# Properties
# ----------
left_virtualspace(mpo::AbstractMPO, site::Int) = left_virtualspace(mpo[site])
right_virtualspace(mpo::AbstractMPO, site::Int) = right_virtualspace(mpo[site])
physicalspace(mpo::AbstractMPO, site::Int) = physicalspace(mpo[site])
physicalspace(mpo::AbstractMPO) = map(physicalspace, mpo)

TensorKit.spacetype(::Union{AbstractMPO{O},Type{AbstractMPO{O}}}) where {O} = spacetype(O)
TensorKit.sectortype(::Union{AbstractMPO{O},Type{AbstractMPO{O}}}) where {O} = sectortype(O)
function TensorKit.storagetype(::Union{AbstractMPO{O},Type{AbstractMPO{O}}}) where {O}
    return storagetype(O)
end

# Utility functions
# -----------------
function jordanmpotensortype(::Type{S}, ::Type{T}) where {S<:VectorSpace,T<:Number}
    TT = Base.promote_typejoin(tensormaptype(S, 2, 2, T), BraidingTensor{T,S})
    return SparseBlockTensorMap{TT}
end

# Linear Algebra
# --------------
Base.:*(α::Number, mpo::AbstractMPO) = scale(mpo, α)
Base.:*(mpo::AbstractMPO, α::Number) = scale(mpo, α)
Base.:/(mpo::AbstractMPO, α::Number) = scale(mpo, inv(α))
Base.:\(α::Number, mpo::AbstractMPO) = scale(mpo, inv(α))

VectorInterface.scale(mpo::AbstractMPO, α::Number) = scale!(copy(mpo), α)

LinearAlgebra.norm(mpo::AbstractMPO) = sqrt(abs(dot(mpo, mpo)))

function Base.:(^)(a::AbstractMPO, n::Int)
    n >= 1 || throw(DomainError(n, "n should be a positive integer"))
    return Base.power_by_squaring(a, n)
end

Base.conj(mpo::AbstractMPO) = conj!(copy(mpo))
function Base.conj!(mpo::AbstractMPO)
    foreach(mpo) do o
        @plansor q[-1 -2; -3 -4] := conj(o[-1 -3; -2 -4])
    end
    return mpo
end
