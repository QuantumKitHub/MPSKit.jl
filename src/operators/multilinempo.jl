# MultilineMPO
# ------------
"""
    const MultilineMPO = Multiline{<:AbstractMPO}

Type that represents multiple lines of `MPO` objects.

# Constructors
    MultilineMPO(mpos::AbstractVector{<:Union{SparseMPO,DenseMPO}})
    MultilineMPO(Os::AbstractMatrix{<:MPOTensor})

See also: [`Multiline`](@ref), [`AbstractMPO`](@ref)
"""
const MultilineMPO = Multiline{<:AbstractMPO}

function MultilineMPO(Os::AbstractMatrix{T}) where {T<:MPOTensor}
    return MultilineMPO(map(FiniteMPO, eachrow(Os)))
end
function MultilineMPO(Os::PeriodicMatrix{T}) where {T<:MPOTensor}
    return MultilineMPO(map(InfiniteMPO, eachrow(Os)))
end
MultilineMPO(mpos::AbstractVector{<:AbstractMPO}) = Multiline(mpos)
MultilineMPO(t::MPOTensor) = MultilineMPO(PeriodicMatrix(fill(t, 1, 1)))

# allow indexing with two indices
Base.getindex(t::MultilineMPO, ::Colon, j::Int) = Base.getindex.(t.data, j)
Base.getindex(t::MultilineMPO, i::Int, j) = Base.getindex(t[i], j)
Base.getindex(t::MultilineMPO, I::CartesianIndex{2}) = t[I.I...]

# converters
Base.convert(::Type{MultilineMPO}, t::AbstractMPO) = Multiline([t])
Base.convert(::Type{DenseMPO}, t::MultilineMPO{<:DenseMPO}) = only(t)
Base.convert(::Type{SparseMPO}, t::MultilineMPO{<:SparseMPO}) = only(t)
Base.convert(::Type{FiniteMPO}, t::MultilineMPO{<:FiniteMPO}) = only(t)
Base.convert(::Type{InfiniteMPO}, t::MultilineMPO{<:InfiniteMPO}) = only(t)

function Base.:*(mpo::MultilineMPO, st::MultilineMPS)
    size(mpo) == size(st) || throw(ArgumentError("dimension mismatch"))
    return Multiline(map(*, zip(mpo, st)))
end

function Base.:*(mpo1::MultilineMPO, mpo2::MultilineMPO)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))
    return Multiline(map(*, zip(mpo1, mpo2)))
end
