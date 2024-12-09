# MPOMultiline
# ------------
"""
    const MPOMultiline = Multiline{<:Union{SparseMPO,DenseMPO}}

Type that represents multiple lines of `MPO` objects.

# Constructors
    MPOMultiline(mpos::AbstractVector{<:Union{SparseMPO,DenseMPO}})
    MPOMultiline(Os::AbstractMatrix{<:MPOTensor})

See also: [`Multiline`](@ref), [`SparseMPO`](@ref), [`DenseMPO`](@ref)
"""
const MPOMultiline = Multiline{<:AbstractMPO}

function MPOMultiline(Os::AbstractMatrix{T}) where {T<:MPOTensor}
    return MPOMultiline(map(FiniteMPO, eachrow(Os)))
end
function MPOMultiline(Os::PeriodicMatrix{T}) where {T<:MPOTensor}
    return MPOMultiline(map(InfiniteMPO, eachrow(Os)))
end
MPOMultiline(mpos::AbstractVector{<:AbstractMPO}) = Multiline(mpos)
MPOMultiline(t::MPOTensor) = MPOMultiline(PeriodicMatrix(fill(t, 1, 1)))

# allow indexing with two indices
Base.getindex(t::MPOMultiline, ::Colon, j::Int) = Base.getindex.(t.data, j)
Base.getindex(t::MPOMultiline, i::Int, j) = Base.getindex(t[i], j)
Base.getindex(t::MPOMultiline, I::CartesianIndex{2}) = t[I.I...]

# converters
Base.convert(::Type{MPOMultiline}, t::AbstractMPO) = Multiline([t])
Base.convert(::Type{DenseMPO}, t::MPOMultiline{<:DenseMPO}) = only(t)
Base.convert(::Type{SparseMPO}, t::MPOMultiline{<:SparseMPO}) = only(t)
Base.convert(::Type{FiniteMPO}, t::MPOMultiline{<:FiniteMPO}) = only(t)
Base.convert(::Type{InfiniteMPO}, t::MPOMultiline{<:InfiniteMPO}) = only(t)

function Base.:*(mpo::MPOMultiline, st::MPSMultiline)
    size(mpo) == size(st) || throw(ArgumentError("dimension mismatch"))
    return Multiline(map(*, zip(mpo, st)))
end

function Base.:*(mpo1::MPOMultiline, mpo2::MPOMultiline)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))
    return Multiline(map(*, zip(mpo1, mpo2)))
end
