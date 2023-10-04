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
const MPOMultiline = Multiline{<:Union{SparseMPO,DenseMPO}}

MPOMultiline(Os::AbstractMatrix{<:MPOTensor}) = MPOMultiline(map(DenseMPO, eachrow(Os)))
MPOMultiline(mpos::AbstractVector{<:Union{SparseMPO,DenseMPO}}) = Multiline(mpos)
MPOMultiline(t::MPOTensor) = MPOMultiline(fill(t, 1, 1))

# allow indexing with two indices
Base.getindex(t::MPOMultiline, ::Colon, j::Int) = Base.getindex.(t.data, j)
Base.getindex(t::MPOMultiline, i::Int, j) = Base.getindex(t[i], j)
Base.getindex(t::MPOMultiline, I::CartesianIndex{2}) = t[I.I...]

# converters
Base.convert(::Type{MPOMultiline}, t::Union{SparseMPO,DenseMPO}) = Multiline([t])
Base.convert(::Type{DenseMPO}, t::MPOMultiline) = only(t)
Base.convert(::Type{SparseMPO}, t::MPOMultiline) = only(t)

function Base.:*(mpo::MPOMultiline, st::MPSMultiline)
    size(mpo) == size(st) || throw(ArgumentError("dimension mismatch"))
    return Multiline(map(*, zip(mpo, st)))
end

function Base.:*(mpo1::MPOMultiline, mpo2::MPOMultiline)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))
    return Multiline(map(*, zip(mpo1, mpo2)))
end
