# MultilineMPO
# ------------
"""
    const MultilineMPO = Multiline{<:AbstractMPO}

Type that represents multiple lines of `MPO` objects.

Only `InfiniteMPO` lines are supported currently.
Hamiltonians and finite MPOs are rejected by the constructors.

# Constructors

    MultilineMPO(mpos::AbstractVector{<:InfiniteMPO})
    MultilineMPO(Os::PeriodicMatrix{<:MPOTensor})
    MultilineMPO(t::MPOTensor)

# See also

[`Multiline`](@ref), [`AbstractMPO`](@ref)
"""
const MultilineMPO = Multiline{<:AbstractMPO}

function MultilineMPO(Os::PeriodicMatrix)
    return MultilineMPO(map(InfiniteMPO, eachrow(Os)))
end
MultilineMPO(mpos::AbstractVector{<:InfiniteMPO}) = Multiline(mpos)
MultilineMPO(t::MPOTensor) = MultilineMPO(PeriodicMatrix(fill(t, 1, 1)))

# allow indexing with two indices
Base.getindex(t::MultilineMPO, ::Colon, j::Int) = Base.getindex.(parent(t), j)
Base.getindex(t::MultilineMPO, i::Int, j) = Base.getindex(t[i], j)
Base.getindex(t::MultilineMPO, I::CartesianIndex{2}) = t[I.I...]

# converters
Base.convert(::Type{MultilineMPO}, t::InfiniteMPO) = Multiline([t])
Base.convert(::Type{DenseMPO}, t::MultilineMPO{<:DenseMPO}) = only(t)
Base.convert(::Type{SparseMPO}, t::MultilineMPO{<:SparseMPO}) = only(t)
Base.convert(::Type{InfiniteMPO}, t::MultilineMPO{<:InfiniteMPO}) = only(t)

function Base.:*(mpo::MultilineMPO, st::MultilineMPS)
    size(mpo) == size(st) || throw(ArgumentError("dimension mismatch"))
    # row i of the operator maps row i of the state onto row i + 1
    return Multiline(circshift(map(*, parent(mpo), parent(st)), 1))
end

function Base.:*(mpo1::MultilineMPO, mpo2::MultilineMPO)
    size(mpo1) == size(mpo2) || throw(ArgumentError("dimension mismatch"))
    # `mpo2[i]` maps row i onto row i + 1, where `mpo1[i + 1]` picks up: the resulting
    # operator spans two rows of the lattice and maps row i onto row i + 2
    return Multiline(map(*, circshift(parent(mpo1), -1), parent(mpo2)))
end

for f_space in (:physicalspace, :left_virtualspace, :right_virtualspace)
    @eval $f_space(t::MultilineMPO, i::Int, j::Int) = $f_space(t[i], j)
    @eval $f_space(t::MultilineMPO, I::CartesianIndex{2}) = $f_space(t, Tuple(I)...)
    @eval $f_space(t::MultilineMPO) = map(Base.Fix1($f_space, t), eachindex(t))
end

TensorKit.leftunit(t::MultilineMPO) = TensorKit.leftunit(t[1]) # same for every line
TensorKit.rightunit(t::MultilineMPO) = TensorKit.rightunit(t[1])
