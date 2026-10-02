# MultilineMPO
# ------------
"""
    const MultilineMPO = Multiline{<:AbstractMPO}

Type that represents multiple lines of `MPO` objects, i.e. the rows of the row-to-row transfer
operator of a two-dimensional tensor network.
See the manual on [MultilineMPO](@ref) for details.

# Constructors

    MultilineMPO(mpos::AbstractVector{<:AbstractMPO})
    MultilineMPO(Os::PeriodicMatrix{<:MPOTensor})
    MultilineMPO(t::MPOTensor)

!!! note "Finite lines"
    Lines may be any `AbstractMPO`, so that for example finite networks can be built and
    inspected. No algorithm supports anything but infinite lines yet: [`leading_boundary`](@ref)
    only accepts an [`InfiniteMultilineMPO`](@ref).

# See also

[`Multiline`](@ref), [`MultilineMPS`](@ref), [`leading_eigenvalue`](@ref)
"""
const MultilineMPO = Multiline{<:AbstractMPO}

"""
    const InfiniteMultilineMPO = Multiline{<:InfiniteMPO}

[`MultilineMPO`](@ref) with infinite lines, as used by [`leading_boundary`](@ref).
"""
const InfiniteMultilineMPO = Multiline{<:InfiniteMPO}

"""
    const FiniteMultilineMPO = Multiline{<:FiniteMPO}

[`MultilineMPO`](@ref) with finite lines. These can be built and inspected, but no algorithm
supports them yet.
"""
const FiniteMultilineMPO = Multiline{<:FiniteMPO}

function MultilineMPO(Os::PeriodicMatrix)
    return MultilineMPO(map(InfiniteMPO, eachrow(Os)))
end
MultilineMPO(mpos::AbstractVector{<:AbstractMPO}) = Multiline(mpos)
MultilineMPO(t::MPOTensor) = MultilineMPO(PeriodicMatrix(fill(t, 1, 1)))

# allow indexing with two indices
Base.getindex(t::MultilineMPO, ::Colon, j::Int) = Base.getindex.(parent(t), j)
Base.getindex(t::MultilineMPO, i::Int, j) = Base.getindex(t[i], j)
Base.getindex(t::MultilineMPO, I::CartesianIndex{2}) = t[I.I...]

# converters
Base.convert(::Type{MultilineMPO}, t::AbstractMPO) = Multiline([t])
Base.convert(::Type{DenseMPO}, t::MultilineMPO{<:DenseMPO}) = only(t)
Base.convert(::Type{SparseMPO}, t::MultilineMPO{<:SparseMPO}) = only(t)
Base.convert(::Type{InfiniteMPO}, t::InfiniteMultilineMPO) = only(t)
Base.convert(::Type{FiniteMPO}, t::FiniteMultilineMPO) = only(t)

function Base.:*(mpo::InfiniteMultilineMPO, st::InfiniteMPS)
    check_length(mpo[1], st)
    for i in 1:size(mpo, 1)
        st = mpo[i] * st
    end
    return st
end

for f_space in (:physicalspace, :left_virtualspace, :right_virtualspace)
    @eval $f_space(t::MultilineMPO, i::Int, j::Int) = $f_space(t[i], j)
    @eval $f_space(t::MultilineMPO, I::CartesianIndex{2}) = $f_space(t, Tuple(I)...)
    @eval $f_space(t::MultilineMPO) = map(Base.Fix1($f_space, t), CartesianIndices((length(t), width(t))))
end

TensorKit.leftunit(t::MultilineMPO) = TensorKit.leftunit(t[1]) # same for every line
TensorKit.rightunit(t::MultilineMPO) = TensorKit.rightunit(t[1])
