# MultilineMPO
# ------------
"""
    const MultilineMPO = Multiline{<:Union{InfiniteMPO, FiniteMPO}}

Type that represents multiple lines of `MPO` objects, i.e. the rows of a two-dimensional
tensor network.

Row `i` maps line `i` of the network onto line `i + 1`, so applying a single row to a
boundary MPS shifts it by one row. Applying every row in turn advances the boundary by one
full period, which is what `*` does.

Lines are restricted to `InfiniteMPO` or `FiniteMPO` objects as `MultilineMPO`
represents rows of a statistical mechanical transfer operator.

# Constructors

    MultilineMPO(mpos::AbstractVector{<:Union{InfiniteMPO, FiniteMPO}})
    MultilineMPO(Os::PeriodicMatrix{<:MPOTensor})
    MultilineMPO(t::MPOTensor)

!!! note "Finite lines"
    Finite lines are accepted by the type and by the constructors, but no algorithm supports
    them yet, so they will fail somewhere further down. This is on purpose: there is currently 
    no support for finite multiline boundaries, but this allows them to be built and inspected.

# See also

[`Multiline`](@ref), [`MultilineMPS`](@ref), [`dominant_eigenvalue`](@ref)
"""
#TODO: add algorithm support for finite MPOs
const _MPOs = Union{InfiniteMPO, FiniteMPO}
const MultilineMPO = Multiline{<:_MPOs}

function MultilineMPO(Os::PeriodicMatrix)
    return MultilineMPO(map(InfiniteMPO, eachrow(Os)))
end
MultilineMPO(mpos::AbstractVector{<:_MPOs}) = Multiline(mpos)
MultilineMPO(t::MPOTensor) = MultilineMPO(PeriodicMatrix(fill(t, 1, 1)))

# allow indexing with two indices
Base.getindex(t::MultilineMPO, ::Colon, j::Int) = Base.getindex.(parent(t), j)
Base.getindex(t::MultilineMPO, i::Int, j) = Base.getindex(t[i], j)
Base.getindex(t::MultilineMPO, I::CartesianIndex{2}) = t[I.I...]

# converters
Base.convert(::Type{MultilineMPO}, t::_MPOs) = Multiline([t])
Base.convert(::Type{DenseMPO}, t::MultilineMPO{<:DenseMPO}) = only(t)
Base.convert(::Type{SparseMPO}, t::MultilineMPO{<:SparseMPO}) = only(t)
Base.convert(::Type{InfiniteMPO}, t::MultilineMPO{<:InfiniteMPO}) = only(t)
Base.convert(::Type{FiniteMPO}, t::MultilineMPO{<:FiniteMPO}) = only(t)

function Base.:*(mpo::MultilineMPO, st::InfiniteMPS)
    check_length(mpo[1], st)
    for i in 1:size(mpo, 1)
        st = mpo[i] * st
    end
    return st
end

for f_space in (:physicalspace, :left_virtualspace, :right_virtualspace)
    @eval $f_space(t::MultilineMPO, i::Int, j::Int) = $f_space(t[i], j)
    @eval $f_space(t::MultilineMPO, I::CartesianIndex{2}) = $f_space(t, Tuple(I)...)
    @eval $f_space(t::MultilineMPO) = map(Base.Fix1($f_space, t), eachindex(t))
end

TensorKit.leftunit(t::MultilineMPO) = TensorKit.leftunit(t[1]) # same for every line
TensorKit.rightunit(t::MultilineMPO) = TensorKit.rightunit(t[1])
