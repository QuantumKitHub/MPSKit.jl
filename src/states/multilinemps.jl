# MultilineMPS
# ------------
#TODO: add support for finite MPS
const _MPSs = Union{InfiniteMPS, FiniteMPS}
const MultilineMPS = Multiline{<:_MPSs}

@doc """
    const MultilineMPS = Multiline{<:Union{InfiniteMPS, FiniteMPS}}

Type that represents multiple lines of MPS objects. When used in the context of 
[`leading_boundary`](@ref) with `InfiniteMPS`, this is not to be confused with the fixed point 
of a 2D tensor network, which is a single `InfiniteMPS`.
See the manual on [MultilineMPS](@ref) for details.

# Constructors

    MultilineMPS(mpss::AbstractVector{<:Union{InfiniteMPS, FiniteMPS}})
    MultilineMPS(
        [f, eltype], physicalspaces::Matrix{<:Union{S, CompositeSpace{S}}},
        virtualspaces::Matrix{<:Union{S, CompositeSpace{S}}}
    ) where {S <: ElementarySpace}
    MultilineMPS(As::AbstractMatrix{<:GenericMPSTensor}; kwargs...)
    MultilineMPS(
        ALs::AbstractMatrix{<:GenericMPSTensor},
        C₀::AbstractVector{<:MPSBondTensor}; kwargs...
    )

# Properties

- `AL`: left-gauged MPS tensors
- `AR`: right-gauged MPS tensors
- `AC`: center-gauged MPS tensors
- `C`: gauge (bond) tensors

Note that `length`, `eltype` and iteration refer to the lines (so e.g. `length(ψ) == nrows`),
while `size` refers to the `(nrows, ncols)` lattice shape.
See [`Multiline`](@ref) for details.

Only the first constructor accepts finite lines; the others build `InfiniteMPS` lines from
spaces or tensors.

!!! note "Finite lines"
    Finite lines are accepted by the type and by the first constructor, but no algorithm
    supports them yet, so they will fail somewhere further down. This is on purpose: there is currently 
    no support for finite multiline boundaries, but this allows them to be built and inspected.

# See also

[`Multiline`](@ref), [`MultilineMPO`](@ref)
"""
function MultilineMPS end

MultilineMPS(mpss::AbstractVector{<:_MPSs}) = Multiline(mpss)
function MultilineMPS(
        pspaces::AbstractMatrix{S}, Dspaces::AbstractMatrix{S}; kwargs...
    ) where {S <: VectorSpace}
    data = map(eachrow(pspaces), eachrow(Dspaces)) do p, D
        return InfiniteMPS(p, D; kwargs...)
    end
    return MultilineMPS(data)
end
function MultilineMPS(As::AbstractMatrix{T}; kwargs...) where {T <: GenericMPSTensor}
    data = map(eachrow(As)) do Arow
        return InfiniteMPS(Arow; kwargs...)
    end
    return MultilineMPS(data)
end
function MultilineMPS(
        ALs::AbstractMatrix{<:GenericMPSTensor}, C₀::AbstractVector{<:MPSBondTensor};
        kwargs...
    )
    data = map(eachrow(ALs), C₀) do ALrow, C₀row
        return InfiniteMPS(ALrow, C₀row; kwargs...)
    end
    return MultilineMPS(data)
end

# TODO: properly rewrite these properties
function Base.getproperty(psi::MultilineMPS, prop::Symbol)
    if prop == :AL
        return ALView(psi)
    elseif prop == :AR
        return ARView(psi)
    elseif prop == :AC
        return ACView(psi)
    elseif prop == :C
        return CView(psi)
    else
        return getfield(psi, prop)
    end
end

function AC2(psi::MultilineMPS, site::CartesianIndex{2}; kwargs...)
    return AC2(psi[site[1]], site[2]; kwargs...)
end
function AC2(psi::MultilineMPS, site::Int; kwargs...)
    return map(1:size(psi, 1)) do row
        return AC2(psi, CartesianIndex(row, site); kwargs...)
    end
end

function Base.propertynames(::MultilineMPS)
    return (:AL, :AR, :AC, :C)
end

for f in (:l_RR, :l_RL, :l_LL, :l_LR)
    @eval $f(t::MultilineMPS, i, j = 1) = $f(t[i], j)
end

for f in (:r_RR, :r_RL, :r_LR, :r_LL)
    @eval $f(t::MultilineMPS, i, j = size(t, 2)) = $f(t[i], j)
end

function TensorKit.dot(a::MultilineMPS, b::MultilineMPS; kwargs...)
    return sum(dot.(parent(a), parent(b); kwargs...))
end
TensorKit.normalize!(a::MultilineMPS) = (normalize!.(parent(a)); return a)

Base.convert(::Type{MultilineMPS}, st::_MPSs) = Multiline([st])
Base.convert(::Type{InfiniteMPS}, st::MultilineMPS{<:InfiniteMPS}) = only(st)
Base.convert(::Type{FiniteMPS}, st::MultilineMPS{<:FiniteMPS}) = only(st)
Base.copy!(ψ::MultilineMPS, ϕ::MultilineMPS) = (copy!.(parent(ψ), parent(ϕ)); ψ)

for f_space in (:physicalspace, :left_virtualspace, :right_virtualspace)
    @eval $f_space(t::MultilineMPS, i::Int, j::Int) = $f_space(t[i], j)
    @eval $f_space(t::MultilineMPS, I::CartesianIndex{2}) = $f_space(t, Tuple(I)...)
    @eval $f_space(t::MultilineMPS) = map(Base.Fix1($f_space, t), eachindex(t))
end

TensorKit.leftunit(t::MultilineMPS) = TensorKit.leftunit(t[1]) # same for every line
TensorKit.rightunit(t::MultilineMPS) = TensorKit.rightunit(t[1])
