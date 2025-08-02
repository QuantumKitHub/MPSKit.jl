# MultilineMPS
# ------------
const MultilineMPS = Multiline{<:InfiniteMPS}

@doc """
    const MultilineMPS = Multiline{<:InfiniteMPS}

Type that represents multiple lines of `InfiniteMPS` objects.

# Constructors
    MultilineMPS(mpss::AbstractVector{<:InfiniteMPS})
    MultilineMPS([f, eltype], physicalspaces::Matrix{<:Union{S, CompositeSpace{S}},
                 virtualspaces::Matrix{<:Union{S, CompositeSpace{S}}) where
                 {S<:ElementarySpace}
    MultilineMPS(As::AbstractMatrix{<:GenericMPSTensor}; kwargs...)
    MultilineMPS(ALs::AbstractMatrix{<:GenericMPSTensor}, 
                 C₀::AbstractVector{<:MPSBondTensor}; kwargs...)

See also: [`Multiline`](@ref)
"""
function MultilineMPS end

MultilineMPS(mpss::AbstractVector{<:InfiniteMPS}) = Multiline(mpss)
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

site_type(::Type{Multiline{S}}) where {S} = site_type(S)
bond_type(::Type{Multiline{S}}) where {S} = bond_type(S)
site_type(st::Multiline) = site_type(typeof(st))
bond_type(st::Multiline) = bond_type(typeof(st))
VectorInterface.scalartype(::Multiline{T}) where {T} = scalartype(T)
TensorKit.sectortype(t::Multiline) = sectortype(typeof(t))
TensorKit.sectortype(::Type{Multiline{T}}) where {T} = sectortype(T)
TensorKit.spacetype(t::Multiline) = spacetype(typeof(t))
TensorKit.spacetype(::Type{Multiline{T}}) where {T} = spacetype(T)

function TensorKit.dot(a::MultilineMPS, b::MultilineMPS; kwargs...)
    return sum(dot.(parent(a), parent(b); kwargs...))
end
TensorKit.normalize!(a::MultilineMPS) = (normalize!.(parent(a)); return a)

Base.convert(::Type{MultilineMPS}, st::InfiniteMPS) = Multiline([st])
Base.convert(::Type{InfiniteMPS}, st::MultilineMPS) = only(st)
Base.eltype(t::MultilineMPS) = eltype(t[1])
Base.copy!(ψ::MultilineMPS, ϕ::MultilineMPS) = (copy!.(parent(ψ), parent(ϕ)); ψ)

for f_space in (:physicalspace, :left_virtualspace, :right_virtualspace)
    @eval $f_space(t::MultilineMPS, i::Int, j::Int) = $f_space(t[i], j)
    @eval $f_space(t::MultilineMPS, I::CartesianIndex{2}) = $f_space(t, Tuple(I)...)
    @eval $f_space(t::MultilineMPS) = map(Base.Fix1($f_space, t), eachindex(t))
end
