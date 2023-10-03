"
It is possible to have matrix product (operators / states) that are also periodic in the vertical direction
For examples, as fix points of statmech problems
These should be represented as respectively MultiLine{<:DenseMPO} / Multiline{<:InfiniteMPS}
"

"""
    struct Multiline{T} <: AbstractVector{T}

Object that represents multiple lines of objects of type `T`. Typically used to represent
multiple lines of `InfiniteMPS` (`MPSMultiline`) or MPO (`Multiline{<:AbstractMPO}`).

# Fields
- `data::PeriodicArray{T,1}`: the data of the multiline object

See also: [`MPSMultiline`](@ref) and [`MPOMultiline`](@ref)
"""
struct Multiline{T} <: AbstractVector{T}
    data::PeriodicArray{T,1}
end

Multiline(data::AbstractVector{T}) where {T} = Multiline{T}(data)

# AbstractArray interface
# -----------------------
Base.parent(m::Multiline) = m.data
Base.size(m) = size(parent(m))
Base.getindex(m, i::Int) = parent(m)[i]
Base.setindex!(m, v, i::Int) = (setindex!(parent(m), v, i); m)

Base.copy(t::Multiline) = Multiline(map(copy, t.data))
# Multiline(t::AbstractArray) = Multiline(PeriodicArray(t));
# Base.iterate(t::Multiline, args...) = iterate(t.data, args...);

# MPSMultiline
# ------------
"""
    const MPSMultiline{A<:InfiniteMPS} = Multiline{A}

Type that represents multiple lines of `InfiniteMPS` objects.

# Constructors
    MPSMultiline(mpss::AbstractVector{<:InfiniteMPS})
    MPSMultiline([f, eltype], physicalspaces::Matrix{<:Union{S, CompositeSpace{S}},
                 virtualspaces::Matrix{<:Union{S, CompositeSpace{S}}) where
                 {S<:ElementarySpace}
    MPSMultiline(As::AbstractMatrix{<:GenericMPSTensor}; kwargs...)
    MPSMultiline(ALs::AbstractMatrix{<:GenericMPSTensor}, 
                 C₀::AbstractVector{<:MPSBondTensor}; kwargs...)

See also: [`Multiline`](@ref)
"""
const MPSMultiline{A<:InfiniteMPS} = Multiline{A}

function MPSMultiline(pspaces::AbstractMatrix{S}, Dspaces::AbstractMatrix{S}; kwargs...) where {S}
    data = map(eachrow(pspaces), eachrow(Dspaces)) do (p, D)
        return InfiniteMPS(p, D; kwargs...)
    end
    return MPSMultiline(data)
end
function MPSMultiline(As::AbstractMatrix{T}; kwargs...) where {T<:GenericMPSTensor}
    data = map(eachrow(As)) do Arow
        return InfiniteMPS(Arow; kwargs...)
    end
    return MPSMultiline(data)
end
function MPSMultiline(ALs::AbstractMatrix{<:GenericMPSTensor}, 
                      C₀::AbstractVector{<:MPSBondTensor}; kwargs...)
    data = map(eachrow(ALs), C₀) do (ALrow, C₀row)
        return InfiniteMPS(ALrow, C₀row; kwargs...)
    end
    return MPSMultiline(data)
end

# TODO: properly rewrite these properties
function Base.getproperty(psi::MPSMultiline, prop::Symbol)
    if prop == :AL
        return ALView(psi)
    elseif prop == :AR
        return ARView(psi)
    elseif prop == :AC
        return ACView(psi)
    elseif prop == :CR
        return CRView(psi)
    else
        return getfield(psi, prop)
    end
end

for f in (:l_RR, :l_RL, :l_LL, :l_LR)
    @eval $f(t::MPSMultiline, i, j=1) = $f(t[i], j)
end

for f in (:r_RR, :r_RL, :r_LR, :r_LL)
    @eval $f(t::MPSMultiline, i, j=size(t, 2)) = $f(t[i], j)
end

site_type(::Type{Multiline{S}}) where {S} = site_type(S)
bond_type(::Type{Multiline{S}}) where {S} = bond_type(S)
site_type(st::Multiline) = site_type(typeof(st))
bond_type(st::Multiline) = bond_type(typeof(st))

function TensorKit.dot(a::MPSMultiline, b::MPSMultiline; kwargs...)
    return sum(dot.(parent(a), parent(b); kwargs...))
end

Base.convert(::Type{MPSMultiline}, st::InfiniteMPS) = Multiline([st])
Base.convert(::Type{InfiniteMPS}, st::MPSMultiline) = only(st)
# Base.eltype(t::MPSMultiline) = eltype(t[1]);
left_virtualspace(t::MPSMultiline, i::Int, j::Int) = left_virtualspace(t[i], j);
right_virtualspace(t::MPSMultiline, i::Int, j::Int) = right_virtualspace(t[i], j);
