# MPSMultiline
# ------------
const MPSMultiline = Multiline{<:InfiniteMPS}

@doc """
    const MPSMultiline = Multiline{<:InfiniteMPS}

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
function MPSMultiline end

MPSMultiline(mpss::AbstractVector{<:InfiniteMPS}) = Multiline(mpss)
function MPSMultiline(pspaces::AbstractMatrix{S}, Dspaces::AbstractMatrix{S};
                      kwargs...) where {S<:VectorSpace}
    data = map(eachrow(pspaces), eachrow(Dspaces)) do p, D
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
    data = map(eachrow(ALs), C₀) do ALrow, C₀row
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
VectorInterface.scalartype(::Multiline{T}) where {T} = scalartype(T)

function TensorKit.dot(a::MPSMultiline, b::MPSMultiline; kwargs...)
    return sum(dot.(parent(a), parent(b); kwargs...))
end
TensorKit.normalize!(a::MPSMultiline) = (normalize!.(parent(a)); return a)

Base.convert(::Type{MPSMultiline}, st::InfiniteMPS) = Multiline([st])
Base.convert(::Type{InfiniteMPS}, st::MPSMultiline) = only(st)
Base.eltype(t::MPSMultiline) = eltype(t[1])
left_virtualspace(t::MPSMultiline, i::Int, j::Int) = left_virtualspace(t[i], j)
right_virtualspace(t::MPSMultiline, i::Int, j::Int) = right_virtualspace(t[i], j)
