"""
    WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS

Type that represents a finite Matrix Product State embedded in an infinte Matrix Product State.

## Fields

- `left_gs::InfiniteMPS` -- left infinite environment
- `window::FiniteMPS` -- finite window Matrix Product State
- `right_gs::InfiniteMPS` -- right infinite environment

---

## Constructors

    WindowMPS(left_gs::InfiniteMPS, window_state::FiniteMPS, [right_gs::InfiniteMPS])
    WindowMPS(left_gs::InfiniteMPS, window_tensors::AbstractVector, [right_gs::InfiniteMPS])
    WindowMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
              virtualspaces::Vector{<:Union{S, CompositeSpace{S}}, left_gs::InfiniteMPS,
              [right_gs::InfiniteMPS])
    WindowMPS([f, eltype], physicalspaces::Vector{<:Union{S,CompositeSpace{S}}},
              maxvirtualspace::S, left_gs::InfiniteMPS, [right_gs::InfiniteMPS])
    
Construct a WindowMPS via a specification of left and right infinite environment, and either
a window state or a vector of tensors to construct the window. Alternatively, it is possible
to supply the same arguments as for the constructor of [`FiniteMPS`](@ref), followed by a
left (and right) environment to construct the WindowMPS in one step.

!!! note
    By default, the right environment is chosen to be equal to the left, however no copy is
    made. In this case, changing the left state will also affect the right state.

    WindowMPS(state::InfiniteMPS, L::Int)

Construct a WindowMPS from an InfiniteMPS, by promoting a region of length `L` to a
`FiniteMPS`.
"""
mutable struct WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS
    left_gs::InfiniteMPS{A,B}
    window::FiniteMPS{A,B}
    right_gs::InfiniteMPS{A,B}

    function WindowMPS{A,B}(Ψₗ::InfiniteMPS{A,B}, Ψₘ::FiniteMPS{A,B},
                       Ψᵣ::InfiniteMPS{A,B}=Ψₗ) where {A<:GenericMPSTensor,B<:MPSBondTensor}
        right_virtualspace(Ψₗ, 0) == left_virtualspace(Ψₘ, 0) &&
            right_virtualspace(Ψₘ, length(Ψₘ)) == left_virtualspace(Ψᵣ, 0) ||
            throw(SpaceMismatch("Mismatch between window and environment virtual spaces"))
        return new{A,B}(Ψₗ, Ψₘ, Ψᵣ)
    end
end

#===========================================================================================
Constructors
===========================================================================================#
# here i would like an outer constructor that copies the left and right inf environments
function WindowMPS(Ψₗ::InfiniteMPS{A,B},Ψₘ::FiniteMPS{A,B},
                Ψᵣ::InfiniteMPS{A,B}=Ψₗ; docopy = true) where {A<:GenericMPSTensor,B<:MPSBondTensor}
    return WindowMPS{A,B}(docopy ? copy(Ψₗ) : Ψₗ, Ψₘ, docopy ? copy(Ψᵣ) : Ψᵣ)
end

function WindowMPS(Ψₗ::InfiniteMPS, site_tensors::AbstractVector{<:GenericMPSTensor},
                   Ψᵣ::InfiniteMPS=Ψₗ)
    return WindowMPS(Ψₗ, FiniteMPS(site_tensors), Ψᵣ)
end

function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   maxvirtspace::S, Ψₗ::InfiniteMPS,
                   Ψᵣ::InfiniteMPS=Ψₗ) where {S<:ElementarySpace}
    Ψₘ = FiniteMPS(f, elt, physspaces, maxvirtspace; left=left_virtualspace(Ψₗ, 0),
                   right=right_virtualspace(Ψᵣ, length(physspaces)))
    return WindowMPS(Ψₗ, Ψₘ, Ψᵣ)
end
function WindowMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   maxvirtspace::S, Ψₗ::InfiniteMPS,
                   Ψᵣ::InfiniteMPS=Ψₗ) where {S<:ElementarySpace}
    return WindowMPS(rand, Defaults.eltype, physspaces, maxvirtspace, Ψₗ, Ψᵣ)
end

function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   virtspaces::Vector{S}, Ψₗ::InfiniteMPS,
                   Ψᵣ::InfiniteMPS=Ψₗ) where {S<:ElementarySpace}
    Ψₘ = FiniteMPS(f, elt, physspaces, virtspaces)
    return WindowMPS(Ψₗ, Ψₘ, Ψᵣ)
end
function WindowMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   virtspaces::Vector{S}, Ψₗ::InfiniteMPS,
                   Ψᵣ::InfiniteMPS=Ψₗ) where {S<:ElementarySpace}
    return WindowMPS(rand, Defaults.eltype, physspaces, virtspaces, Ψₗ, Ψᵣ)
end

function WindowMPS(f, elt, P::ProductSpace, args...; kwargs...)
    return WindowMPS(f, elt, collect(P), args...; kwargs...)
end
function WindowMPS(P::ProductSpace, args...; kwargs...)
    return WindowMPS(collect(P), args...; kwargs...)
end

function WindowMPS(f, elt, N::Int, V::VectorSpace, args...; kwargs...)
    return WindowMPS(f, elt, fill(V, N), args...; kwargs...)
end
function WindowMPS(N::Int, V::VectorSpace, args...; kwargs...)
    return WindowMPS(fill(V, N), args...; kwargs...)
end

function WindowMPS(Ψ::InfiniteMPS{A,B}, L::Int) where {A,B}
    CLs = Vector{Union{Missing,B}}(missing, L + 1)
    ALs = Vector{Union{Missing,A}}(missing, L)
    ARs = Vector{Union{Missing,A}}(missing, L)
    ACs = Vector{Union{Missing,A}}(missing, L)

    ALs .= Ψ.AL[1:L]
    ARs .= Ψ.AR[1:L]
    ACs .= Ψ.AC[1:L]
    CLs .= Ψ.CR[0:L]

    return WindowMPS(Ψ, FiniteMPS(ALs, ARs, ACs, CLs), Ψ)
end

#===========================================================================================
Utility
===========================================================================================#

function Base.copy(Ψ::WindowMPS)
    return WindowMPS(Ψ.left_gs, copy(Ψ.window), Ψ.right_gs)
end

# not sure about the underlying methods...
Base.length(Ψ::WindowMPS) = length(Ψ.window)
Base.size(Ψ::WindowMPS, i...) = size(Ψ.window, i...)
Base.eltype(::Type{<:WindowMPS{A}}) where {A} = A

site_type(::Type{<:WindowMPS{A}}) where {A} = A
bond_type(::Type{<:WindowMPS{<:Any,B}}) where {B} = B

TensorKit.space(Ψ::WindowMPS, n::Integer) = space(Ψ.AC[n], 2)
left_virtualspace(Ψ::WindowMPS, n::Integer) = left_virtualspace(Ψ.window, n);
right_virtualspace(Ψ::WindowMPS, n::Integer) = right_virtualspace(Ψ.window, n);

r_RR(Ψ::WindowMPS) = r_RR(Ψ.right_gs, length(Ψ))
l_LL(Ψ::WindowMPS) = l_LL(Ψ.left_gs, 1)

function Base.getproperty(Ψ::WindowMPS, prop::Symbol)
    if prop == :AL
        return ALView(Ψ)
    elseif prop == :AR
        return ARView(Ψ)
    elseif prop == :AC
        return ACView(Ψ)
    elseif prop == :CR
        return CRView(Ψ)
    else
        return getfield(Ψ, prop)
    end
end

max_Ds(Ψ::WindowMPS) = max_Ds(Ψ.window)

Base.:*(Ψ::WindowMPS, a::Number) = rmul!(copy(Ψ), a)
Base.:*(a::Number, Ψ::WindowMPS) = lmul!(a, copy(Ψ))

function TensorKit.lmul!(a::Number, Ψ::WindowMPS)
    lmul!(a, Ψ.window)
    return Ψ
end

function TensorKit.rmul!(Ψ::WindowMPS, a::Number)
    rmul!(Ψ.window, a)
    return Ψ
end

function TensorKit.dot(Ψ₁::WindowMPS, Ψ₂::WindowMPS; kwargs...)
    length(Ψ₁) == length(Ψ₂) || throw(ArgumentError("MPS with different length"))
    isapprox(Ψ₁.left_gs,Ψ₂.left_gs;kwargs...) || throw(ArgumentError("left InfiniteMPS is different"))
    isapprox(Ψ₁.right_gs,Ψ₂.right_gs;kwargs...) || throw(ArgumentError("right InfiniteMPS is different"))

    ρr = TransferMatrix(Ψ₂.AR[2:end], Ψ₁.AR[2:end]) * r_RR(Ψ₂)
    return tr(_transpose_front(Ψ₁.AC[1])' * _transpose_front(Ψ₂.AC[1]) * ρr)
end

TensorKit.norm(Ψ::WindowMPS) = norm(Ψ.window)

TensorKit.normalize!(Ψ::WindowMPS) = normalize!(Ψ.window)
TensorKit.normalize(Ψ::WindowMPS) = normalize!(copy(Ψ))
