"""
    WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractFiniteMPS

Type that represents a finite Matrix Product State embedded in an infinite Matrix Product State.

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

    WindowMPS(state::InfiniteMPS, L::Int)

Construct a WindowMPS from an InfiniteMPS, by promoting a region of length `L` to a
`FiniteMPS`.
"""
struct WindowMPS{A<:GenericMPSTensor,B<:MPSBondTensor,Vₗ,Vᵣ} <: AbstractMPS
    left_gs::InfiniteMPS{A,B}
    window::FiniteMPS{A,B}
    right_gs::InfiniteMPS{A,B}

    function WindowMPS{A,B,Vₗ,Vᵣ}(ψₗ::InfiniteMPS{A,B}, ψₘ::FiniteMPS{A,B},
                       ψᵣ::InfiniteMPS{A,B}) where {A<:GenericMPSTensor,
                                                             B<:MPSBondTensor,Vₗ,Vᵣ}
        left_virtualspace(ψₗ, 0) == left_virtualspace(ψₘ, 0) &&
            right_virtualspace(ψₘ, length(ψₘ)) == right_virtualspace(ψᵣ, length(ψₘ)) ||
            throw(SpaceMismatch("Mismatch between window and environment virtual spaces"))
        return new{A,B,Vₗ,Vᵣ}(ψₗ, ψₘ, ψᵣ)
    end
    
end

function WindowMPS(ψₗ::InfiniteMPS{A,B}, ψₘ::FiniteMPS{A,B},
                   ψᵣ::InfiniteMPS{A,B}) where {A<:GenericMPSTensor,
                                                         B<:MPSBondTensor}
    return WindowMPS{A,B,:V,:V}(ψₗ, ψₘ, ψᵣ)
end

# alias to help dispatching?

#===========================================================================================
Constructors
===========================================================================================#
function WindowMPS(ψₗ::InfiniteMPS, site_tensors::AbstractVector{<:GenericMPSTensor},
                   ψᵣ::InfiniteMPS)
    return WindowMPS(ψₗ, FiniteMPS(site_tensors), ψᵣ)
end

function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   maxvirtspace::S, ψₗ::InfiniteMPS,
                   ψᵣ::InfiniteMPS) where {S<:ElementarySpace}
    ψₘ = FiniteMPS(f, elt, physspaces, maxvirtspace; left=left_virtualspace(ψₗ, 0),
                   right=right_virtualspace(ψᵣ, length(physspaces)))
    return WindowMPS(ψₗ, ψₘ, ψᵣ)
end
function WindowMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, maxvirtspace::S,
                   ψₗ::InfiniteMPS, ψᵣ::InfiniteMPS) where {S<:ElementarySpace}
    return WindowMPS(rand, Defaults.eltype, physspaces, maxvirtspace, ψₗ, ψᵣ)
end

function WindowMPS(f, elt, physspaces::Vector{<:Union{S,CompositeSpace{S}}},
                   virtspaces::Vector{S}, ψₗ::InfiniteMPS,
                   ψᵣ::InfiniteMPS) where {S<:ElementarySpace}
    ψₘ = FiniteMPS(f, elt, physspaces, virtspaces)
    return WindowMPS(ψₗ, ψₘ, ψᵣ)
end
function WindowMPS(physspaces::Vector{<:Union{S,CompositeSpace{S}}}, virtspaces::Vector{S},
                   ψₗ::InfiniteMPS, ψᵣ::InfiniteMPS) where {S<:ElementarySpace}
    return WindowMPS(rand, Defaults.eltype, physspaces, virtspaces, ψₗ, ψᵣ)
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

function WindowMPS(ψ::InfiniteMPS{A,B}, L::Int) where {A,B}
    CLs = Vector{Union{Missing,B}}(missing, L + 1)
    ALs = Vector{Union{Missing,A}}(missing, L)
    ARs = Vector{Union{Missing,A}}(missing, L)
    ACs = Vector{Union{Missing,A}}(missing, L)

    ALs .= ψ.AL[1:L]
    ARs .= ψ.AR[1:L]
    ACs .= ψ.AC[1:L]
    CLs .= ψ.CR[0:L]

    return WindowMPS(ψ, FiniteMPS(ALs, ARs, ACs, CLs), ψ)
end

#===========================================================================================
Left and right variable status changes (usefull for time evolution)
===========================================================================================#
function fix_left(ψ::WindowMPS{A,B,Vₗ,Vᵣ}) where {A,B,Vₗ,Vᵣ}
    return WindowMPS{A,B,:F,Vᵣ}(ψ.left_gs,ψ.window,ψ.right_gs)
end
function fix_left(ψ::WindowMPS{A,B,:V,:F}) where {A,B}
    return copy(ψ.window)
end

function fix_right(ψ::WindowMPS{A,B,Vₗ,Vᵣ}) where {A,B,Vₗ,Vᵣ}
    return WindowMPS{A,B,Vₗ,:F}(ψ.left_gs,ψ.window,ψ.right_gs)
end
function fix_right(ψ::WindowMPS{A,B,:F,:V}) where {A,B}
    return copy(ψ.window)
end

function fix_infinite(ψ::WindowMPS)
    return copy(ψ.window)
end

#===========================================================================================
Utility
===========================================================================================#
function Base.copy(ψ::WindowMPS{A,B,:V,:V}) where {A,B}
    return WindowMPS(copy(ψ.left_gs), copy(ψ.window), copy(ψ.right_gs))
end

function Base.copy(ψ::WindowMPS{A,B,:F,:V}) where {A,B}
    return WindowMPS(ψ.left_gs, copy(ψ.window), copy(ψ.right_gs))
end

function Base.copy(ψ::WindowMPS{A,B,:V,:F}) where {A,B}
    return WindowMPS(copy(ψ.left_gs), copy(ψ.window), ψ.right_gs)
end

# not sure about the underlying methods...
Base.length(ψ::WindowMPS) = length(ψ.window)
Base.size(ψ::WindowMPS, i...) = size(ψ.window, i...)
Base.eltype(::Type{<:WindowMPS{A}}) where {A} = A

site_type(::Type{<:WindowMPS{A}}) where {A} = A
bond_type(::Type{<:WindowMPS{<:Any,B}}) where {B} = B

TensorKit.space(ψ::WindowMPS, n::Integer) = space(ψ.AC[n], 2)
left_virtualspace(ψ::WindowMPS, n::Integer) = left_virtualspace(ψ.window, n);
right_virtualspace(ψ::WindowMPS, n::Integer) = right_virtualspace(ψ.window, n);

r_RR(ψ::WindowMPS) = r_RR(ψ.right_gs, length(ψ))
l_LL(ψ::WindowMPS) = l_LL(ψ.left_gs, 1)

function Base.getproperty(ψ::WindowMPS, prop::Symbol)
    if prop == :AL
        return ALView(ψ)
    elseif prop == :AR
        return ARView(ψ)
    elseif prop == :AC
        return ACView(ψ)
    elseif prop == :CR
        return CRView(ψ)
    else
        return getfield(ψ, prop)
    end
end

max_Ds(ψ::WindowMPS) = max_Ds(ψ.window)

Base.:*(ψ::WindowMPS, a::Number) = rmul!(copy(ψ), a)
Base.:*(a::Number, ψ::WindowMPS) = lmul!(a, copy(ψ))

function TensorKit.lmul!(a::Number, ψ::WindowMPS)
    lmul!(a, ψ.window)
    return ψ
end

function TensorKit.rmul!(ψ::WindowMPS, a::Number)
    rmul!(ψ.window, a)
    return ψ
end

function TensorKit.dot(ψ₁::WindowMPS, ψ₂::WindowMPS)
    length(ψ₁) == length(ψ₂) || throw(ArgumentError("MPS with different length"))
    ψ₁.left_gs == ψ₂.left_gs ||
        dot(ψ₁.left_gs, ψ₂.left_gs) ≈ 1 ||
        throw(ArgumentError("left InfiniteMPS are different"))
    ψ₁.right_gs == ψ₂.right_gs ||
        dot(ψ₁.right_gs, ψ₂.right_gs) ≈ 1 ||
        throw(ArgumentError("right InfiniteMPS are different"))

    ρr = TransferMatrix(ψ₂.AR[2:end], ψ₁.AR[2:end]) * r_RR(ψ₂)
    return tr(_transpose_front(ψ₁.AC[1])' * _transpose_front(ψ₂.AC[1]) * ρr)
end

TensorKit.norm(ψ::WindowMPS) = norm(ψ.window)

TensorKit.normalize!(ψ::WindowMPS) = normalize!(ψ.window)
TensorKit.normalize(ψ::WindowMPS) = normalize!(copy(ψ))
