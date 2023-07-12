"""
    InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbtractMPS

Type that represents an infinite Matrix Product State.

## Fields
- `AL` -- left-gauged MPS tensors
- `AR` -- right-gauged MPS tensors
- `AC` -- center-gauged MPS tensors
- `CR` -- gauge tensors

## Notes
By convention, we have that:
- `AL[i] * CR[i]` = `AC[i]` = `CR[i-1] * AR[i]`
- `AL[i]' * AL[i] = 1`
- `AR[i] * AR[i]' = 1`

---

## Constructors
    InfiniteMPS([f, eltype], physicalspaces::Vector{<:Union{S, CompositeSpace{S}},
                virtualspaces::Vector{<:Union{S, CompositeSpace{S}}) where
                {S<:ElementarySpace}
    InfiniteMPS(As::AbstractVector{<:GenericMPSTensor}; tol=1e-14, maxiter=100)
    InfiniteMPS(ALs::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor;
                tol=1e-14, maxiter=100)

Construct an MPS via a specification of physical and virtual spaces, or from a list of
tensors `As`, or a list of left-gauged tensors `ALs`.

### Arguments
- `As::AbstractVector{<:GenericMPSTensor}`: vector of site tensors
- `ALs::AbstractVector{<:GenericMPSTensor}`: vector of left-gauged site tensors
- `C₀::MPSBondTensor`: initial gauge tensor

- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspaces::AbstractVector{<:Union{S, CompositeSpace{S}}`: list of physical spaces
- `virtualspaces::AbstractVector{<:Union{S, CompositeSpace{S}}`: list of virtual spaces

### Keywords
- `tol`: gauge fixing tolerance
- `maxiter`: gauge fixing maximum iterations
"""
struct InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} <: AbstractMPS
    AL::PeriodicArray{A,1}
    AR::PeriodicArray{A,1}
    CR::PeriodicArray{B,1}
    AC::PeriodicArray{A,1}
end

#===========================================================================================
Constructors
===========================================================================================#

function InfiniteMPS(AL::AbstractVector{A}, AR::AbstractVector{A}, CR::AbstractVector{B},
                     AC::AbstractVector{A}=PeriodicArray(AL .* CR)) where {A<:GenericMPSTensor,
                                                                           B<:MPSBondTensor}
    return InfiniteMPS{A,B}(AL, AR, CR, AC)
end

function InfiniteMPS(pspaces::AbstractVector{S}, Dspaces::AbstractVector{S};
                     kwargs...) where {S<:IndexSpace}
    return InfiniteMPS(MPSTensor.(pspaces, circshift(Dspaces, 1), Dspaces); kwargs...)
end

InfiniteMPS(d::S, D::S) where {S<:Union{Int,<:IndexSpace}} = InfiniteMPS([d], [D])
function InfiniteMPS(ds::AbstractVector{Int}, Ds::AbstractVector{Int})
    return InfiniteMPS(ComplexSpace.(ds), ComplexSpace.(Ds))
end

function InfiniteMPS(A::AbstractVector{<:GenericMPSTensor}; kwargs...)
    AR = PeriodicArray(copy.(A)) # copy to avoid side effects
    leftvspaces = circshift(_firstspace.(AR), -1)
    rightvspaces = conj.(_lastspace.(AR))
    isnothing(findfirst(leftvspaces .!= rightvspaces)) ||
        throw(SpaceMismatch("incompatible virtual spaces $leftvspaces and $rightvspaces"))

    CR = PeriodicArray(isomorphism.(storagetype(eltype(A)), leftvspaces, leftvspaces))
    AL = similar.(AR)

    uniform_leftorth!(AL, CR, AR; kwargs...)
    uniform_rightorth!(AR, CR, AL; kwargs...)

    return InfiniteMPS(AL, AR, CR)
end

function InfiniteMPS(AL::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor; kwargs...)
    CR = PeriodicArray(fill(copy(C₀), length(AL)))
    AL = PeriodicArray(copy.(AL))
    AR = similar(AL)
    uniform_rightorth!(AR, CR, AL; kwargs...)
    return InfiniteMPS(AL, AR, CR)
end

function InfiniteMPS(C₀::MPSBondTensor, AR::AbstractVector{<:GenericMPSTensor}; kwargs...)
    CR = PeriodicArray(fill(copy(C₀), length(AR)))
    AR = PeriodicArray(copy.(AR))
    AL = similar(AR)
    uniform_leftorth!(AL, CR, AR; kwargs...)
    return InfiniteMPS(AL, AR, CR)
end

#===========================================================================================
Utility
===========================================================================================#

Base.size(Ψ::InfiniteMPS, args...) = size(Ψ.AL, args...)
Base.length(Ψ::InfiniteMPS) = length(Ψ.AL)
Base.eltype(Ψ::InfiniteMPS) = eltype(Ψ.AL)
Base.copy(Ψ::InfiniteMPS) = InfiniteMPS(copy(Ψ.AL), copy(Ψ.AR), copy(Ψ.CR), copy(Ψ.AC))
function Base.repeat(Ψ::InfiniteMPS, i::Int)
    return InfiniteMPS(repeat(Ψ.AL, i), repeat(Ψ.AR, i),
                       repeat(Ψ.CR, i), repeat(Ψ.AC, i))
end
function Base.similar(Ψ::InfiniteMPS)
    return InfiniteMPS(similar(Ψ.AL), similar(Ψ.AR), similar(Ψ.CR),
                       similar(Ψ.AC))
end
function Base.circshift(st::InfiniteMPS, n)
    return InfiniteMPS(circshift(st.AL, n), circshift(st.AR, n),
                       circshift(st.CR, n), circshift(st.AC, n))
end

function Base.show(io::IO, ::MIME"text/plain", Ψ::InfiniteMPS)
    println(io, "$(length(Ψ))-site InfiniteMPS:")
    for (i, AL) in enumerate(Ψ.AL)
        println(io, "\t$i: ", AL)
    end
    return
end

site_type(::Type{<:InfiniteMPS{A}}) where {A} = A
bond_type(::Type{<:InfiniteMPS{<:Any,B}}) where {B} = B

left_virtualspace(Ψ::InfiniteMPS, n::Integer) = _firstspace(Ψ.CR[n])
right_virtualspace(Ψ::InfiniteMPS, n::Integer) = dual(_lastspace(Ψ.CR[n]))

function physicalspace(Ψ::InfiniteMPS{<:GenericMPSTensor{<:Any,N}}, n::Integer) where {N}
    if N == 1
        return ProductSpace{spacetype(Ψ)}()
    elseif N == 2
        return space(Ψ.AL[n], 2)
    else
        return ProductSpace{spacetype(Ψ), N-1}(space.(Ref(Ψ.AL[n]), Base.front(Base.tail(TensorKit.allind(Ψ.AL[n])))))
    end
end

TensorKit.space(Ψ::InfiniteMPS{<:MPSTensor}, n::Integer) = space(Ψ.AC[n], 2)
function TensorKit.space(Ψ::InfiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = Ψ.AC[n]
    S = spacetype(t)
    return ProductSpace{S}(space.(Ref(t), Base.front(Base.tail(TensorKit.allind(t)))))
end

TensorKit.norm(Ψ::InfiniteMPS) = norm(Ψ.AC[1])
function TensorKit.normalize!(Ψ::InfiniteMPS)
    normalize!.(Ψ.CR)
    normalize!.(Ψ.AC)
    return Ψ
end

function TensorKit.dot(Ψ₁::InfiniteMPS, Ψ₂::InfiniteMPS; krylovdim=30)
    init = similar(Ψ₁.AL[1], _firstspace(Ψ₂.AL[1]) ← _firstspace(Ψ₁.AL[1]))
    randomize!(init)
    (vals, _, convhist) = eigsolve(TransferMatrix(Ψ₂.AL, Ψ₁.AL), init, 1, :LM,
                                   Arnoldi(; krylovdim=krylovdim))
    convhist.converged == 0 && @info "dot mps not converged"
    return vals[1]
end

#===========================================================================================
Fixedpoints
===========================================================================================#

"""
    l_RR(Ψ, location)

Left dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
l_RR(Ψ::InfiniteMPS, loc::Int=1) = adjoint(Ψ.CR[loc - 1]) * Ψ.CR[loc - 1]

"""
    l_RL(Ψ, location)

Left dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
l_RL(Ψ::InfiniteMPS, loc::Int=1) = Ψ.CR[loc - 1]

"""
    l_LR(Ψ, location)

Left dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
l_LR(Ψ::InfiniteMPS, loc::Int=1) = Ψ.CR[loc - 1]'

"""
    l_LL(Ψ, location)

Left dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
function l_LL(Ψ::InfiniteMPS{A}, loc::Int=1) where {A}
    return isomorphism(storagetype(A), space(Ψ.AL[loc], 1), space(Ψ.AL[loc], 1))
end

"""
    r_RR(Ψ, location)

Right dominant eigenvector of the `AR`-`AR` transfermatrix.
"""
function r_RR(Ψ::InfiniteMPS{A}, loc::Int=length(Ψ)) where {A}
    return isomorphism(storagetype(A), domain(Ψ.AR[loc]), domain(Ψ.AR[loc]))
end

"""
    r_RL(Ψ, location)

Right dominant eigenvector of the `AR`-`AL` transfermatrix.
"""
r_RL(Ψ::InfiniteMPS, loc::Int=length(Ψ)) = Ψ.CR[loc]'

"""
    r_LR(Ψ, location)

Right dominant eigenvector of the `AL`-`AR` transfermatrix.
"""
r_LR(Ψ::InfiniteMPS, loc::Int=length(Ψ)) = Ψ.CR[loc]

"""
    r_LL(Ψ, location)

Right dominant eigenvector of the `AL`-`AL` transfermatrix.
"""
r_LL(Ψ::InfiniteMPS, loc::Int=length(Ψ)) = Ψ.CR[loc] * adjoint(Ψ.CR[loc])
