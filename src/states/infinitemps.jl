"""
    struct InfiniteMPS{A<:GenericMPSTensor,B<:MPSBondTensor} end

Represents an infinite matrix product state in the center gauge.
By convention, this means that
- `AL[i] * CR[i]` = `AC[i]` = `CR[i-1] * AR[i]`
- `AL[i]' * AL[i] = 1`
- `AR[i] * AR[i]' = 1`
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

"""
    InfiniteMPS(pspaces, vspaces)
    InfiniteMPS(d, D)

Construct a random MPS with given physical and virtual spaces.
"""
function InfiniteMPS(pspaces::AbstractVector{S}, Dspaces::AbstractVector{S};
                     kwargs...) where {S<:IndexSpace}
    return InfiniteMPS(MPSTensor.(pspaces, circshift(Dspaces, 1), Dspaces); kwargs...)
end
InfiniteMPS(d::S, D::S) where {S<:Union{Int,<:IndexSpace}} = InfiniteMPS([d], [D])
function InfiniteMPS(ds::AbstractArray{Int}, Ds::AbstractArray{Int})
    return InfiniteMPS(ComplexSpace.(ds), ComplexSpace.(Ds))
end

"""
    InfiniteMPS(A; kwargs...)

Construct a gauged mps from tensors A.
"""
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

"""
    InfiniteMPS(AL, C₀; kwargs...)

Construct a gauged mps from left-gauged tensors AL.
"""
function InfiniteMPS(AL::AbstractVector{<:GenericMPSTensor}, C₀::MPSBondTensor; kwargs...)
    CR = PeriodicArray(fill(copy(C₀), length(AL)))
    AL = PeriodicArray(copy.(AL))
    AR = similar(AL)
    uniform_rightorth!(AR, CR, AL; kwargs...)
    return InfiniteMPS(AL, AR, CR)
end

#===========================================================================================
Utility
===========================================================================================#

Base.size(Ψ::InfiniteMPS, args...) = size(Ψ.AL, args...)
Base.length(Ψ::InfiniteMPS) = size(Ψ, 1)
Base.eltype(Ψ::InfiniteMPS) = eltype(Ψ.AL)
Base.copy(Ψ::InfiniteMPS) = InfiniteMPS(copy(Ψ.AL), copy(Ψ.AR), copy(Ψ.CR), copy(Ψ.AC))
Base.repeat(Ψ::InfiniteMPS, i::Int) = InfiniteMPS(repeat(Ψ.AL, i), repeat(Ψ.AR, i),
                                                  repeat(Ψ.CR, i), repeat(Ψ.AC, i))
Base.similar(Ψ::InfiniteMPS) = InfiniteMPS(similar(Ψ.AL), similar(Ψ.AR), similar(Ψ.CR),
                                           similar(Ψ.AC))
Base.circshift(st::InfiniteMPS, n) = InfiniteMPS(circshift(st.AL, n), circshift(st.AR, n),
                                                 circshift(st.CR, n), circshift(st.AC, n))
                                                 
function Base.show(io::IO, ::MIME"text/plain", Ψ::InfiniteMPS)
    println(io, "$(length(Ψ))-site InfiniteMPS:")
    for (i, AL) in enumerate(Ψ.AL)
        println(io, "\t$i: ", AL)
    end
    return
end

site_type(Ψ::InfiniteMPS) = site_type(typeof(Ψ))
site_type(::Type{<:InfiniteMPS{A}}) where {A} = A
bond_type(Ψ::InfiniteMPS) = bond_type(typeof(Ψ))
bond_type(::Type{InfiniteMPS{<:Any,B}}) where {B} = B

left_virtualspace(Ψ::InfiniteMPS, n::Integer) = _firstspace(Ψ.CR[n])
right_virtualspace(Ψ::InfiniteMPS, n::Integer) = dual(_lastspace(Ψ.CR[n]))

TensorKit.space(psi::InfiniteMPS{<:MPSTensor}, n::Integer) = space(psi.AC[n], 2)
function TensorKit.space(psi::InfiniteMPS{<:GenericMPSTensor}, n::Integer)
    t = psi.AC[n]
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

"
    l_RR(state,location)
    Left dominant eigenvector of the AR-AR transfermatrix
"
l_RR(state::InfiniteMPS, loc::Int=1) = adjoint(state.CR[loc - 1]) * state.CR[loc - 1]

"
    l_RL(state,location)
    Left dominant eigenvector of the AR-AL transfermatrix
"
l_RL(state::InfiniteMPS, loc::Int=1) = state.CR[loc - 1]

"
    l_LR(state,location)
    Left dominant eigenvector of the AL-AR transfermatrix
"
l_LR(state::InfiniteMPS, loc::Int=1) = state.CR[loc - 1]'

"
    l_LL(state,location)
    Left dominant eigenvector of the AL-AL transfermatrix
"
function l_LL(state::InfiniteMPS{A}, loc::Int=1) where {A}
    return isomorphism(storagetype(A), space(state.AL[loc], 1), space(state.AL[loc], 1))
end

"
    r_RR(state,location)
    Right dominant eigenvector of the AR-AR transfermatrix
"
function r_RR(state::InfiniteMPS{A}, loc::Int=length(state)) where {A}
    return isomorphism(storagetype(A), domain(state.AR[loc]), domain(state.AR[loc]))
end

"
    r_RL(state,location)
    Right dominant eigenvector of the AR-AL transfermatrix
"
r_RL(state::InfiniteMPS, loc::Int=length(state)) = state.CR[loc]'

"
    r_LR(state,location)
    Right dominant eigenvector of the AL-AR transfermatrix
"
r_LR(state::InfiniteMPS, loc::Int=length(state)) = state.CR[loc]

"
    r_LL(state,location)
    Right dominant eigenvector of the AL-AL transfermatrix
"
r_LL(state::InfiniteMPS, loc::Int=length(state)) = state.CR[loc] * adjoint(state.CR[loc])
