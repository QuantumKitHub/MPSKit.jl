#===========================================================================================
Tensor types
===========================================================================================#

const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}
const MPSBondTensor{S} = AbstractTensorMap{S,1,1} where {S<:EuclideanSpace}
const GenericMPSTensor{S,N} = AbstractTensorMap{S,N,1} where {S<:EuclideanSpace,N} #some functions are also defined for "general mps tensors" (used in peps code)
const MPSTensor{S} = GenericMPSTensor{S,2} where {S<:EuclideanSpace} #the usual mps tensors on which we work
#const ExMPSTensor{S,N,A,G,F1,F2}=GenericMPSTensor{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg

"""
    MPSTensor([f, eltype], d::Int, left_D::Int, [right_D]::Int])
    MPSTensor([f, eltype], physicalspace::Union{S,CompositeSpace{S}}, 
              left_virtualspace::S, [right_virtualspace]::S) where {S<:ElementarySpace}

Construct an `MPSTensor` with given physical and virtual spaces.

### Arguments
- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors

- `physicalspace::Union{S,CompositeSpace{S}}`: physical space
- `left_virtualspace::S`: left virtual space
- `right_virtualspace::S`: right virtual space, defaults to equal left

- `d::Int`: physical dimension
- `left_D::Int`: left virtual dimension
- `right_D::Int`: right virtual dimension
"""
function MPSTensor(f, eltype, P::Union{S,CompositeSpace{S}}, Vₗ::S,
                   Vᵣ::S=Vₗ) where {S<:ElementarySpace}
    return TensorMap(f, eltype, Vₗ ⊗ P ← Vᵣ)
end
function MPSTensor(P::Union{S,CompositeSpace{S}}, Vₗ::S,
                   Vᵣ::S=Vₗ) where {S<:ElementarySpace}
    return MPSTensor(rand, Defaults.eltype, P, Vₗ, Vᵣ)
end

"""
    MPSTensor([f, eltype], d::Int, Dₗ::Int, [Dᵣ]::Int])

Construct an `MPSTensor` with given physical and virtual dimensions.

### Arguments
- `f::Function=rand`: initializer function for tensor data
- `eltype::Type{<:Number}=ComplexF64`: scalar type of tensors
- `d::Int`: physical dimension
- `Dₗ::Int`: left virtual dimension
- `Dᵣ::Int`: right virtual dimension
"""
MPSTensor(f, eltype, d::Int, Dₗ::Int, Dᵣ::Int=Dₗ) = MPSTensor(f, eltype, ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)
MPSTensor(d::Int, Dₗ::Int; Dᵣ::Int=Dₗ) = MPSTensor(ℂ^d, ℂ^Dₗ, ℂ^Dᵣ)

"""
    MPSTensor(A::AbstractArray)

Convert an array to an `MPSTensor`.
"""
function MPSTensor(A::AbstractArray{T}) where {T<:Number}
    @assert ndims(A) > 2 "MPSTensor should have at least 3 dims, but has $ndims(A)"
    sz = size(A)
    t = TensorMap(undef, T, foldl(⊗, ComplexSpace.(sz[1:(end - 1)])) ← ℂ^sz[end])
    t[] .= A
    return t
end

#===========================================================================================
MPS types
===========================================================================================#

abstract type AbstractMPS end

Base.eltype(Ψ::AbstractMPS) = eltype(typeof(Ψ))

"""
    site_type(Ψ::AbstractMPS)
    site_type(Ψtype::Type{<:AbstractMPS})

Return the type of the site tensors of an `AbstractMPS`.
"""
site_type(Ψ::AbstractMPS) = site_type(typeof(Ψ))

"""
    bond_type(Ψ::AbstractMPS)
    bond_type(Ψtype::Type{<:AbstractMPS})

Return the type of the bond tensors of an `AbstractMPS`.
"""
bond_type(Ψ::AbstractMPS) = bond_type(typeof(Ψ))

TensorKit.spacetype(Ψ::AbstractMPS) = spacetype(typeof(Ψ))
TensorKit.spacetype(Ψtype::Type{<:AbstractMPS}) = spacetype(site_type(Ψtype))
TensorKit.sectortype(Ψ::AbstractMPS) = sectortype(typeof(Ψ))
TensorKit.sectortype(Ψtype::Type{<:AbstractMPS}) = sectortype(site_type(Ψtype))

Base.isapprox(Ψ1::AbstractMPS, Ψ2::AbstractMPS; kwargs...) = isapprox(abs(dot(Ψ1,Ψ2)),norm(Ψ1)*norm(Ψ2); kwargs...)

"""
    left_virtualspace(Ψ::AbstractMPS, i::Int)
    
Return the left virtual space of the bond tensor at site `i`. This is equivalent to the
left virtual space of the left-gauged site tensor at site `i + 1`.
"""
function left_virtualspace end

"""
    right_virtualspace(Ψ::AbstractMPS, i::Int)

Return the right virtual space of the bond tensor at site `i`. This is equivalent to the
right virtual space of the right-gauged site tensor at site `i`.
"""
function right_virtualspace end

"""
    physicalspace(Ψ::AbstractMPS, i::Int)

Return the physical space of the site tensor at site `i`.
"""
function physicalspace end


abstract type AbstractFiniteMPS <: AbstractMPS end
