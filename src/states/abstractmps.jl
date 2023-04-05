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

site_type(Ψ::AbstractMPS) = site_type(typeof(Ψ))
bond_type(Ψ::AbstractMPS) = bond_type(typeof(Ψ))

abstract type AbstractFiniteMPS <: AbstractMPS end
