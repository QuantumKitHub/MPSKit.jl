# Interesting MPS TensorMap types
const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}
const MPSBondTensor{S} = AbstractTensorMap{S,1,1} where {S<:EuclideanSpace}
const GenericMPSTensor{S,N} = AbstractTensorMap{S,N,1} where {S<:EuclideanSpace,N} #some functions are also defined for "general mps tensors" (used in peps code)
const MPSTensor{S} = GenericMPSTensor{S,2} where {S<:EuclideanSpace} #the usual mps tensors on which we work
#const ExMPSTensor{S,N,A,G,F1,F2}=GenericMPSTensor{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg

abstract type AbstractMPS end

Base.eltype(psi::AbstractMPS) = eltype(typeof(psi))

abstract type AbstractFiniteMPS <: AbstractMPS end;

"""
    MPSTensor(d::Int, D_left::Int [, D_right::Int])
    MPSTensor(pspace, vspace_left [, vspace_right])

Construct a random `MPSTensor` with given physical and virtual spaces.
"""
MPSTensor(pspace::S, vspace_left::S, vspace_right::S=vspace_left) where {S<:IndexSpace} = 
    TensorMap(rand, Defaults.eltype, vspace_left ⊗ pspace ← vspace_right)
MPSTensor(d::Int, D_left::Int, D_right::Int=D_left) = MPSTensor(ℂ^d, ℂ^D_left, ℂ^D_right)
function MPSTensor(A::AbstractArray{T,3}) where {T<:Number}
    t = TensorMap(undef, T, ℂ^size(A, 1) ⊗ ℂ^size(A, 2) ← ℂ^size(A, 3))
    t[] .= A
    return t
end