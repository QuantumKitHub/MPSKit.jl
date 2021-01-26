# Interesting MPS TensorMap types
const MPOTensor{S} = AbstractTensorMap{S,2,2} where {S<:EuclideanSpace}
const MPSBondTensor{S} = AbstractTensorMap{S,1,1} where {S<:EuclideanSpace}
const GenericMPSTensor{S,N} = AbstractTensorMap{S,N,1} where {S<:EuclideanSpace,N} #some functions are also defined for "general mps tensors" (used in peps code)
const MPSTensor{S} = GenericMPSTensor{S,2} where {S<:EuclideanSpace} #the usual mps tensors on which we work
#const ExMPSTensor{S,N,A,G,F1,F2}=GenericMPSTensor{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg

abstract type AbstractMPS end

Base.eltype(psi::AbstractMPS) = eltype(typeof(psi))
