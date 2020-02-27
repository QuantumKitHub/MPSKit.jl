#probably doesn't belong in utility; we fix the convention for the orientation of the legs
const MPOTensor{S}=AbstractTensorMap{S,2,2}
const MPSBondTensor{S}=AbstractTensorMap{S,1,1}
const GenericMPSTensor{S,N}=AbstractTensorMap{S,N,1} #some functions are also defined for "general mps tensors" (used in peps code)
const MPSTensor{S}=GenericMPSTensor{S,2} #the usual mps tensors on which we work
#const ExMPSTensor{S,N,A,G,F1,F2}=GenericMPSTensor{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg
