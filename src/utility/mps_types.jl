#probably doesn't belong in utility; we fix the convention for the orientation of the legs
const MPOType{S}=AbstractTensorMap{S,2,2}
const MPSVecType{S}=AbstractTensorMap{S,1,1}
const GenMPSType{S,N}=AbstractTensorMap{S,N,1} #some functions are also defined for "general mps tensors" (used in peps code)
const MPSType{S}=GenMPSType{S,2} #the usual mps tensors on which we work
#const ExMPSType{S,N,A,G,F1,F2}=GenMPSType{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg
