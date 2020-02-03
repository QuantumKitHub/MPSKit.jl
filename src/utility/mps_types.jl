#probably doesn't belong in utility; we fix the convention for the orientation of the legs
const MpoType{S}=AbstractTensorMap{S,2,2}
const MpsVecType{S}=AbstractTensorMap{S,1,1}
const GenMpsType{S,N}=AbstractTensorMap{S,N,1} #some functions are also defined for "general mps tensors" (used in peps code)
const MpsType{S}=GenMpsType{S,2} #the usual mps tensors on which we work
#const ExMpsType{S,N,A,G,F1,F2}=GenMpsType{S,3,A,G,F1,F2} #and mps tensor with an extra excitation - utility leg

struct Periodic{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end

Periodic{T,N}(len...) where {T,N} = Periodic(Array{T,N}(undef,len...))
Base.size(arr::Periodic) = size(arr.data)
Base.size(arr::Periodic,i) = size(arr.data,i)
Base.length(arr::Periodic) = length(arr.data)
Base.getindex(arr::Periodic{T,N},I::Vararg{Int, N}) where {T,N} = getindex(arr.data,broadcast(mod1,I,size(arr.data))...)
Base.setindex!(arr::Periodic{T,N},v,I::Vararg{Int, N}) where {T,N} = setindex!(arr.data,v,broadcast(mod1,I,size(arr.data))...)
Base.eltype(arr::Periodic{T}) where T = T
Base.iterate(arr::Periodic,state=1) = Base.iterate(arr.data,state)
Base.lastindex(arr::Periodic) = lastindex(arr.data)
Base.lastindex(arr::Periodic,d) = lastindex(arr.data,d)
Base.copy(arr::Periodic) = Periodic(copy(arr.data));
Base.deepcopy(arr::Periodic) = Periodic(deepcopy(arr.data));
Base.similar(arr::Periodic) = Periodic(similar(arr.data))
Base.convert(::Type{Periodic{A,N}},B::Periodic{C,N}) where {A,C,N} = Periodic(convert(Array{A,N},B.data))
Base.repeat(arr::Periodic,i::Integer...) = Periodic(repeat(arr.data,i...))
Base.checkbounds(arr::Periodic,I...) = true;
#Base.circshift(arr::Periodic,shift=1) = Periodic(circshift(arr.data,shift))
function Base.circshift!(dest,orig,shiftamt)
    circshift!(dest.data,orig.data,shiftamt);
    dest;
end
