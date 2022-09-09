struct PeriodicArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
PeriodicArray(a::PeriodicArray) = a;
PeriodicArray{T}(initializer, args...) where T =
    PeriodicArray(Array{T}(initializer, args...))
PeriodicArray{T,N}(initializer, args...) where {T,N} =
    PeriodicArray(Array{T,N}(initializer, args...))

Base.size(a::PeriodicArray) = size(a.data)
Base.size(a::PeriodicArray,i) = size(a.data,i)
Base.length(a::PeriodicArray) = length(a.data)
Base.getindex(a::PeriodicArray{T,N}, I::Vararg{Int,N}) where {T,N} =
    @inbounds getindex(a.data, map(mod1, I, size(a.data))...)
Base.setindex!(a::PeriodicArray{T,N}, v, I::Vararg{Int,N}) where {T,N} =
    @inbounds setindex!(a.data, v, map(mod1, I, size(a.data))...)

Base.similar(a::PeriodicArray, ::Type{T}) where T = PeriodicArray(similar(a.data,T))
Base.similar(a::PeriodicArray, dims::Dims) =
    PeriodicArray(similar(a.data, dims))
Base.similar(a::PeriodicArray, ::Type{T}, dims::Dims) where T =
    PeriodicArray(similar(a.data, T, dims))

Base.copy(a::PeriodicArray) = PeriodicArray(copy(a.data))
function Base.copyto!(dst::PeriodicArray,src::PeriodicArray)
    copyto!(dst.data,src.data);
    dst
end
# not necessary but maybe more efficient
Base.convert(::Type{PeriodicArray{T}}, a::PeriodicArray) where {T} =
    PeriodicArray(convert(Array{T}, a.data))
Base.convert(::Type{PeriodicArray{T,N}}, a::PeriodicArray) where {T,N} =
    PeriodicArray(convert(Array{T,N}, a.data))

#should this copy?
Base.convert(::Type{Array{T,N}},a::PeriodicArray{T,N}) where {T,N} = a.data;

Base.checkbounds(a::PeriodicArray, I...) = true

Base.circshift(t::PeriodicArray{T,N},tup::Tuple{Vararg{Integer,N}}) where{T,N}= PeriodicArray{T,N}(circshift(t.data,tup))
Base.repeat(t::PeriodicArray,args::Vararg{Integer,N} where N) = PeriodicArray(repeat(t.data,args...))
Base.BroadcastStyle(::Type{T}) where T <:PeriodicArray = Broadcast.ArrayStyle{T}()
Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}},::Type{Elt}) where {T<:PeriodicArray,Elt} = PeriodicArray(similar(Array{Elt},axes(bc))); 