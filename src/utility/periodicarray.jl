struct PeriodicArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
PeriodicArray(a::PeriodicArray) = a;
function PeriodicArray{T}(initializer, args...) where {T}
    return PeriodicArray(Array{T}(initializer, args...))
end
function PeriodicArray{T,N}(initializer, args...) where {T,N}
    return PeriodicArray(Array{T,N}(initializer, args...))
end

Base.size(a::PeriodicArray) = size(a.data)
Base.size(a::PeriodicArray, i) = size(a.data, i)
Base.length(a::PeriodicArray) = length(a.data)
function Base.getindex(a::PeriodicArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @inbounds getindex(a.data, map(mod1, I, size(a.data))...)
end
function Base.setindex!(a::PeriodicArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    @inbounds setindex!(a.data, v, map(mod1, I, size(a.data))...)
end

Base.similar(a::PeriodicArray, ::Type{T}) where {T} = PeriodicArray(similar(a.data, T))
Base.similar(a::PeriodicArray, dims::Dims) = PeriodicArray(similar(a.data, dims))
function Base.similar(a::PeriodicArray, ::Type{T}, dims::Dims) where {T}
    return PeriodicArray(similar(a.data, T, dims))
end

Base.copy(a::PeriodicArray) = PeriodicArray(copy(a.data))
function Base.copyto!(dst::PeriodicArray, src::PeriodicArray)
    copyto!(dst.data, src.data)
    return dst
end
# not necessary but maybe more efficient
function Base.convert(::Type{PeriodicArray{T}}, a::PeriodicArray) where {T}
    return PeriodicArray(convert(Array{T}, a.data))
end
function Base.convert(::Type{PeriodicArray{T,N}}, a::PeriodicArray) where {T,N}
    return PeriodicArray(convert(Array{T,N}, a.data))
end

#should this copy?
Base.convert(::Type{Array{T,N}}, a::PeriodicArray{T,N}) where {T,N} = a.data;

Base.checkbounds(a::PeriodicArray, I...) = true

function Base.circshift(t::PeriodicArray{T,N}, tup::Tuple{Vararg{Integer,N}}) where {T,N}
    return PeriodicArray{T,N}(circshift(t.data, tup))
end
function Base.repeat(t::PeriodicArray, args::Vararg{Integer,N} where {N})
    return PeriodicArray(repeat(t.data, args...))
end
Base.BroadcastStyle(::Type{T}) where {T<:PeriodicArray} = Broadcast.ArrayStyle{T}()
function Base.similar(
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{T}}, ::Type{Elt}
) where {T<:PeriodicArray,Elt}
    return PeriodicArray(similar(Array{Elt}, axes(bc)))
end;
