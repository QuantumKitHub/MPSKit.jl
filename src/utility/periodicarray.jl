"""
    PeriodicArray{T,N} <: AbstractArray{T,N}

Array wrapper with periodic boundary conditions.

# Fields
- `data::Array{T,N}`: the data of the array

# Examples
```jldoctest
A = PeriodicArray([1, 2, 3])
A[0], A[2], A[4]

# output

(3, 2, 1)
```
```jldoctest
A = PeriodicArray([1 2; 3 4])
A[-1, 1], A[1, 1], A[4, 5]

# output

(1, 1, 3)
```

See also [`PeriodicVector`](@ref), [`PeriodicMatrix`](@ref)
"""
struct PeriodicArray{T,N} <: AbstractArray{T,N}
    data::Array{T,N}
end
PeriodicArray(data::AbstractArray{T,N}) where {T,N} = PeriodicArray{T,N}(data)
function PeriodicArray{T}(initializer, args...) where {T}
    return PeriodicArray(Array{T}(initializer, args...))
end
function PeriodicArray{T,N}(initializer, args...) where {T,N}
    return PeriodicArray(Array{T,N}(initializer, args...))
end

const PeriodicVector{T} = PeriodicArray{T,1}
const PeriodicMatrix{T} = PeriodicArray{T,2}

Base.parent(A::PeriodicArray) = A.data

# AbstractArray interface
# -----------------------
Base.size(A::PeriodicArray) = size(parent(A))

function Base.getindex(A::PeriodicArray{T,N}, I::Vararg{Int,N}) where {T,N}
    @inbounds getindex(parent(A), map(mod1, I, size(A))...)
end
function Base.setindex!(A::PeriodicArray{T,N}, v, I::Vararg{Int,N}) where {T,N}
    @inbounds setindex!(parent(A), v, map(mod1, I, size(A))...)
end

Base.checkbounds(A::PeriodicArray, I...) = true

Base.LinearIndices(A::PeriodicArray) = PeriodicArray(LinearIndices(parent(A)))
Base.CartesianIndices(A::PeriodicArray) = PeriodicArray(CartesianIndices(parent(A)))

function Base.similar(A::PeriodicArray, ::Type{S}, dims::Dims) where {S}
    return PeriodicArray(similar(parent(A), S, dims))
end
Base.copy(A::PeriodicArray) = PeriodicArray(copy(parent(A)))
function Base.copyto!(dst::PeriodicArray, src::PeriodicArray)
    copyto!(parent(dst), parent(src))
    return dst
end

# Broadcasting
# ------------
Base.BroadcastStyle(::Type{T}) where {T<:PeriodicArray} = Broadcast.ArrayStyle{T}()

function Base.similar(bc::Broadcast.Broadcasted{<:Broadcast.ArrayStyle{<:PeriodicArray}},
                      ::Type{T}) where {T}
    return PeriodicArray(similar(Array{T}, axes(bc)))
end

# Conversion
# ----------
Base.convert(::Type{T}, A::AbstractArray) where {T<:PeriodicArray} = T(A)
Base.convert(::Type{T}, A::PeriodicArray) where {T<:AbstractArray} = convert(T, parent(A))
# fix ambiguities
Base.convert(::Type{T}, A::PeriodicArray) where {T<:PeriodicArray} = A
Base.convert(::Type{T}, A::PeriodicArray) where {T<:Array} = parent(A)
