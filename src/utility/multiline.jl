"""
    struct Multiline{T}

Object that represents multiple lines of objects of type `T`. Typically used to represent
multiple lines of `InfiniteMPS` (`MultilineMPS`) or MPO (`Multiline{<:AbstractMPO}`).

# Fields
- `data::PeriodicArray{T,1}`: the data of the multiline object

See also: [`MultilineMPS`](@ref) and [`MultilineMPO`](@ref)
"""
struct Multiline{T}
    data::PeriodicArray{T,1}
    function Multiline{T}(data::AbstractVector{T}) where {T}
        @assert allequal(length.(data)) "All lines must have the same length"
        return new{T}(data)
    end
end
Multiline(data::AbstractVector{T}) where {T} = Multiline{T}(data)

# AbstractArray interface
# -----------------------
Base.parent(m::Multiline) = m.data
Base.size(m::Multiline) = (length(parent(m)), length(parent(m)[1]))
Base.size(m::Multiline, i::Int) = getindex(size(m), i)
Base.length(m::Multiline) = prod(size(m))
function Base.axes(m::Multiline, i::Int)
    return i == 1 ? axes(parent(m), 1) :
           i == 2 ? axes(parent(m)[1], 1) : throw(ArgumentError("Invalid index $i"))
end
Base.eachindex(m::Multiline) = CartesianIndices(size(m))

Base.getindex(m::Multiline, i::Int) = getindex(parent(m), i)
Base.setindex!(m::Multiline, v, i::Int) = (setindex!(parent(m), v, i); m)

Base.copy(m::Multiline) = Multiline(map(copy, parent(m)))
Base.iterate(m::Multiline, args...) = iterate(parent(m), args...)

# Utility functions
# -----------------
Base.circshift(A::Multiline, n::Int) = Multiline(circshift(parent(A), n))
function Base.circshift(A::Multiline, shifts::Tuple{Int,Int})
    data′ = circshift.(parent(A), shifts[2])
    return Multiline(circshift!(data′, shifts[1]))
end
Base.reverse(A::Multiline) = Multiline(reverse(parent(A)))
Base.only(A::Multiline) = only(parent(A))

function Base.repeat(A::Multiline, rows::Int, cols::Int)
    inner = map(Base.Fix2(repeat, cols), A.data)
    outer = repeat(inner, rows)
    return Multiline(outer)
end

# VectorInterface
# ---------------
VectorInterface.scalartype(::Type{Multiline{T}}) where {T} = scalartype(T)

function VectorInterface.zerovector(x::Multiline, ::Type{S}) where {S<:Number}
    return Multiline(zerovector.(parent(x), S))
end
VectorInterface.zerovector!(x::Multiline) = (zerovector!.(parent(x)); x)

function VectorInterface.scale(x::Multiline, α::Number)
    return scale!(zerovector(x, VectorInterface.promote_scale(x, α)), x, α)
end

function VectorInterface.scale!(x::Multiline, α::Number)
    scale!.(parent(x), α)
    return x
end
VectorInterface.scale!!(x::Multiline, α::Number) = scale!(x, α)

function VectorInterface.scale!(x::Multiline, x′::Multiline, α::Number)
    scale!.(parent(x), parent(x′), α)
    return x
end

VectorInterface.scale!!(x::Multiline, x′::Multiline, α::Number) = scale!(x, x′, α)

function VectorInterface.add(x::Multiline, y::Multiline, α::Number, β::Number)
    z = zerovector(x, VectorInterface.promote_add(x, y, α, β))
    return add!(scale!(z, x, β), y, α)
end

function VectorInterface.add!(x::Multiline, y::Multiline, α::Number, β::Number)
    add!.(parent(x), parent(y), α, β)
    return x
end

VectorInterface.add!!(x::Multiline, y::Multiline, α::Number, β::Number) = add!(x, y, α, β)

function VectorInterface.inner(x::Multiline, y::Multiline)
    T = VectorInterface.promote_inner(x, y)
    init = zero(T)
    return sum(inner, parent(parent(x)), parent(parent(y)); init)
end

LinearAlgebra.norm(x::Multiline) = norm(norm.(parent(x)))
