"""
$(TYPEDEF)

Object that represents multiple lines of objects of type `T`. Typically used to represent
multiple lines of `InfiniteMPS` (`MultilineMPS`) or `InfiniteMPO` (`MultilineMPO`).

`Multiline` plays two different, orthogonal roles at once, and its Base overloads are split
accordingly:

- As a sequence of lines, matching what is actually stored: `length`, `eltype`, `iterate`
  and `m[i]` (a single integer index) all refer to the `T`-typed lines themselves, i.e.
  `length(m) == nrows` and `m[i]::T`.
- As a lattice, describing the 2D shape spanned by the lines together: `size(m)` is
  `(nrows, ncols)`, and `axes`/`eachindex` follow `size`.

These two views disagree on purpose (`length(m) != prod(size(m))`).
Code that wants to work line-by-line should use `m[i]`/`parent(m)`,
while code that wants the lattice shape should use `size`.

# Fields

- `data::PeriodicArray{T, 1}`: the data of the multiline object

# See also

[`MultilineMPS`](@ref) and [`MultilineMPO`](@ref)
"""
struct Multiline{T}
    data::PeriodicArray{T, 1}
    function Multiline{T}(data::AbstractVector{T}) where {T}
        # @assert allequal(length.(data)) "All lines must have the same length"
        return new{T}(data)
    end
end
Multiline(data::AbstractVector{T}) where {T} = Multiline{T}(data)

# AbstractArray interface
# -----------------------
Base.parent(m::Multiline) = m.data
Base.size(m::Multiline) = (length(parent(m)), length(parent(m)[1]))
function Base.size(m::Multiline, i::Int) # acts like abstract array
    return i == 1 ? length(parent(m)) : i == 2 ? length(parent(m)[1]) : 1
end
Base.length(m::Multiline) = length(parent(m))
function Base.axes(m::Multiline, d::Int)
    return d <= 2 ? axes(m)[d] : Base.OneTo(1) # matches size
end
Base.eachindex(m::Multiline) = CartesianIndices(size(m))
Base.isfinite(m::Multiline) = isfinite(typeof(m))
Base.isfinite(::Type{Multiline{T}}) where {T} = isfinite(T)
Base.eltype(::Type{Multiline{T}}) where {T} = T

eachsite(m::Multiline) = eachsite(first(parent(m)))

Base.getindex(m::Multiline, i::Int) = getindex(parent(m), i)
Base.setindex!(m::Multiline, v, i::Int) = (setindex!(parent(m), v, i); m)

Base.copy(m::Multiline) = Multiline(map(copy, parent(m)))
Base.iterate(m::Multiline, args...) = iterate(parent(m), args...)

# Utility functions
# -----------------
Base.circshift(A::Multiline, n::Int) = Multiline(circshift(parent(A), n))
function Base.circshift(A::Multiline, shifts::Tuple{Int, Int})
    data′ = circshift.(parent(A), shifts[2])
    return Multiline(circshift!(data′, shifts[1]))
end
Base.reverse(A::Multiline) = Multiline(reverse(parent(A)))
Base.only(A::Multiline) = only(parent(A))

function Base.repeat(A::Multiline, rows::Int, cols::Int)
    inner = map(Base.Fix2(repeat, cols), parent(A))
    outer = repeat(inner, rows)
    return Multiline(outer)
end

# Style
# ----------------

OperatorStyle(::Type{Multiline{T}}) where {T} = OperatorStyle(T)
GeometryStyle(::Type{Multiline{T}}) where {T} = GeometryStyle(T)

# VectorInterface
# ---------------
VectorInterface.scalartype(::Type{Multiline{T}}) where {T} = scalartype(T)

function VectorInterface.zerovector(x::Multiline, ::Type{S}) where {S <: Number}
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

# is it intentional that a nontrivial multilinemps of normalised rows never has norm 1?
function VectorInterface.inner(x::Multiline, y::Multiline)
    T = VectorInterface.promote_inner(x, y)
    init = zero(T)
    return sum(splat(inner), zip(parent(parent(x)), parent(parent(y))); init)
end

LinearAlgebra.norm(x::Multiline) = sqrt(real(inner(x, x)))

# TensorKit
#----------

site_type(::Type{Multiline{S}}) where {S} = site_type(S)
bond_type(::Type{Multiline{S}}) where {S} = bond_type(S)
site_type(st::Multiline) = site_type(typeof(st))
bond_type(st::Multiline) = bond_type(typeof(st))
TensorKit.sectortype(::Type{Multiline{T}}) where {T} = sectortype(T)
TensorKit.spacetype(::Type{Multiline{T}}) where {T} = spacetype(T)
TensorKit.storagetype(::Type{Multiline{T}}) where {T} = storagetype(T)
