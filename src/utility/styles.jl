"""
    abstract type OperatorStyle
    OperatorStyle(x)
    OperatorStyle(::Type{T})

Trait to describe the operator behavior of the input `x` or type `T`, which can be either
*   `MPOStyle()`: product of local factors;
*   `HamiltonianStyle()`: sum of local terms.
"""
abstract type OperatorStyle end
OperatorStyle(x) = OperatorStyle(typeof(x))
OperatorStyle(T::Type) = throw(MethodError(OperatorStyle, T)) # avoid stackoverflow if not defined
OperatorStyle(x::OperatorStyle) = x

OperatorStyle(x, y) = OperatorStyle(OperatorStyle(x)::OperatorStyle, OperatorStyle(y)::OperatorStyle)
OperatorStyle(::T, ::T) where {T <: OperatorStyle} = T()
OperatorStyle(x::OperatorStyle, y::OperatorStyle) = error("Unknown combination of operator styles $x and $y")
@inline OperatorStyle(x, y, zs...) = OperatorStyle(OperatorStyle(x, y), zs...)

struct MPOStyle <: OperatorStyle end
struct HamiltonianStyle <: OperatorStyle end

@doc (@doc OperatorStyle) MPOStyle
@doc (@doc OperatorStyle) HamiltonianStyle

"""
    abstract type GeometryStyle
    GeometryStyle(x)
    GeometryStyle(::Type{T})

Trait to describe the geometry of the input `x` or type `T`, which can be either
*   `FiniteChainStyle()`: object is defined on a finite chain;
*   `InfiniteChainStyle()`: object is defined on an infinite chain.
"""
abstract type GeometryStyle end
GeometryStyle(x) = GeometryStyle(typeof(x))
GeometryStyle(T::Type) = throw(MethodError(GeometryStyle, T)) # avoid stackoverflow if not defined
GeometryStyle(x::GeometryStyle) = x

GeometryStyle(x, y) = GeometryStyle(GeometryStyle(x)::GeometryStyle, GeometryStyle(y)::GeometryStyle)
GeometryStyle(::T, ::T) where {T <: GeometryStyle} = T()
GeometryStyle(x::GeometryStyle, y::GeometryStyle) = error("Unknown combination of geometry styles $x and $y")
@inline GeometryStyle(x, y, zs...) = GeometryStyle(GeometryStyle(x, y), zs...)

struct FiniteChainStyle <: GeometryStyle end
struct InfiniteChainStyle <: GeometryStyle end

@doc (@doc GeometryStyle) FiniteChainStyle
@doc (@doc GeometryStyle) InfiniteChainStyle
