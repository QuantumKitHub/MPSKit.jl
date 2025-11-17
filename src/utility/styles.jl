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

struct FiniteChainStyle <: GeometryStyle end
struct InfiniteChainStyle <: GeometryStyle end

@doc (@doc GeometryStyle) FiniteChainStyle
@doc (@doc GeometryStyle) InfiniteChainStyle
