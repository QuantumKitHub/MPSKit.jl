"""
    abstract type OperatorStyle end

Holy trait used as a dispatch tag for operator representations.
Concrete subtypes (`MPOStyle` and `HamiltonianStyle`) indicate
whether an operator represents a Hamiltonian operator (sum of terms)
or a transfer matrix (product of factors).

To opt a custom operator type into this dispatch scheme implement:
```julia
OperatorStyle(::T) where {T <: YourOperatorType}
```
"""
abstract type OperatorStyle end
OperatorStyle(x) = OperatorStyle(typeof(x))
OperatorStyle(T::Type) = throw(MethodError(OperatorStyle, T)) # avoid stackoverflow if not defined

struct MPOStyle <: OperatorStyle end
struct HamiltonianStyle <: OperatorStyle end


"""
    abstract type GeometryStyle end

Holy trait used as a dispatch tag to distinguish between different
geometries Concrete subtypes
(`FiniteChainStyle`, `InfiniteChainStyle`) indicate whether a system is
a finite or infinite chain.

To opt a custom type into this dispatch scheme implement:
```julia
GeometryStyle(::T) where {T <: YourType}
```
"""
abstract type GeometryStyle end
GeometryStyle(x) = GeometryStyle(typeof(x))
GeometryStyle(T::Type) = throw(MethodError(GeometryStyle, T)) # avoid stackoverflow if not defined

struct FiniteChainStyle <: GeometryStyle end
struct InfiniteChainStyle <: GeometryStyle end
