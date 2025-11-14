"""
`OperatorStyle`

Holy trait used as a dispatch tag for operator representations.
Concrete subtypes (`MPOStyle` and `HamiltonianStyle`) indicate
whether an operator is stored as an MPO or as a Hamiltonian.
Use `OperatorStyle` in method signatures to select implementation-
specific code paths for different operator types.

To opt a custom operator type into this dispatch scheme implement:
```julia
OperatorStyle(::T) where {T<:YourOperatorType}
```
"""
abstract type OperatorStyle end
OperatorStyle(x) = OperatorStyle(typeof(x))
OperatorStyle(T::Type) = throw(MethodError(OperatorStyle, T)) # avoid stackoverflow if not defined

struct MPOStyle <: OperatorStyle end
struct HamiltonianStyle <: OperatorStyle end


"""
`GeometryStyle`

Holy trait used as a dispatch tag to distinguish between different
geometry (currently finite and infinite). Concrete subtypes
(`FiniteStyle` and `InfiniteStyle`) indicate whether a system is
finite or infinite. Use `GeometryStyle` in method signatures to
select implementation-specific code paths for different types.

To opt a custom type into this dispatch scheme implement:
```julia
GeometryStyle(::T) where {T<:YourType}
```
"""
abstract type GeometryStyle end
GeometryStyle(x) = GeometryStyle(typeof(x))
GeometryStyle(T::Type) = throw(MethodError(GeometryStyle, T)) # avoid stackoverflow if not defined

struct FiniteStyle <: GeometryStyle end
struct InfiniteStyle <: GeometryStyle end
struct WindowStyle <: GeometryStyle end
