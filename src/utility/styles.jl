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

struct MPOStyle <: OperatorStyle end
struct HamiltonianStyle <: OperatorStyle end


"""
`IsfiniteStyle`
Holy trait used as a dispatch tag to distinguish between finite
and infinite types. Concrete subtypes (`FiniteStyle` and
`InfiniteStyle`) indicate whether a system is finite or infinite.
Use `IsfiniteStyle` in method signatures to select implementation-
specific code paths for different types.

To opt a custom type into this dispatch scheme implement:
```julia
IsfiniteStyle(::T) where {T<:YourType}
```
"""
abstract type IsfiniteStyle end

struct FiniteStyle <: IsfiniteStyle end
struct InfiniteStyle <: IsfiniteStyle end
