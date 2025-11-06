# This file contains the definition of the IterativeSolver type and the solve! function.
# Attempts to remove as much of the boilerplate code as possible from the iterative solvers.

mutable struct IterativeSolver{A, B}
    alg::A
    state::B
end

function Base.getproperty(it::IterativeSolver{A, B}, name::Symbol) where {A, B}
    (name === :alg || name === :state) && return getfield(it, name)

    alg = getfield(it, :alg)
    name in propertynames(alg) && return getproperty(alg, name)

    state = getfield(it, :state)
    name in propertynames(state) && return getproperty(state, name)

    throw(ArgumentError("Field $name not found in IterativeSolver"))
end

Base.iterate(it::IterativeSolver) = iterate(it, it.state)
