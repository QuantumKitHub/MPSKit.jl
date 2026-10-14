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

# iteration logs are named after their algorithm unless the algorithm specifies otherwise
IterativeLoggers.IterLog(alg::Algorithm) = IterLog(string(nameof(typeof(alg))))

# Finite sweeps update `sweep_sites(alg)` neighbouring sites at once: 1 for a site, 2 for a bond.
function sweep_sites end

# number of distinct local updates in a sweep: all sites, or all bonds
num_updates(alg, ψ) = length(ψ) - sweep_sites(alg) + 1

# forward and backward positions of a sweep, without updating the turning point twice
function sweep_ranges(alg, ψ)
    L = length(ψ)
    return sweep_sites(alg) == 1 ? (1:(L - 1), L:-1:2) : (1:(L - 1), (L - 2):-1:1)
end
