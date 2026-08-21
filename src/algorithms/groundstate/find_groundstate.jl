"""
    find_groundstate(ψ₀, H, [environments]; kwargs...) -> (ψ, environments, info)
    find_groundstate(ψ₀, H, algorithm, [environments]) -> (ψ, environments, info)

Compute the ground state for Hamiltonian `H` with initial guess `ψ₀`. If no `algorithm` is
specified, one is selected automatically from the type of `ψ₀` and the supplied keywords
(see the automatic-selection notes below).

# Arguments

- `ψ₀::AbstractMPS`: initial guess
- `H::AbstractMPO`: operator for which to find the ground state
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

# Keyword Arguments

- `tol::Float64 = $(Defaults.tol)`: tolerance for the convergence criterion
- `maxiter::Int = $(Defaults.maxiter)`: maximum number of iterations
- `verbosity::Int = $(Defaults.verbosity)`: display progress information
- `trunc = nothing`: if supplied, a truncation strategy that enables bond-dimension growth
  through a two-site algorithm (see below)

# Automatic algorithm selection

When no `algorithm` is passed, the choice depends on the type of `ψ₀`:
- `InfiniteMPS`: [`VUMPS`](@ref) (with its tolerance floored at `1e-4`), refined by
  [`GradientGrassmann`](@ref) when `tol < 1e-4`. If `trunc` is given, an [`IDMRG2`](@ref)
  stage is prepended to grow the bond dimension.
- `AbstractFiniteMPS`: [`DMRG`](@ref). If `trunc` is given, a [`DMRG2`](@ref) stage is
  prepended to grow the bond dimension.

Because single-site [`DMRG`](@ref) preserves the bond dimension of `ψ₀`, passing a
`trunc` (or an explicit two-site `algorithm`) is the usual way to converge from a
low-bond-dimension initial guess such as a product state.

# Returns

- `ψ::AbstractMPS`: converged ground state
- `environments`: environments corresponding to the converged state
- `info::AlgorithmInfo`: how the algorithm terminated. `info.normres` is the quantity compared
    against `tol` and `info.converged` says whether it got there. Which measure `normres` is depends
    on the algorithm. A truncating algorithm additionally fills `info.ϵ_max`/`info.ϵ_total`
    with what its final sweep discarded. See [`AlgorithmInfo`](@ref),
    and [The error convention](@ref) in the manual.

# Examples

Ground state of a 4-site transverse-field Ising chain, `H = -∑ XₖXₖ₊₁ - ∑ Zₖ`, starting
from a product state and letting `DMRG2` grow the bond dimension:

```jldoctest
julia> X = TensorMap(Float64[0 1; 1 0], ℂ^2, ℂ^2);

julia> Z = TensorMap(Float64[1 0; 0 -1], ℂ^2, ℂ^2);

julia> L = 4; lattice = fill(ℂ^2, L);

julia> H = FiniteMPOHamiltonian(lattice, ((i, i + 1) => -(X ⊗ X) for i in 1:(L - 1))) +
           FiniteMPOHamiltonian(lattice, ((i,) => -Z for i in 1:L));

julia> ψ₀ = FiniteMPS(ones(Float64, (ℂ^2)^L));

julia> ψ, envs, info = find_groundstate(ψ₀, H; verbosity = 0, trunc = truncrank(16));

julia> round(real(expectation_value(ψ, H)); digits = 4)
-4.7588
```
"""
function find_groundstate(
        ψ::AbstractMPS, H, envs::AbstractMPSEnvironments = environments(ψ, H, ψ);
        tol = Defaults.tol, maxiter = Defaults.maxiter,
        verbosity = Defaults.verbosity, trunc = nothing
    )
    if isa(ψ, InfiniteMPS)
        alg = VUMPS(; tol = max(1.0e-4, tol), verbosity, maxiter)
        if tol < 1.0e-4
            alg = alg & GradientGrassmann(; tol = tol, maxiter, verbosity)
        end
        if !isnothing(trunc)
            alg = IDMRG2(; tol = min(1.0e-2, 100tol), verbosity, trunc) & alg
        end
    elseif isa(ψ, AbstractFiniteMPS)
        alg = DMRG(; tol, maxiter, verbosity)
        if !isnothing(trunc)
            alg = DMRG2(; tol = min(1.0e-2, 100tol), verbosity, trunc) & alg
        end
    else
        throw(ArgumentError("Unknown input state type"))
    end
    return find_groundstate(ψ, H, alg, envs)
end
