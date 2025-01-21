"""
    find_groundstate(ψ₀, H, [environments]; kwargs...) -> (ψ, environments, ϵ)
    find_groundstate(ψ₀, H, algorithm, environments) -> (ψ, environments, ϵ)

Compute the groundstate for Hamiltonian `H` with initial guess `ψ`. If not specified, an
optimization algorithm will be attempted based on the supplied keywords.

## Arguments
- `ψ₀::AbstractMPS`: initial guess
- `H::AbstractMPO`: operator for which to find the groundstate
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information

## Returns
- `ψ::AbstractMPS`: converged groundstate
- `environments`: environments corresponding to the converged state
- `ϵ::Float64`: final convergence error upon terminating the algorithm
"""
function find_groundstate(ψ::AbstractMPS, H,
                          envs::AbstractMPSEnvironments=environments(ψ, H);
                          tol=Defaults.tol, maxiter=Defaults.maxiter,
                          verbosity=Defaults.verbosity, trscheme=nothing)
    if isa(ψ, InfiniteMPS)
        alg = VUMPS(; tol=max(1e-4, tol), verbosity, maxiter)
        if tol < 1e-4
            alg = alg &
                  GradientGrassmann(; tol=tol, maxiter, verbosity)
        end
        if !isnothing(trscheme)
            alg = IDMRG2(; tol=min(1e-2, 100tol), verbosity,
                         trscheme) & alg
        end
    elseif isa(ψ, AbstractFiniteMPS)
        alg = DMRG(; tol, maxiter, verbosity)
        if !isnothing(trscheme)
            alg = DMRG2(; tol=min(1e-2, 100tol), verbosity, trscheme) & alg
        end
    else
        throw(ArgumentError("Unknown input state type"))
    end
    return find_groundstate(ψ, H, alg, envs)
end
