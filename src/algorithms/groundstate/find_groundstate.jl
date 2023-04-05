"""
    find_groundstate(Ψ, H, [environments]; kwargs...)
    find_groundstate(Ψ, H, algorithm, environments)

Compute the groundstate for Hamiltonian `H` with initial guess `Ψ`. If not specified, an
optimization algorithm will be attempted based on the supplied keywords.

## Arguments
- `Ψ::AbstractMPS`: initial guess
- `H::AbstractMPO`: operator for which to find the groundstate
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbose::Bool`: display progress information
"""
function find_groundstate(Ψ::AbstractMPS, H, envs=environments(Ψ, H);
                          tol=Defaults.tol, maxiter=Defaults.maxiter,
                          verbose=Defaults.verbose, trscheme=nothing)
    if isa(Ψ, InfiniteMPS)
        alg = VUMPS(; tol_galerkin=max(1e-4, tol), verbose=verbose, maxiter=maxiter)
        tol < 1e-4 && alg = alg & GradientGrassmann(; tol=tol, maxiter=maxiter,
                                                    verbosity=verbose ? 2 : 0)
        isnothing(trscheme) || alg = IDMRG2(; tol_galerkin=min(1e-2, 100tol),
                                            verbose=verbose,
                                            trscheme=trscheme) & alg
    elseif isa(Ψ, AbstractFiniteMPS)
        alg = DMRG(; tol=tol, maxiter=maxiter, verbose=verbose)
        isnothing(trschemer) || alg = DMRG2(; tol=min(1e-2, 100tol), verbose=verbose,
                                            trscheme=trscheme) & alg
    else
        throw(ArgumentError("Unknown input state type"))
    end
    return find_groundstate(Ψ, H, alg, envs)
end
