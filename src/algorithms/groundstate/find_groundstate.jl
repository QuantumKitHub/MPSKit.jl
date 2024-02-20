"""
    find_groundstate(ψ, H, [environments]; kwargs...)
    find_groundstate(ψ, H, algorithm, environments)

Compute the groundstate for Hamiltonian `H` with initial guess `ψ`. If not specified, an
optimization algorithm will be attempted based on the supplied keywords.

## Arguments
- `ψ::AbstractMPS`: initial guess
- `H::AbstractMPO`: operator for which to find the groundstate
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbose::Bool`: display progress information
"""
function find_groundstate(ψ::AbstractMPS, H, envs::Cache=environments(ψ, H);
                          tol::Real=Defaults.tol, maxiter::Integer=Defaults.maxiter,
                          verbosity::Integer=Defaults.verbosity, trscheme=nothing,
                          verbose=nothing)
    verbosity = if !isnothing(verbose)
        Base.depwarn("find_groundstate(ψ, H; verbose=...) is deprecated. Use find_groundstate(ψ, H; verbosity=...) instead.",
                     :find_groundstate; force=true)
        verbose ? VERBOSE_ITER : VERBOSE_WARN
    else
        verbosity
    end
    if isa(ψ, InfiniteMPS)
        alg = VUMPS(; tol=max(1e-4, tol), verbosity, maxiter)
        if tol < 1e-4
            alg = alg &
                  GradientGrassmann(; tol, maxiter, verbosity)
        end
        if !isnothing(trscheme)
            alg = IDMRG2(; tol=min(1e-2, 100tol), verbosity, trscheme) & alg
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
