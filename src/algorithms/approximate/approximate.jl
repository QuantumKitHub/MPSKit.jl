@doc """
    approximate(ψ₀, (O, ψ), algorithm, [environments]; kwargs...)
    approximate!(ψ₀, (O, ψ), algorithm, [environments]; kwargs...)

Compute an approximation to the application of an operator `O` to the state `ψ` in the form
of an MPS `ψ₀`.

## Arguments
- `ψ₀::AbstractMPS`: initial guess of the approximated state
- `(O::AbstractMPO, ψ::AbstractMPS)`: operator `O` and state `ψ` to be approximated
- `algorithm`: approximation algorithm. See below for a list of available algorithms.
- `[environments]`: MPS environment manager

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information

## Algorithms
- `DMRG`: Alternating least square method for maximizing the fidelity with a single-site scheme.
- `DMRG2`: Alternating least square method for maximizing the fidelity with a two-site scheme.

- `IDMRG1`: Variant of `DMRG` for maximizing fidelity density in the thermodynamic limit.
- `IDMRG2`: Variant of `DMRG2` for maximizing fidelity density in the thermodynamic limit.
- `VOMPS`: Tangent space method for truncating uniform MPS.
"""
approximate, approximate!

# implementation in terms of Multiline
function approximate(ψ::InfiniteMPS,
                     toapprox::Tuple{<:InfiniteMPO,<:InfiniteMPS},
                     algorithm,
                     envs=environments(ψ, toapprox))
    envs′ = Multiline([envs])
    multi, envs = approximate(convert(MultilineMPS, ψ),
                              (convert(MultilineMPO, toapprox[1]),
                               convert(MultilineMPS, toapprox[2])), algorithm, envs′)
    ψ = convert(InfiniteMPS, multi)
    return ψ, envs
end
