@doc """
    approximate(Ψ₀, (O, Ψ), algorithm, [environments]; kwargs...)
    approximate!(Ψ₀, (O, Ψ), algorithm, [environments]; kwargs...)

Compute an approximation to the application of an operator `O` to the state `Ψ` in the form
of an MPS `Ψ₀`.

## Arguments
- `Ψ₀::AbstractMPS`: initial guess of the approximated state
- `(O::AbstractMPO, Ψ::AbstractMPS)`: operator `O` and state `Ψ` to be approximated
- `algorithm`: approximation algorithm. See below for a list of available algorithms.
- `[environments]`: MPS environment manager

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbose::Bool`: display progress information

## Algorithms
- `DMRG`: Alternating least square method for maximizing the fidelity with a single-site scheme.
- `DMRG2`: Alternating least square method for maximizing the fidelity with a two-site scheme.

- `IDMRG1`: Variant of `DMRG` for maximizing fidelity density in the thermodynamic limit.
- `IDMRG2`: Variant of `DMRG2` for maximizing fidelity density in the thermodynamic limit.
- `VUMPS`: Tangent space method for truncating uniform MPS. See [SciPost:4.1.004](https://scipost.org/SciPostPhysCore.4.1.004). Also known as "VOMPS".
"""
approximate, approximate!
