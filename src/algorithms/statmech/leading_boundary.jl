
@doc """
    leading_boundary(ψ, O, [environments]; kwargs...)
    leading_boundary(ψ, O, algorithm, environments)

Compute the leading boundary MPS for operator `O` with initial guess `ψ`. If not specified, an
optimization algorithm will be attempted based on the supplied keywords.

## Arguments
- `ψ::AbstractMPS`: initial guess
- `O::AbstractMPO`: operator for which to find the leading_boundary
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information
""" leading_boundary

# TODO: alg selector

# implementation always in terms of Multiline objects
function leading_boundary(state::InfiniteMPS, operator::InfiniteMPO, alg,
                          envs=environments(state, operator))
    state_multi = convert(MultilineMPS, state)
    operator_multi = convert(MultilineMPO, operator)
    envs_multi = Multiline([envs])
    state_multi′, envs_multi′, err = leading_boundary(state_multi, operator_multi, alg,
                                                      envs_multi)
    state′ = convert(InfiniteMPS, state_multi′)
    return state′, envs, err
end
