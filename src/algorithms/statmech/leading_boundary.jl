@doc """
    leading_boundary(ψ₀, O, [environments]; kwargs...) -> (ψ, environments, info)
    leading_boundary(ψ₀, O, algorithm, environments) -> (ψ, environments, info)

Compute the leading boundary MPS for operator `O` with initial guess `ψ`. If not specified, an
optimization algorithm will be attempted based on the supplied keywords.

# Arguments

- `ψ₀::AbstractMPS`: initial guess
- `O::AbstractMPO`: operator for which to find the leading_boundary
- `[environments]`: MPS environment manager
- `algorithm`: optimization algorithm

# Keyword Arguments

- `tol::Float64`: convergence tolerance, compared against the Galerkin error (see Returns below)
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information

# Returns

- `ψ::AbstractMPS`: converged leading boundary MPS
- `environments`: environments corresponding to the converged boundary
- `info::AlgorithmInfo`: how the algorithm terminated; `info.normres` is the Galerkin error compared
    against `tol`, and `info.converged` whether it got there. It is not a truncation error. See
    [`AlgorithmInfo`](@ref), [`find_groundstate`](@ref), and the manual on the `ϵ` convention under
    [The error convention](@ref) and [Ground state accuracy](@ref).
""" leading_boundary

# TODO: alg selector

# implementation always in terms of Multiline objects
function leading_boundary(state::InfiniteMPS, operator::InfiniteMPO, alg)
    state_multi = convert(MultilineMPS, state)
    operator_multi = convert(MultilineMPO, operator)
    state_multi′, envs_multi′, err = leading_boundary(
        state_multi, operator_multi, alg,
    )
    state′ = convert(InfiniteMPS, state_multi′)
    return state′, only(envs_multi′), err
end
function leading_boundary(state::InfiniteMPS, operator::InfiniteMPO, alg, envs)
    state_multi = convert(MultilineMPS, state)
    operator_multi = convert(MultilineMPO, operator)
    envs_multi = Multiline([envs])
    state_multi′, envs_multi′, err = leading_boundary(
        state_multi, operator_multi, alg,
        envs_multi
    )
    state′ = convert(InfiniteMPS, state_multi′)
    return state′, only(envs_multi′), err
end
