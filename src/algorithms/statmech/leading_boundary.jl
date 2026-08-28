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

- `tol::Float64`: convergence tolerance, compared against the convergence entry of the returned
    `info` (see Returns below). Which quantity that is depends on the algorithm
- `maxiter::Int`: maximum amount of iterations
- `verbosity::Int`: display progress information

# Returns

- `ψ::AbstractMPS`: converged leading boundary MPS
- `environments`: environments corresponding to the converged boundary
- `info::AlgorithmInfo`: how the algorithm terminated; `info.converged` says whether it got there.
    The quantity compared against `tol` is stored under a key naming which measure it is, and is
    never a truncation error:
    - [`VUMPS`](@ref) and [`VOMPS`](@ref) report `galerkin`, the maximum over sites of the
      local update projected onto the orthogonal complement of the current tensor.
    - [`GradientGrassmann`](@ref) reports `gradientnorm` from its optimiser.
    - [`IDMRG`](@ref) and [`IDMRG2`](@ref) report `bondresidual`, the change in the center bond
      tensor over a sweep. For `Multiline` methods this is extensive in the number of rows.

    [`convergence_measure`](@ref) returns whichever is present, for code that only wants the
    number.

    See [`AlgorithmInfo`](@ref), [`find_groundstate`](@ref), and the manual on the `ϵ` convention
    under [The error convention](@ref) and [Ground state accuracy](@ref).
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
