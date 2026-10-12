# Finite-MPS sweeps for `linear_solve`. Convergence is measured, as in `find_groundstate!`, by the
# local residual of the original system `(a₀ + a₁·A)·x = b`, evaluated before each local solve.

sweep_sites(::DMRGSolve) = 1
sweep_sites(::DMRGSolve2) = 2

# environments beyond `⟨x|A|x⟩`: `LeastSquares` also needs `A²` and the mixed sandwich `⟨x|A|b⟩`
linear_solve_environments(::Galerkin, x, A, b, openvs) = (; openvs, rhsenvs = environments(x, b))
function linear_solve_environments(::LeastSquares, x, A, b, openvs)
    Asq, sqenvs = squaredenvs(x, A, openvs)
    return (; openvs, rhsenvs = environments(x, b), Asq, sqenvs, abenvs = environments(x, A, b))
end

# `⟨∂x|A|b⟩` through the generic MPO derivative, since the JordanMPO one assumes `below === above`
function mixed_projection(alg, pos, x, A, b, envs; backend, allocator)
    W = A isa WindowMPOHamiltonian ? A.finite_ham : A
    GL = leftenv(envs, pos, x; backend, allocator)
    if sweep_sites(alg) == 1
        GR = rightenv(envs, pos, x; backend, allocator)
        return MPO_AC_Hamiltonian(GL, W[pos], GR, backend, allocator) * b.AC[pos]
    else
        GR = rightenv(envs, pos + 1, x; backend, allocator)
        return MPO_AC2_Hamiltonian(GL, W[pos], W[pos + 1], GR, backend, allocator) * AC2(b, pos)
    end
end

function local_hamiltonian(alg, pos, x, A, envs; kwargs...)
    return sweep_sites(alg) == 1 ? AC_hamiltonian(pos, x, A, x, envs; kwargs...) :
        AC2_hamiltonian(pos, x, A, x, envs; kwargs...)
end

# the local system `(α₀ + α₁·M)·v = rhs` actually handed to the Krylov solver
local_linear_system(::Galerkin, alg, pos, x, A, b, ctx, A_eff, b_eff, a₀, a₁; kwargs...) =
    (A_eff, b_eff, a₀, a₁)
function local_linear_system(::LeastSquares, alg, pos, x, A, b, ctx, A_eff, b_eff, a₀, a₁; kwargs...)
    Asq_eff = local_hamiltonian(alg, pos, x, ctx.Asq, ctx.sqenvs; kwargs...)
    N_eff = LinearCombination((A_eff, Asq_eff), (2 * real(conj(a₀) * a₁), abs2(a₁)))
    Ab_eff = mixed_projection(alg, pos, x, A, b, ctx.abenvs; kwargs...)
    return N_eff, conj(a₀) * b_eff + conj(a₁) * Ab_eff, abs2(a₀), one(a₁)
end

# returns the updated local tensor and the original-system residual before the update
function local_linear_solve(alg, pos, x, state; kind = :ACAR, adapt_kwargs...)
    A, b, ctx, (a₀, a₁) = state.operator, state.rhs, state.context, state.shift
    backend, allocator = alg.backend, state.allocator
    onesite = sweep_sites(alg) == 1
    A_eff = local_hamiltonian(alg, pos, x, A, ctx.openvs; backend, allocator)
    b_eff = onesite ? AC_projection(pos, x, b, ctx.rhsenvs; backend, allocator) :
        AC2_projection(pos, x, b, ctx.rhsenvs; backend, allocator)
    v = onesite ? x.AC[pos] : AC2(x, pos; kind)
    # `‖(a₀ + a₁·A)·x − b‖`, reusing the buffer of `A·x`
    res = norm(add!!(add!!(A_eff * v, v, a₀, a₁), b_eff, -1))
    M, rhs, α₀, α₁ = local_linear_system(
        alg.formulation, alg, pos, x, A, b, ctx, A_eff, b_eff, a₀, a₁; backend, allocator
    )
    v′, info = KrylovKit.linsolve(M, rhs, v, adapt_solver(alg.solver; adapt_kwargs...), α₀, α₁)
    # only warn when the local solve also misses the accuracy the sweep needs
    info.converged == 0 && info.normres > alg.tol * state.normb &&
        @warn "linear_solve: local solve at $pos did not converge (normres = $(info.normres))"
    return v′, res
end

function local_update!(pos, direction, x, alg::DMRGSolve, state, g_global, iter)
    AC′, res = local_linear_solve(alg, pos, x, state; iter, g_global)
    x.AC[pos] = AC′
    return x, res, 0.0
end
function local_update!(pos, direction, x, alg::DMRGSolve2, state, g_global, iter)
    kind = direction === Val(:right) ? :ACAR : :ALAC
    eps_trunc = state.truncation_errors[pos]
    AC2′, res = local_linear_solve(alg, pos, x, state; kind, iter, g_global, eps_trunc)
    x, ϵ_trunc = gauge2!(x, pos, direction, AC2′, alg.alg_gauge; normalize = false)
    return x, res, ϵ_trunc
end

# Internal state of the finite linear-solve sweeps, where `ϵ` is the largest local residual
# relative to `‖b‖`
struct LinearSolveState{S, O, B, E, C, T, A}
    mps::S
    operator::O
    rhs::B
    envs::E
    context::C
    shift::T
    normb::Float64
    iter::Int
    ϵ::Float64
    residuals::Vector{Float64}
    truncation_errors::Vector{Float64}
    allocator::A
end

function LinearSolveState(x, A, b, alg::Union{DMRGSolve, DMRGSolve2}, envs, a₀, a₁)
    n = num_updates(alg, x)
    context = linear_solve_environments(alg.formulation, x, A, b, envs)
    allocator = default_allocator(x, SerialScheduler())
    # guard the relative residual against a zero right-hand side
    normb = norm(b)
    iszero(normb) && (normb = one(normb))
    return LinearSolveState(
        x, A, b, envs, context, (a₀, a₁), normb, 0, 1.0, ones(n), zeros(n), allocator
    )
end

function sweep!(
        it::IterativeSolver{<:Union{DMRGSolve, DMRGSolve2}}, state::LinearSolveState,
        direction, iter
    )
    fwd, bwd = sweep_ranges(it.alg, state.mps)
    x = state.mps
    for pos in (direction === Val(:right) ? fwd : bwd)
        # the next local solver uses the residuals accumulated so far, as in `DMRG`
        g_global = maximum(state.residuals) * state.normb
        x, res, state.truncation_errors[pos] = local_update!(
            pos, direction, x, it.alg, state, g_global, iter
        )
        state.residuals[pos] = res / state.normb
    end
    return setproperties(state; mps = x, ϵ = maximum(state.residuals))
end

function Base.iterate(
        it::IterativeSolver{<:Union{DMRGSolve, DMRGSolve2}}, state::LinearSolveState
    )
    iter = state.iter + 1
    state = sweep!(it, state, Val(:right), iter)
    state = sweep!(it, state, Val(:left), iter)
    x, envs = it.finalize(
        iter, state.mps, state.operator, state.envs
    )::Tuple{typeof(state.mps), typeof(state.envs)}
    it.state = setproperties(state; mps = x, envs, iter)
    return (x, envs, state.ϵ), it.state
end

# the residual cannot drop below the level set by the discarded weight
sweep_converged(alg, state::LinearSolveState) =
    state.ϵ <= max(alg.tol, maximum(state.truncation_errors) / state.normb)

function linear_solve!(
        x::AbstractFiniteMPS, A, b, alg::Union{DMRGSolve, DMRGSolve2},
        envs = environments(x, A, x); a₀ = 0, a₁ = 1
    )
    log = IterLog(alg)
    it = IterativeSolver(alg, LinearSolveState(x, A, b, alg, envs, a₀, a₁))

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, it.ϵ)
        for (_, _, ϵ) in Iterators.take(it, alg.maxiter)
            if sweep_converged(alg, it.state)
                @log_convergence logfinish!(log, it.iter, ϵ)
                break
            elseif it.iter == alg.maxiter
                @log_nonconvergence logcancel!(log, it.iter, ϵ)
            else
                @log_iteration logiter!(log, it.iter, ϵ)
            end
        end
    end

    state = it.state
    info = AlgorithmInfo(;
        converged = sweep_converged(alg, state), galerkin = state.ϵ, numiter = state.iter,
        truncation_errors = alg isa DMRGSolve2 ? copy(state.truncation_errors) : nothing,
    )
    return state.mps, state.envs, info
end
