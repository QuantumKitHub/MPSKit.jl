# Finite-MPS implementations of the linear solver `linsolve`.
#
# The sweep drivers mirror the variational structure of `find_groundstate!`/`corvector.jl`: at each
# site (single-site) or bond (two-site) we build the local effective operator from the operator
# environments and the local right-hand side from the overlap environments, solve a small local
# linear problem with `KrylovKit.linsolve`, and install the result.
#
# Convergence is measured, as in `find_groundstate!` (via `calc_galerkin`), by the local residual of
# the *original* system `(a₀ + a₁·A)·x = b`: at each site the norm of `(a₀ + a₁·A_eff)·AC − b_eff`
# (relative to `‖b‖`), evaluated on the current tensor before the local solve, maximized over the
# sweep. It is formed as a local tensor, so it resolves to machine precision even for
# shifted/resolvent systems (no `√eps` cancellation) and needs no MPS-level `+`/`-`.
#
# The local solves use adaptive *tolerances* by default: `adapt_solver` retunes the inner solver's
# tolerance per bond from the previous-sweep residual (`g_global`) — a `DynamicTol` wrapper (which
# works for any KrylovKit solver, `GMRES`/`CG`/`BiCGStab`/…, since it only sets `tol`). The Krylov
# budget is left fixed. For a plain (unwrapped) solver `adapt_solver` is the identity.

# Right-hand-side context
# -----------------------
# Bundles the environments a formulation needs beyond the operator sandwich `openvs`. `Galerkin`
# only needs the overlap `⟨x|b⟩`; `LeastSquares` additionally needs the squared operator (for
# `A²`) and the mixed sandwich `⟨x|A|b⟩` (for `A·b`).
_rhs_context(::Galerkin, x, A, b, openvs) = (; openvs, rhsenvs = environments(x, b))
function _rhs_context(::LeastSquares, x, A, b, openvs)
    Asq, sqenvs = squaredenvs(x, A, openvs)
    # the `A·b` term of the normal-equation right-hand side comes from the mixed sandwich
    # `⟨x|A|b⟩`, so `A·b` is never materialized as an MPS
    return (; openvs, rhsenvs = environments(x, b), Asq, sqenvs, abenvs = environments(x, A, b))
end

# `⟨∂x|A|b⟩`, the mixed-sandwich projection supplying the `A†b` term of the normal equations.
# `AC_hamiltonian(pos, x, A, b, envs)` cannot be used for an `MPOHamiltonian`: the JordanMPO
# effective operator assumes `below === above`, and teaching it to branch would make the return type
# of that hot function a union. The generic sparse MPO derivative handles the mixed case, and one
# application per bond makes its cost irrelevant. Window operators store the finite part in the
# environments, so unwrap them the same way the derivative constructors do.
_finite_operator(A) = A
_finite_operator(A::WindowMPOHamiltonian) = A.finite_ham

function _mixed_projection(::Val{1}, pos, x, A, b, envs)
    W = _finite_operator(A)
    H = MPO_AC_Hamiltonian(leftenv(envs, pos, x), W[pos], rightenv(envs, pos, x))
    return H * b.AC[pos]
end
function _mixed_projection(::Val{2}, pos, x, A, b, envs)
    W = _finite_operator(A)
    H = MPO_AC2_Hamiltonian(
        leftenv(envs, pos, x), W[pos], W[pos + 1], rightenv(envs, pos + 1, x)
    )
    return H * AC2(b, pos)
end

# KrylovKit's `linsolve` is silent on non-convergence, so check it here. `warn_tol` is the absolute
# residual the outer sweep actually needs (`alg.tol · ‖b‖`): an inner solve that undershoots its own
# adaptive tolerance but still beats that has done its job, and warning about it would fire on nearly
# every tightly-converged sweep (adaptive tolerances routinely dip below the round-off floor of the
# local problem).
function _warn_unconverged(info, pos, warn_tol)
    return info.converged == 0 && info.normres > warn_tol &&
        @warn "linsolve: local solve at $pos did not converge (normres = $(info.normres))"
end

# denominator for the relative residual; guards against a zero right-hand side
_rhs_norm(b) = (n = norm(b); iszero(n) ? one(n) : n)

# Local single-site solves. Each returns the updated center tensor and the local residual of the
# *original* system (relative-residual convergence uses `res/‖b‖`; adaptation uses the previous
# sweep's residual via `g_global`).
function _local_linsolve(::Galerkin, ::Val{1}, pos, x, A, b, ctx, solver, a₀, a₁; iter = 1, g_global = 0.0, warn_tol = 0.0)
    A_eff = AC_hamiltonian(pos, x, A, x, ctx.openvs)
    b_eff = AC_projection(pos, x, b, ctx.rhsenvs)
    AC = x.AC[pos]
    res = norm(a₀ * AC + a₁ * (A_eff * AC) - b_eff)
    AC′, info = KrylovKit.linsolve(A_eff, b_eff, AC, adapt_solver(solver; iter, g_global), a₀, a₁)
    _warn_unconverged(info, "site $pos", warn_tol)
    return AC′, res
end
function _local_linsolve(::LeastSquares, ::Val{1}, pos, x, A, b, ctx, solver, a₀, a₁; iter = 1, g_global = 0.0, warn_tol = 0.0)
    A_eff = AC_hamiltonian(pos, x, A, x, ctx.openvs)
    Asq_eff = AC_hamiltonian(pos, x, ctx.Asq, x, ctx.sqenvs)
    N_eff = LinearCombination((A_eff, Asq_eff), (2 * real(conj(a₀) * a₁), abs2(a₁)))
    b_eff = AC_projection(pos, x, b, ctx.rhsenvs)
    Ab_eff = _mixed_projection(Val(1), pos, x, A, b, ctx.abenvs)
    AC = x.AC[pos]
    # convergence uses the ORIGINAL-system residual, not the normal-equation residual
    res = norm(a₀ * AC + a₁ * (A_eff * AC) - b_eff)
    rhs = conj(a₀) * b_eff + conj(a₁) * Ab_eff
    AC′, info = KrylovKit.linsolve(N_eff, rhs, AC, adapt_solver(solver; iter, g_global), abs2(a₀), one(a₁))
    _warn_unconverged(info, "site $pos", warn_tol)
    return AC′, res
end

# Local two-site solves. Return the updated two-site tensor and the original-system residual.
# `eps_trunc` (the previous discarded weight at this bond) floors the adaptive tolerance so the
# local solve is not driven far below what the SVD truncation will discard.
function _local_linsolve(::Galerkin, ::Val{2}, pos, x, A, b, ctx, solver, a₀, a₁, kind; iter = 1, g_global = 0.0, eps_trunc = 0.0, warn_tol = 0.0)
    A_eff = AC2_hamiltonian(pos, x, A, x, ctx.openvs)
    b_eff = AC2_projection(pos, x, b, ctx.rhsenvs)
    ac2 = AC2(x, pos; kind)
    res = norm(a₀ * ac2 + a₁ * (A_eff * ac2) - b_eff)
    AC2′, info = KrylovKit.linsolve(A_eff, b_eff, ac2, adapt_solver(solver; iter, g_global, eps_trunc), a₀, a₁)
    _warn_unconverged(info, "bond $pos", warn_tol)
    return AC2′, res
end
function _local_linsolve(::LeastSquares, ::Val{2}, pos, x, A, b, ctx, solver, a₀, a₁, kind; iter = 1, g_global = 0.0, eps_trunc = 0.0, warn_tol = 0.0)
    A_eff = AC2_hamiltonian(pos, x, A, x, ctx.openvs)
    Asq_eff = AC2_hamiltonian(pos, x, ctx.Asq, x, ctx.sqenvs)
    N_eff = LinearCombination((A_eff, Asq_eff), (2 * real(conj(a₀) * a₁), abs2(a₁)))
    b_eff = AC2_projection(pos, x, b, ctx.rhsenvs)
    Ab_eff = _mixed_projection(Val(2), pos, x, A, b, ctx.abenvs)
    ac2 = AC2(x, pos; kind)
    res = norm(a₀ * ac2 + a₁ * (A_eff * ac2) - b_eff)
    rhs = conj(a₀) * b_eff + conj(a₁) * Ab_eff
    AC2′, info = KrylovKit.linsolve(N_eff, rhs, ac2, adapt_solver(solver; iter, g_global, eps_trunc), abs2(a₀), one(a₁))
    _warn_unconverged(info, "bond $pos", warn_tol)
    return AC2′, res
end

# Single-site driver (bond-preserving, so no truncation term in the stop test)
# ---------------------------------------------------------------------------
function linsolve!(
        x::AbstractFiniteMPS, A, b, alg::DMRGSolve,
        envs = environments(x, A, x); a₀ = 0, a₁ = 1
    )
    ctx = _rhs_context(alg.formulation, x, A, b, envs)
    normb = _rhs_norm(b)
    warn_tol = alg.tol * normb   # accuracy the outer sweep actually needs
    ϵ::Float64 = 2 * alg.tol   # relative residual, for the stop test
    ϵ_global = Inf             # previous-sweep absolute residual, drives the adaptive tolerance
    log = IterLog("linsolve")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0
            res_max = 0.0
            for pos in [1:(length(x) - 1); length(x):-1:2]
                AC′, res = _local_linsolve(
                    alg.formulation, Val(1), pos, x, A, b, ctx, alg.solver, a₀, a₁;
                    iter, g_global = ϵ_global, warn_tol
                )
                ϵ = max(ϵ, res / normb)
                res_max = max(res_max, res)
                x.AC[pos] = AC′
            end

            x, envs = alg.finalize(iter, x, A, envs)::Tuple{typeof(x), typeof(envs)}
            ϵ_global = res_max

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    return x, envs, ϵ
end

# Two-site driver (truncation-aware stop: the residual cannot beat the discarded weight)
# -------------------------------------------------------------------------------------
function linsolve!(
        x::AbstractFiniteMPS, A, b, alg::DMRGSolve2,
        envs = environments(x, A, x); a₀ = 0, a₁ = 1
    )
    ctx = _rhs_context(alg.formulation, x, A, b, envs)
    normb = _rhs_norm(b)
    warn_tol = alg.tol * normb   # accuracy the outer sweep actually needs
    ϵ_truncs = zeros(length(x) - 1)   # per-bond discarded weight
    ϵ::Float64 = 2 * alg.tol
    ϵ_global = Inf
    log = IterLog("linsolve2")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0
            res_max = 0.0
            for pos in 1:(length(x) - 1)
                AC2′, res = _local_linsolve(
                    alg.formulation, Val(2), pos, x, A, b, ctx, alg.solver, a₀, a₁, :ACAR;
                    iter, g_global = ϵ_global, eps_trunc = ϵ_truncs[pos], warn_tol
                )
                ϵ = max(ϵ, res / normb)
                res_max = max(res_max, res)
                x, tr = gauge2!(x, pos, Val(:right), AC2′, alg.alg_gauge; normalize = false)
                ϵ_truncs[pos] = tr
            end
            for pos in (length(x) - 2):-1:1
                AC2′, res = _local_linsolve(
                    alg.formulation, Val(2), pos, x, A, b, ctx, alg.solver, a₀, a₁, :ALAC;
                    iter, g_global = ϵ_global, eps_trunc = ϵ_truncs[pos], warn_tol
                )
                ϵ = max(ϵ, res / normb)
                res_max = max(res_max, res)
                x, tr = gauge2!(x, pos, Val(:left), AC2′, alg.alg_gauge; normalize = false)
                ϵ_truncs[pos] = tr
            end

            x, envs = alg.finalize(iter, x, A, envs)::Tuple{typeof(x), typeof(envs)}
            ϵ_global = res_max

            # the Galerkin residual cannot drop below the level set by the discarded weight
            if ϵ <= max(alg.tol, maximum(ϵ_truncs) / normb)
                @infov 2 logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ)
            else
                @infov 3 logiter!(log, iter, ϵ)
            end
        end
    end

    return x, envs, ϵ
end
