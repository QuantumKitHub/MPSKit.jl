# Internal state of the finite DMRG/DMRG2 approximation, where `ϵ` is the largest relative
# local change of the last sweep
struct ApproximateState{S, O, E, A}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
    allocator::A
end

function approximate!(
        ψ::AbstractFiniteMPS, Oϕ, alg::Union{DMRG, DMRG2},
        envs = environments(ψ, _environment_args(Oϕ)...)
    )
    allocator = default_allocator(ψ, SerialScheduler())
    log = IterLog(string(nameof(typeof(alg))))
    it = IterativeSolver(alg, ApproximateState(ψ, Oϕ, envs, 0, 2 * alg.tol, allocator))

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, it.ϵ)
        for (_, _, ϵ) in Iterators.take(it, alg.maxiter)
            if ϵ <= alg.tol
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
    info = AlgorithmInfo(; converged = state.ϵ <= alg.tol, localchange = state.ϵ, numiter = state.iter)
    return state.mps, state.envs, info
end

function Base.iterate(it::IterativeSolver{<:Union{DMRG, DMRG2}}, state::ApproximateState)
    iter = state.iter + 1
    ϵ = approximate_sweep!(state.mps, state.operator, it.alg, state.envs, state.allocator)
    ψ, envs = it.finalize(
        iter, state.mps, state.operator, state.envs
    )::Tuple{typeof(state.mps), typeof(state.envs)}
    it.state = ApproximateState(ψ, state.operator, envs, iter, ϵ, state.allocator)
    return (ψ, envs, ϵ), it.state
end

function approximate_sweep!(ψ, Oϕ, alg::DMRG2, envs, allocator)
    ϵ = 0.0
    for pos in [1:(length(ψ) - 1); (length(ψ) - 2):-1:1]
        AC2′ = AC2_projection(pos, ψ, Oϕ, envs; alg.backend, allocator)
        al, c, ar, = svd_trunc!(AC2′, inner_alg_gauge(alg))

        AC2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
        ϵ = max(ϵ, norm(al * c * ar - AC2) / norm(AC2))

        ψ.AC[pos] = (al, complex(c))
        ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
    end
    return ϵ
end

function approximate_sweep!(ψ, Oϕ, alg::DMRG, envs, allocator)
    ϵ = 0.0
    for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
        AC′ = AC_projection(pos, ψ, Oϕ, envs; alg.backend, allocator)
        AC = ψ.AC[pos]
        ϵ = max(ϵ, norm(AC′ - AC) / norm(AC′))

        ψ.AC[pos] = AC′
    end
    return ϵ
end
