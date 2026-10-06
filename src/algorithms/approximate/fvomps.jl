function approximate!(ψ::AbstractFiniteMPS, Oϕ, alg::DMRG2, envs = environments(ψ, _environment_args(Oϕ)...))
    allocator = default_allocator(ψ, SerialScheduler())
    ϵ::Float64 = 2 * alg.tol
    iter = 0
    log = IterLog("DMRG2")

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            ϵ = 0.0
            for pos in [1:(length(ψ) - 1); (length(ψ) - 2):-1:1]
                AC2′ = AC2_projection(pos, ψ, Oϕ, envs; alg.backend, allocator)
                al, c, ar, = svd_trunc!(AC2′, inner_alg_gauge(alg))

                AC2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
                ϵ = max(ϵ, norm(al * c * ar - AC2) / norm(AC2))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            # finalize
            ψ, envs = alg.finalize(iter, ψ, Oϕ, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
                @log_convergence logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @log_nonconvergence logcancel!(log, iter, ϵ)
            else
                @log_iteration logiter!(log, iter, ϵ)
            end
        end
    end

    return ψ, envs, AlgorithmInfo(; converged = ϵ <= alg.tol, localchange = ϵ, numiter = iter)
end

function approximate!(ψ::AbstractFiniteMPS, Oϕ, alg::DMRG, envs = environments(ψ, _environment_args(Oϕ)...))
    allocator = default_allocator(ψ, SerialScheduler())
    ϵ::Float64 = 2 * alg.tol
    iter = 0
    log = IterLog("DMRG")

    with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, ϵ)
        for outer iter in 1:(alg.maxiter)
            ϵ = 0.0
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                AC′ = AC_projection(pos, ψ, Oϕ, envs; alg.backend, allocator)
                AC = ψ.AC[pos]
                ϵ = max(ϵ, norm(AC′ - AC) / norm(AC′))

                ψ.AC[pos] = AC′
            end

            # finalize
            ψ, envs = alg.finalize(iter, ψ, Oϕ, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ <= alg.tol
                @log_convergence logfinish!(log, iter, ϵ)
                break
            end
            if iter == alg.maxiter
                @log_nonconvergence logcancel!(log, iter, ϵ)
            else
                @log_iteration logiter!(log, iter, ϵ)
            end
        end
    end

    return ψ, envs, AlgorithmInfo(; converged = ϵ <= alg.tol, localchange = ϵ, numiter = iter)
end
