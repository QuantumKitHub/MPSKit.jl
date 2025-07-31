function approximate!(ψ::AbstractFiniteMPS, Oϕ, alg::DMRG2, envs = environments(ψ, Oϕ))
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0
            for pos in [1:(length(ψ) - 1); (length(ψ) - 2):-1:1]
                AC2′ = AC2_projection(pos, ψ, Oϕ, envs)
                al, c, ar, = tsvd!(AC2′; trunc = alg.trscheme, alg = alg.alg_svd)

                AC2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
                ϵ = max(ϵ, norm(al * c * ar - AC2) / norm(AC2))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            # finalize
            ψ, envs = alg.finalize(iter, ψ, Oϕ, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ < alg.tol
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

    return ψ, envs, ϵ
end

function approximate!(ψ::AbstractFiniteMPS, Oϕ, alg::DMRG, envs = environments(ψ, Oϕ))
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                AC′ = AC_projection(pos, ψ, Oϕ, envs)
                AC = ψ.AC[pos]
                ϵ = max(ϵ, norm(AC′ - AC) / norm(AC′))

                ψ.AC[pos] = AC′
            end

            # finalize
            ψ, envs = alg.finalize(iter, ψ, Oϕ, envs)::Tuple{typeof(ψ), typeof(envs)}

            if ϵ < alg.tol
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

    return ψ, envs, ϵ
end
