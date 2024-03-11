# dispatch to in-place method
function approximate(ψ, toapprox, alg::Union{DMRG,DMRG2}, envs...)
    return approximate!(copy(ψ), toapprox, alg, envs...)
end

function approximate!(ψ::AbstractFiniteMPS, sq, alg, envs=environments(ψ, sq))
    tor = approximate!(ψ, [sq], alg, [envs])
    return (tor[1], tor[2][1], tor[3])
end

function approximate!(ψ::AbstractFiniteMPS, squash::Vector, alg::DMRG2,
                      envs=[environments(ψ, sq) for sq in squash])
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("DMRG2")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0
            for pos in [1:(length(ψ) - 1); (length(ψ) - 2):-1:1]
                AC2′ = sum(zip(squash, envs)) do (sq, pr)
                    return ac2_proj(pos, ψ, pr)
                end
                al, c, ar, = tsvd!(AC2′; trunc=alg.trscheme)

                AC2 = ψ.AC[pos] * _transpose_tail(ψ.AR[pos + 1])
                ϵ = max(ϵ, norm(al * c * ar - AC2) / norm(AC2))

                ψ.AC[pos] = (al, complex(c))
                ψ.AC[pos + 1] = (complex(c), _transpose_front(ar))
            end

            # finalize
            ψ, envs = alg.finalize(iter, ψ, squash, envs)::Tuple{typeof(ψ),typeof(envs)}

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

function approximate!(ψ::AbstractFiniteMPS, squash::Vector, alg::DMRG,
                      envs=[environments(ψ, sq) for sq in squash])
    ϵ::Float64 = 2 * alg.tol
    log = IterLog("DMRG")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            ϵ = 0.0
            for pos in [1:(length(ψ) - 1); length(ψ):-1:2]
                AC′ = sum(zip(squash, envs)) do (sq, pr)
                    return ac_proj(pos, ψ, pr)
                end

                AC = ψ.AC[pos]
                ϵ = max(ϵ, norm(AC′ - AC) / norm(AC′))

                ψ.AC[pos] = AC′
            end

            # finalize
            ψ, envs = alg.finalize(iter, ψ, squash, envs)::Tuple{typeof(ψ),typeof(envs)}

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
