Base.@deprecate(approximate(ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO,<:MultilineMPS},
                            alg::VUMPS, envs...; kwargs...),
                approximate(ψ, toapprox,
                            VOMPS(; alg.tol, alg.maxiter, alg.finalize,
                                  alg.verbosity, alg.alg_gauge, alg.alg_environments),
                            envs...; kwargs...))

function approximate(ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO,<:MultilineMPS},
                     alg::VOMPS, envs=environments(ψ, toapprox))
    ϵ::Float64 = calc_galerkin(ψ, toapprox..., envs)
    temp_ACs = similar.(ψ.AC)
    scheduler = Defaults.scheduler[]
    log = IterLog("VOMPS")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)
        for iter in 1:(alg.maxiter)
            tmap!(eachcol(temp_ACs), 1:size(ψ, 2); scheduler) do col
                return _vomps_localupdate(col, ψ, toapprox, envs)
            end

            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = MultilineMPS(temp_ACs, ψ.C[:, end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, toapprox, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, toapprox..., envs)

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

    return ψ, envs, ϵ
end

function _vomps_localupdate(loc, ψ, (O, ψ₀), envs, factalg=QRpos())
    local tmp_AC, tmp_C
    if Defaults.scheduler[] isa SerialScheduler
        tmp_AC = circshift([ac_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
        tmp_C = circshift([c_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
    else
        @sync begin
            Threads.@spawn begin
                tmp_AC = circshift([ac_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
            end
            Threads.@spawn begin
                tmp_C = circshift([c_proj(row, loc, ψ, envs) for row in 1:size(ψ, 1)], 1)
            end
        end
    end
    return regauge!.(tmp_AC, tmp_C; alg=factalg)
end
