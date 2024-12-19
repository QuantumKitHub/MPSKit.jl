#the statmech VUMPS
#it made sense to seperate both vumpses as
# - leading_boundary primarily works on MultilineMPS
# - they search for different eigenvalues
# - Hamiltonian vumps should use Lanczos, this has to use arnoldi
# - this vumps updates entire collumns (ψ[:,i]); incompatible with InfiniteMPS
# - (a)c-prime takes a different number of arguments
# - it's very litle duplicate code, but together it'd be a bit more convoluted (primarily because of the indexing way)

"""
    leading_boundary(ψ, opp, alg, envs=environments(ψ, opp))

Approximate the leading eigenvector for opp.
"""
function leading_boundary(ψ::InfiniteMPS, H, alg, envs=environments(ψ, H))
    (st, pr, de) = leading_boundary(convert(MultilineMPS, ψ), Multiline([H]), alg, envs)
    return convert(InfiniteMPS, st), pr, de
end

function leading_boundary(ψ::MultilineMPS, H, alg::VUMPS, envs=environments(ψ, H))
    # initialization
    log = IterLog("VUMPS")
    ϵ::Float64 = calc_galerkin(ψ, H, ψ, envs)
    scheduler = Defaults.scheduler[]
    temp_ACs = similar.(ψ.AC)
    alg_environments = updatetol(alg.alg_environments, 0, ϵ)
    recalculate!(envs, ψ, H; alg_environments.tol)

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, H, envs))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            tmap!(eachcol(temp_ACs), 1:size(ψ, 2); scheduler) do col
                return _vumps_localupdate(col, ψ, H, envs, alg_eigsolve)
            end
            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = MultilineMPS(temp_ACs, ψ.C[:, end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ, H, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, envs)

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, H, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, H, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, H, envs))
            end
        end
    end

    return ψ, envs, ϵ
end

function _vumps_localupdate(col, ψ::MultilineMPS, O::MultilineMPO, envs, eigalg,
                            factalg=QRpos())
    local AC′, C′
    if Defaults.scheduler[] isa SerialScheduler
        _, AC′ = fixedpoint(∂∂AC(col, ψ, O, envs), ψ.AC[:, col], :LM, eigalg)
        _, C′ = fixedpoint(∂∂C(col, ψ, O, envs), ψ.C[:, col], :LM, eigalg)
    else
        @sync begin
            Threads.@spawn begin
                _, AC′ = fixedpoint(∂∂AC(col, ψ, O, envs), ψ.AC[:, col], :LM, eigalg)
            end
            Threads.@spawn begin
                _, C′ = fixedpoint(∂∂C(col, ψ, O, envs), ψ.C[:, col], :LM, eigalg)
            end
        end
    end
    return regauge!.(AC′, C′; alg=factalg)[:]
end
