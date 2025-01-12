"""
$(TYPEDEF)
    
Power method algorithm for infinite MPS.
[SciPost:4.1.004](https://scipost.org/SciPostPhysCore.4.1.004)
    
## Fields

$(TYPEDFIELDS)
"""
@kwdef struct VOMPS{F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the `InfiniteMPS`"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the MPS environments"
    alg_environments = Defaults.alg_environments()

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F = Defaults._finalize
end

function leading_boundary(ψ::MultilineMPS, O::MultilineMPO, alg::VOMPS,
                          envs=environments(ψ, O))
    ϵ::Float64 = calc_galerkin(ψ, O, ψ, envs)
    temp_ACs = similar.(ψ.AC)
    scheduler = Defaults.scheduler[]
    log = IterLog("VOMPS")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, expectation_value(ψ, O, envs))
        for iter in 1:(alg.maxiter)
            tmap!(eachcol(temp_ACs), 1:size(ψ, 2); scheduler) do col
                return _vomps_localupdate(col, ψ, O, envs)
            end

            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = MultilineMPS(temp_ACs, ψ.C[:, end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ, O, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, O, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, O, ψ, envs)

            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, expectation_value(ψ, O, envs))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, expectation_value(ψ, O, envs))
            else
                @infov 3 logiter!(log, iter, ϵ, expectation_value(ψ, O, envs))
            end
        end
    end

    return ψ, envs, ϵ
end

function _vomps_localupdate(col, ψ::MultilineMPS, O::MultilineMPO, envs, factalg=QRpos())
    local AC′, C′
    if Defaults.scheduler[] isa SerialScheduler
        AC′ = ∂∂AC(col, ψ, O, envs) * ψ.AC[:, col]
        C′ = ∂∂C(col, ψ, O, envs) * ψ.C[:, col]
    else
        @sync begin
            Threads.@spawn begin
                AC′ = ∂∂AC(col, ψ, O, envs) * ψ.AC[:, col]
            end
            Threads.@spawn begin
                C′ = ∂∂C(col, ψ, O, envs) * ψ.C[:, col]
            end
        end
    end
    return regauge!.(AC′, C′; alg=factalg)[:]
end
