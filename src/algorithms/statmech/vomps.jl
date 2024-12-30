"""
    VOMPS{F} <: Algorithm
    
Power method algorithm for infinite MPS.
[SciPost:4.1.004](https://scipost.org/SciPostPhysCore.4.1.004)
    
## Fields
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, toapprox, envs) -> ψ, envs`
- `verbosity::Int`: display progress information

- `alg_gauge=Defaults.alg_gauge()`: algorithm for gauging
- `alg_environments=Defaults.alg_environments()`: algorithm for updating environments
"""
@kwdef struct VOMPS{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbosity::Int = Defaults.verbosity

    alg_gauge = Defaults.alg_gauge()
    alg_environments = Defaults.alg_environments()
end

function leading_boundary(ψ::MultilineMPS, O::MultilineMPO, alg::VOMPS,
                          envs=environments(ψ, O))
    ϵ::Float64 = calc_galerkin(ψ, envs)
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
            recalculate!(envs, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, O, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, envs)

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
