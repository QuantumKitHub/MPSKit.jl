"""
$(TYPEDEF)

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct VUMPS{F} <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol

    "maximal amount of iterations"
    maxiter::Int = Defaults.maxiter

    "setting for how much information is displayed"
    verbosity::Int = Defaults.verbosity

    "algorithm used for gauging the `InfiniteMPS`"
    alg_gauge = Defaults.alg_gauge()

    "algorithm used for the eigenvalue solvers"
    alg_eigsolve = Defaults.alg_eigsolve()

    "algorithm used for the MPS environments"
    alg_environments = Defaults.alg_environments()

    "callback function applied after each iteration, of signature `finalize(iter, ψ, H, envs) -> ψ, envs`"
    finalize::F = Defaults._finalize
end

function find_groundstate(ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(ψ, H))
    # initialization
    scheduler = Defaults.scheduler[]
    log = IterLog("VUMPS")
    ϵ::Float64 = calc_galerkin(ψ, H, ψ, envs)
    temp_ACs = similar.(ψ.AC)
    alg_environments = updatetol(alg.alg_environments, 0, ϵ)
    recalculate!(envs, ψ, H, ψ; alg_environments.tol)

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, sum(expectation_value(ψ, H, envs)))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            tmap!(temp_ACs, 1:length(ψ); scheduler) do loc
                return _vumps_localupdate(loc, ψ, H, envs, alg_eigsolve)
            end

            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = InfiniteMPS(temp_ACs, ψ.C[end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ, H, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, H, ψ, envs)

            # breaking conditions
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

function _vumps_localupdate(loc, ψ, H, envs, eigalg, factalg=QRpos())
    local AC′, C′
    if Defaults.scheduler[] isa SerialScheduler
        _, AC′ = fixedpoint(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], :SR, eigalg)
        _, C′ = fixedpoint(∂∂C(loc, ψ, H, envs), ψ.C[loc], :SR, eigalg)
    else
        @sync begin
            Threads.@spawn begin
                _, AC′ = fixedpoint(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], :SR, eigalg)
            end
            Threads.@spawn begin
                _, C′ = fixedpoint(∂∂C(loc, ψ, H, envs), ψ.C[loc], :SR, eigalg)
            end
        end
    end
    return regauge!(AC′, C′; alg=factalg)
end
