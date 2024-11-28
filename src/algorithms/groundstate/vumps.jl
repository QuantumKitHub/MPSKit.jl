"""
    VUMPS{F} <: Algorithm

Variational optimization algorithm for uniform matrix product states, as introduced in
https://arxiv.org/abs/1701.07035.

# Fields
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `finalize::F`: user-supplied function which is applied after each iteration, with
    signature `finalize(iter, ψ, H, envs) -> ψ, envs`
- `verbosity::Int`: display progress information

- `alg_gauge=Defaults.alg_gauge()`: algorithm for gauging
- `alg_eigsolve=Defaults.alg_eigsolve()`: algorithm for eigensolvers
- `alg_environments=Defaults.alg_environments()`: algorithm for updating environments
"""
@kwdef struct VUMPS{F} <: Algorithm
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    finalize::F = Defaults._finalize
    verbosity::Int = Defaults.verbosity

    alg_gauge = Defaults.alg_gauge()
    alg_eigsolve = Defaults.alg_eigsolve()
    alg_environments = Defaults.alg_environments()
end

function find_groundstate(ψ::InfiniteMPS, H, alg::VUMPS, envs=environments(ψ, H))
    # initialization
    ϵ::Float64 = calc_galerkin(ψ, envs)
    temp_ACs = similar.(ψ.AC)
    log = IterLog("VUMPS")

    LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, sum(expectation_value(ψ, H, envs)))
        for iter in 1:(alg.maxiter)
            alg_eigsolve = updatetol(alg.alg_eigsolve, iter, ϵ)
            @static if Defaults.parallelize_sites
                @sync for loc in 1:length(ψ)
                    Threads.@spawn begin
                        temp_ACs[loc] = _vumps_localupdate(loc, ψ, H, envs, alg_eigsolve)
                    end
                end
            else
                for loc in 1:length(ψ)
                    temp_ACs[loc] = _vumps_localupdate(loc, ψ, H, envs, alg_eigsolve)
                end
            end

            alg_gauge = updatetol(alg.alg_gauge, iter, ϵ)
            ψ = InfiniteMPS(temp_ACs, ψ.CR[end]; alg_gauge.tol, alg_gauge.maxiter)

            alg_environments = updatetol(alg.alg_environments, iter, ϵ)
            recalculate!(envs, ψ; alg_environments.tol)

            ψ, envs = alg.finalize(iter, ψ, H, envs)::Tuple{typeof(ψ),typeof(envs)}

            ϵ = calc_galerkin(ψ, envs)

            # breaking conditions
            if ϵ <= alg.tol
                @infov 2 logfinish!(log, iter, ϵ, sum(expectation_value(ψ, H, envs)))
                break
            end
            if iter == alg.maxiter
                @warnv 1 logcancel!(log, iter, ϵ, sum(expectation_value(ψ, H, envs)))
            else
                @infov 3 logiter!(log, iter, ϵ, sum(expectation_value(ψ, H, envs)))
            end
        end
    end

    return ψ, envs, ϵ
end

function _vumps_localupdate(loc, ψ, H, envs, eigalg, factalg=QRpos())
    local AC′, C′
    @static if Defaults.parallelize_sites
        @sync begin
            Threads.@spawn begin
                _, AC′ = fixedpoint(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], :SR, eigalg)
            end
            Threads.@spawn begin
                _, C′ = fixedpoint(∂∂C(loc, ψ, H, envs), ψ.CR[loc], :SR, eigalg)
            end
        end
    else
        _, AC′ = fixedpoint(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], :SR, eigalg)
        _, C′ = fixedpoint(∂∂C(loc, ψ, H, envs), ψ.CR[loc], :SR, eigalg)
    end
    return regauge!(AC′, C′; alg=factalg)
end
