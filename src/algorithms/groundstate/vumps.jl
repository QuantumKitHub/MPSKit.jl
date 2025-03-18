"""
$(TYPEDEF)

Variational optimization algorithm for uniform matrix product states, based on the combination of DMRG with matrix product state tangent space concepts.

## Fields

$(TYPEDFIELDS)

## References

* [Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018)
* [Vanderstraeten et al. SciPost Phys. Lect. Notes 7 (2019)](@cite vanderstraeten2019)
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
    log = IterLog("VUMPS")
    ϵ::Float64 = calc_galerkin(ψ, H, ψ, envs)
    ACs = similar.(ψ.AC)
    alg_environments = updatetol(alg.alg_environments, 0, ϵ)
    recalculate!(envs, ψ, H, ψ; alg_environments.tol)

    state = (; ψ, H, envs, ACs, iter=0, ϵ)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, sum(expectation_value(ψ, H, envs)))

        for (ψ, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ, expectation_value(ψ, H, envs))
                return ψ, envs, ϵ
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ, expectation_value(ψ, H, envs))
                return ψ, envs, ϵ
            end
            @infov 3 logiter!(log, it.iter, ϵ, expectation_value(ψ, H, envs))
        end

        return it.state.ψ, it.state.envs, it.state.ϵ
    end
end

function Base.iterate(it::IterativeSolver{<:VUMPS}, state=it.state)
    # eigsolver step
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    scheduler = Defaults.scheduler[]
    ACs = tmap!(state.ACs, 1:length(state.ψ); scheduler) do site
        return _vumps_localupdate(site, state.ψ, state.H, state.envs, alg_eigsolve)
    end

    # gauge step
    alg_gauge = updatetol(it.alg_gauge, state.iter, state.ϵ)
    ψ = InfiniteMPS(ACs, state.ψ.C[end]; alg_gauge.tol, alg_gauge.maxiter)

    # environment step
    alg_environments = updatetol(it.alg_environments, state.iter, state.ϵ)
    envs = recalculate!(state.envs, ψ, state.H, ψ; alg_environments.tol)

    # finalizer step
    ψ′, envs′ = it.finalize(state.iter, ψ, state.H, envs)::Tuple{typeof(ψ),typeof(envs)}

    # error criterion
    ϵ = calc_galerkin(ψ′, state.H, ψ′, envs′)

    # update state
    it.state = (; ψ=ψ′, H=state.H, envs=envs′, ACs, iter=state.iter + 1, ϵ)

    return (ψ′, envs′, ϵ), it.state
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
