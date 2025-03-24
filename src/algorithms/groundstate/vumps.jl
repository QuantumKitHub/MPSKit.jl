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

# Internal state of the VUMPS algorithm
struct VUMPSState{S,O,E}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
    which::Symbol
end

function find_groundstate(mps::InfiniteMPS, operator, alg::VUMPS,
                          envs=environments(mps, operator))
    return dominant_eigsolve(operator, mps, alg, envs; which=:SR)
end

function dominant_eigsolve(operator, mps, alg::VUMPS, envs=environments(mps, operator);
                           which)
    log = IterLog("VUMPS")
    iter = 0
    ϵ = calc_galerkin(mps, operator, mps, envs)
    alg_environments = updatetol(alg.alg_environments, iter, ϵ)
    recalculate!(envs, mps, operator, mps; alg_environments.tol)

    state = VUMPSState(mps, operator, envs, iter, ϵ, which)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, sum(expectation_value(mps, operator, envs)))

        for (mps, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ, expectation_value(mps, operator, envs))
                return mps, envs, ϵ
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ, expectation_value(mps, operator, envs))
                return mps, envs, ϵ
            end
            @infov 3 logiter!(log, it.iter, ϵ, expectation_value(mps, operator, envs))
        end

        # this should never be reached
        return it.state.mps, it.state.envs, it.state.ϵ
    end
end

function _vumps_localupdate(loc, ψ, H, envs, alg_eigsolve, factalg=QRpos())
    local AC′, C′
    if Defaults.scheduler[] isa SerialScheduler
        _, AC′ = fixedpoint(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], :SR, alg_eigsolve)
        _, C′ = fixedpoint(∂∂C(loc, ψ, H, envs), ψ.C[loc], :SR, alg_eigsolve)
    else
        @sync begin
            Threads.@spawn begin
                _, AC′ = fixedpoint(∂∂AC(loc, ψ, H, envs), ψ.AC[loc], :SR, alg_eigsolve)
            end
            Threads.@spawn begin
                _, C′ = fixedpoint(∂∂C(loc, ψ, H, envs), ψ.C[loc], :SR, alg_eigsolve)
            end
        end
    end
    return regauge!(AC, C; alg=alg_orth)
end

function gauge_step!(it::IterativeSolver{<:VUMPS}, state, ACs::AbstractVector)
    alg_gauge = updatetol(it.alg_gauge, state.iter, state.ϵ)
    return InfiniteMPS(ACs, state.mps.C[end]; alg_gauge.tol, alg_gauge.maxiter)
end
function gauge_step!(it::IterativeSolver{<:VUMPS}, state, ACs::AbstractMatrix)
    alg_gauge = updatetol(it.alg_gauge, state.iter, state.ϵ)
    return MultilineMPS(ACs, @view(state.mps.C[:, end]); alg_gauge.tol, alg_gauge.maxiter)
end

function envs_step!(it::IterativeSolver{<:VUMPS}, state, mps)
    alg_environments = updatetol(it.alg_environments, state.iter, state.ϵ)
    return recalculate!(state.envs, mps, state.operator, mps; alg_environments.tol)
end
