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

function Base.iterate(it::IterativeSolver{<:VUMPS}, state=it.state)
    ACs = localupdate_step!(it, state)
    mps = gauge_step!(it, state, ACs)
    envs = envs_step!(it, state, mps)

    # finalizer step
    mps, envs = it.finalize(state.iter, mps, state.operator, envs)::typeof((mps, envs))

    # error criterion
    ϵ = calc_galerkin(mps, state.operator, mps, envs)

    # update state
    it.state = VUMPSState(mps, state.operator, envs, state.iter + 1, ϵ, state.which)

    return (mps, envs, ϵ), it.state
end

function localupdate_step!(it::IterativeSolver{<:VUMPS}, state,
                           scheduler=Defaults.scheduler[])
    alg_eigsolve = updatetol(it.alg_eigsolve, state.iter, state.ϵ)
    alg_orth = QRpos()

    mps = state.mps
    src_Cs = mps isa Multiline ? eachcol(mps.C) : mps.C
    src_ACs = mps isa Multiline ? eachcol(mps.AC) : mps.AC
    ACs = similar(mps.AC)
    dst_ACs = mps isa Multiline ? eachcol(ACs) : ACs

    tforeach(eachsite(mps), src_ACs, src_Cs; scheduler) do site, AC₀, C₀
        dst_ACs[site] = _localupdate_vumps_step!(site, mps, state.operator, state.envs,
                                                 AC₀, C₀; parallel=false, alg_orth,
                                                 state.which, alg_eigsolve)
        return nothing
    end

    return ACs
end

function _localupdate_vumps_step!(site, mps, operator, envs, AC₀, C₀;
                                  parallel::Bool=false, alg_orth=QRpos(),
                                  alg_eigsolve=Defaults.eigsolver, which)
    if !parallel
        Hac = AC_hamiltonian(site, mps, operator, mps, envs)
        _, AC = fixedpoint(Hac, AC₀, which, alg_eigsolve)
        Hc = C_hamiltonian(site, mps, operator, mps, envs)
        _, C = fixedpoint(Hc, C₀, which, alg_eigsolve)
        return regauge!(AC, C; alg=alg_orth)
    end

    local AC, C
    @sync begin
        @spawn begin
            Hac = AC_hamiltonian(site, mps, operator, mps, envs)
            _, AC = fixedpoint(Hac, AC₀, which, alg_eigsolve)
        end
        @spawn begin
            Hc = C_hamiltonian(site, mps, operator, mps, envs)
            _, C = fixedpoint(Hc, C₀, which, alg_eigsolve)
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
