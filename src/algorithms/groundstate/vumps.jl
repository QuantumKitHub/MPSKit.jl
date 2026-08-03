"""
$(TYPEDEF)

Variational optimization algorithm for uniform matrix product states, based on the combination of DMRG with matrix product state tangent space concepts.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`find_groundstate`](@ref) and [`leading_boundary`](@ref).

# References

* [Zauner-Stauber et al. Phys. Rev. B 97 (2018)](@cite zauner-stauber2018)
* [Vanderstraeten et al. SciPost Phys. Lect. Notes 7 (2019)](@cite vanderstraeten2019)
"""
@kwdef struct VUMPS{F, B} <: Algorithm
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

    "backend for tensor contractions and index manipulations"
    backend::B = Defaults.backend()
end

# Internal state of the VUMPS algorithm
struct VUMPSState{S, O, E}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
    which::Symbol
    timeroutput::TimerOutput
end

function find_groundstate(
        mps::InfiniteMPS, operator, alg::VUMPS, envs...
    )
    return dominant_eigsolve(operator, mps, alg, envs...; which = :SR)
end

function dominant_eigsolve(
        operator, mps, alg::VUMPS, envs = environments(mps, operator, mps, alg.alg_environments);
        which
    )
    log = IterLog("VUMPS")
    timeroutput = TimerOutput("VUMPS")
    alg.verbosity > 3 || disable_timer!(timeroutput)
    iter = 0

    mps = copy(mps)
    ϵ = calc_galerkin(mps, operator, mps, envs)
    alg_environments = adapt_solver(alg.alg_environments; iter, g_global = ϵ)
    recalculate!(envs, mps, operator, mps, alg_environments; timeroutput)

    state = VUMPSState(mps, operator, envs, iter, ϵ, which, timeroutput)
    it = IterativeSolver(alg, state)

    result = LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, sum(expectation_value(mps, operator, envs)))

        for (mps, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 4 timeroutput
                @infov 2 logfinish!(log, it.iter, ϵ, expectation_value(mps, operator, envs))
                return mps, envs, ϵ
            end
            if it.iter ≥ alg.maxiter
                @infov 4 timeroutput
                @warnv 1 logcancel!(log, it.iter, ϵ, expectation_value(mps, operator, envs))
                return mps, envs, ϵ
            end
            @infov 3 logiter!(log, it.iter, ϵ, expectation_value(mps, operator, envs))
        end

        # this should never be reached
        return it.state.mps, it.state.envs, it.state.ϵ
    end

    return result
end

function Base.iterate(it::IterativeSolver{<:VUMPS}, state = it.state)
    timeroutput = state.timeroutput
    ACs = @timeit timeroutput "localupdate (parallel)" localupdate_step!(it, state)
    mps = @timeit timeroutput "gauge" gauge_step!(it, state, ACs)
    envs = @timeit timeroutput "envs (parallel)" envs_step!(it, state, mps)

    # finalizer step
    mps, envs = @timeit timeroutput "finalize" it.finalize(
        state.iter, mps, state.operator, envs
    )::typeof((mps, envs))

    # error criterion
    ϵ = @timeit timeroutput "calc_galerkin" calc_galerkin(mps, state.operator, mps, envs)

    # update state
    it.state = VUMPSState(
        mps, state.operator, envs, state.iter + 1, ϵ, state.which, timeroutput,
    )

    return (mps, envs, ϵ), it.state
end

function localupdate_step!(
        it::IterativeSolver{<:VUMPS}, state, scheduler = Defaults.scheduler[]
    )
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    alg_eigsolve = adapt_solver(it.alg_eigsolve; iter = state.iter, g_global = state.ϵ)
    alg_orth = alg_gauge.alg_orth

    mps = state.mps
    src_Cs = mps isa Multiline ? eachcol(mps.C) : mps.C
    src_ACs = mps isa Multiline ? eachcol(mps.AC) : mps.AC
    ACs = mps.AL
    dst_ACs = mps isa Multiline ? eachcol(ACs) : ACs

    tree_point = String[section.name for section in state.timeroutput.timer_stack]
    allocator = default_allocator(mps, scheduler)
    tforeach(eachsite(mps); scheduler) do site
        sub_timeroutput = TimerOutput()
        dst_ACs[site] = _localupdate_vumps_step!(
            site, mps, state.operator, state.envs, src_ACs[site], src_Cs[site];
            alg_orth, state.which, alg_eigsolve,
            timeroutput = sub_timeroutput, it.backend, allocator,
        )
        state.timeroutput.enabled &&
            merge!(state.timeroutput, sub_timeroutput; tree_point)
    end

    return ACs
end

function _localupdate_vumps_step!(
        site, mps, operator, envs, AC₀, C₀;
        alg_orth = Defaults.alg_orth(),
        alg_eigsolve = Defaults.eigsolver, which,
        timeroutput::TimerOutput = DISABLED_TIMER,
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator(),
    )
    local AC, C
    @timeit timeroutput "AC_eigsolve" begin
        Hac = AC_hamiltonian(site, mps, operator, mps, envs; backend, allocator)
        _, AC = fixedpoint(Hac, AC₀, which, alg_eigsolve)
    end
    @timeit timeroutput "C_eigsolve" begin
        Hc = C_hamiltonian(site, mps, operator, mps, envs; backend, allocator)
        _, C = fixedpoint(Hc, C₀, which, alg_eigsolve)
    end
    return regauge!(AC, C; alg = alg_orth)
end

function gauge_step!(it::IterativeSolver{<:VUMPS}, state, ACs::AbstractVector)
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    mps = gaugefix!(
        state.mps, ACs, state.mps.C[end];
        order = :R, timeroutput = state.timeroutput, alg_gauge...,
    )
    mul!.(mps.AC, mps.AL, mps.C)
    return mps
end
function gauge_step!(it::IterativeSolver{<:VUMPS}, state, ACs::AbstractMatrix)
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    return MultilineMPS(ACs, @view(state.mps.C[:, end]); alg_gauge.tol, alg_gauge.maxiter, alg_gauge.alg_orth)
end

function envs_step!(it::IterativeSolver{<:VUMPS}, state, mps)
    alg_environments = adapt_solver(it.alg_environments; iter = state.iter, g_global = state.ϵ)
    return recalculate!(state.envs, mps, state.operator, mps, alg_environments; state.timeroutput)
end
