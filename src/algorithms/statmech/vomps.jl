"""
$(TYPEDEF)

Power method algorithm for finding dominant eigenvectors of infinite MPOs.
This method works by iteratively approximating the product of an operator and a state
with a new state of the same bond dimension.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`leading_boundary`](@ref) and [`approximate`](@ref).

# References

* [Vanhecke et al. SciPost Phys. Core 4 (2021)](@cite vanhecke2021)
"""
@kwdef struct VOMPS{F, B} <: Algorithm
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

    "backend for tensor contractions and index manipulations"
    backend::B = Defaults.backend()
end

# Internal state of the VOMPS algorithm
struct VOMPSState{S, O, E}
    mps::S
    operator::O
    envs::E
    iter::Int
    ϵ::Float64
end

function leading_boundary(ψ::MultilineMPS, O::MultilineMPO, alg::Union{VOMPS, VUMPS}, envs...)
    return dominant_eigsolve(O, ψ, alg, envs...; which = :LM)
end

function dominant_eigsolve(
        operator, mps, alg::VOMPS, envs = environments(mps, operator, mps, alg.alg_environments);
        which
    )
    @assert which === :LM "VOMPS only supports the LM eigenvalue problem"
    log = IterLog("VOMPS")
    iter = 0
    ϵ = calc_galerkin(mps, operator, mps, envs; alg.backend)
    alg_environments = adapt_solver(alg.alg_environments; iter, g_global = ϵ)
    recalculate!(envs, mps, operator, mps, alg_environments)

    state = VOMPSState(mps, operator, envs, iter, ϵ)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ, dominant_eigenvalue(mps, operator, envs))

        for (mps, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ, dominant_eigenvalue(mps, operator, envs))
                return mps, envs, ϵ
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ, dominant_eigenvalue(mps, operator, envs))
                return mps, envs, ϵ
            end
            @infov 3 logiter!(log, it.iter, ϵ, dominant_eigenvalue(mps, operator, envs))
        end

        # this should never be reached
        return it.state.mps, it.state.envs, it.state.ϵ
    end
end

function Base.iterate(it::IterativeSolver{<:VOMPS}, state)
    ACs = localupdate_step!(it, state)
    mps = gauge_step!(it, state, ACs)
    envs = envs_step!(it, state, mps)

    # finalizer step
    mps, envs = it.finalize(state.iter, mps, state.operator, envs)::typeof((mps, envs))

    # error criterion
    ϵ = calc_galerkin(mps, state.operator, mps, envs; it.backend)

    # update state
    it.state = VOMPSState(mps, state.operator, envs, state.iter + 1, ϵ)

    return (mps, envs, ϵ), it.state
end

function localupdate_step!(
        it::IterativeSolver{<:VOMPS}, state, scheduler = Defaults.scheduler[]
    )
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    alg_orth = alg_gauge.alg_orth
    mps = state.mps
    ACs = similar(mps.AC)
    dst_ACs = state.mps isa Multiline ? eachcol(ACs) : ACs

    allocator = default_allocator(mps, scheduler)
    tforeach(eachsite(mps); scheduler) do site
        dst_ACs[site] = _localupdate_vomps_step!(
            site, mps, state.operator, state.envs;
            alg_orth, it.backend, allocator
        )
        return nothing
    end

    return ACs
end

function _localupdate_vomps_step!(
        site, mps, operator, envs; alg_orth = Defaults.alg_orth(),
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    AC = AC_projection(site, mps, operator, mps, envs; backend, allocator)
    C = C_projection(site, mps, operator, mps, envs; backend, allocator)
    return regauge!(AC, C; alg = alg_orth)
end

function gauge_step!(it::IterativeSolver{<:VOMPS}, state, ACs::AbstractVector)
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    return InfiniteMPS(ACs, state.mps.C[end]; alg_gauge...)
end
function gauge_step!(it::IterativeSolver{<:VOMPS}, state, ACs::AbstractMatrix)
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    return MultilineMPS(ACs, @view(state.mps.C[:, end]); alg_gauge...)
end

function envs_step!(it::IterativeSolver{<:VOMPS}, state, mps)
    alg_environments = adapt_solver(it.alg_environments; iter = state.iter, g_global = state.ϵ)
    return recalculate!(state.envs, mps, state.operator, mps, alg_environments)
end
