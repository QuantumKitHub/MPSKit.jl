Base.@deprecate(
    approximate(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS}, alg::VUMPS, envs...;
        kwargs...
    ),
    approximate(
        ψ, toapprox,
        VOMPS(;
            alg.tol, alg.maxiter, alg.finalize, alg.verbosity, alg.alg_gauge,
            alg.alg_environments, alg.backend,
        ),
        envs...; kwargs...
    )
)

function approximate(
        mps::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS}, alg::VOMPS,
        envs = environments(mps, toapprox...)
    )
    return _approximate_vomps(mps, toapprox, alg, envs)
end

function _approximate_vomps(mps, toapprox, alg::VOMPS, envs)
    log = IterLog(alg)
    iter = 0
    ϵ = calc_galerkin(mps, toapprox..., envs; alg.backend)
    alg_environments = adapt_solver(alg.alg_environments; iter, g_global = ϵ)
    recalculate!(envs, mps, toapprox..., alg_environments)

    state = VOMPSState(mps, toapprox, envs, iter, ϵ)
    it = IterativeSolver(alg, state)

    return with_verbosity(; alg.verbosity) do
        @log_initialization loginit!(log, ϵ)

        for (mps, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @log_convergence logfinish!(log, it.iter, ϵ)
                return mps, envs, AlgorithmInfo(; converged = true, galerkin = ϵ, numiter = it.iter)
            end
            if it.iter ≥ alg.maxiter
                @log_nonconvergence logcancel!(log, it.iter, ϵ)
                return mps, envs, AlgorithmInfo(; converged = false, galerkin = ϵ, numiter = it.iter)
            end
            @log_iteration logiter!(log, it.iter, ϵ)
        end

        # this should never be reached
        return it.state.mps, it.state.envs, it.state.ϵ
    end
end

# need to specialize a bunch of functions because different arguments are passed with tuples
# TODO: can we avoid this?
function Base.iterate(it::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple})
    ACs = localupdate_step!(it, state)
    mps = gauge_step!(it, state, ACs)
    envs = envs_step!(it, state, mps)

    # finalizer step
    mps, envs = it.finalize(state.iter, mps, state.operator, envs)::typeof((mps, envs))

    # error criterion
    ϵ = calc_galerkin(mps, state.operator..., envs; it.backend)

    # update state
    it.state = VOMPSState(mps, state.operator, envs, state.iter + 1, ϵ)

    return (mps, envs, ϵ), it.state
end

function localupdate_step!(
        it::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, scheduler
    )
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    alg_orth = alg_gauge.alg_orth

    ACs = similar(state.mps.AC)
    dst_ACs = state.mps isa Multiline ? eachcol(ACs) : ACs

    # the allocator is shared between the sites, so it has to tolerate the scheduler running them
    # concurrently
    allocator = default_allocator(state.mps, scheduler)
    tforeach(eachsite(state.mps); scheduler) do site
        dst_ACs[site] = _localupdate_vomps_step!(
            site, state.mps, state.operator, state.envs;
            alg_orth, it.backend, allocator
        )
        return nothing
    end

    return ACs
end

function _localupdate_vomps_step!(
        site, mps, operator::Tuple, envs; alg_orth = Defaults.alg_orth(),
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    AC = map(1:size(mps, 1)) do row
        AC_projection(CartesianIndex(row, site), mps, operator, envs; backend, allocator)
    end
    circshift!(AC, 1)
    C = map(1:size(mps, 1)) do row
        C_projection(CartesianIndex(row, site), mps, operator, envs; backend, allocator)
    end
    circshift!(C, 1)
    return regauge!(AC, C; alg = alg_orth)
end

function envs_step!(it::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, mps)
    alg_environments = adapt_solver(it.alg_environments; iter = state.iter, g_global = state.ϵ)
    return recalculate!(state.envs, mps, state.operator..., alg_environments)
end
