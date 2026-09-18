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
    log = IterLog("VOMPS")
    iter = 0
    ϵ = calc_galerkin(mps, toapprox..., envs; alg.backend)
    alg_environments = adapt_solver(alg.alg_environments; iter, g_global = ϵ)
    recalculate!(envs, mps, toapprox..., alg_environments)

    state = VOMPSState(mps, toapprox, envs, iter, ϵ)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)

        for (mps, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ)
                return mps, envs, AlgorithmInfo(; converged = true, galerkin = ϵ, numiter = it.iter)
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ)
                return mps, envs, AlgorithmInfo(; converged = false, galerkin = ϵ, numiter = it.iter)
            end
            @infov 3 logiter!(log, it.iter, ϵ)
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
        it::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, ::SerialScheduler
    )
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    alg_orth = alg_gauge.alg_orth

    ACs = similar(state.mps.AC)
    dst_ACs = state.mps isa Multiline ? eachcol(ACs) : ACs

    # the sweep is serial, so a single allocator serves all sites
    allocator = default_allocator(state.mps, SerialScheduler())
    for site in eachsite(state.mps)
        AC = map(1:size(state.mps, 1)) do row
            AC_projection(
                CartesianIndex(row, site), state.mps, state.operator, state.envs;
                it.backend, allocator
            )
        end
        circshift!(AC, 1)
        C = map(1:size(state.mps, 1)) do row
            C_projection(
                CartesianIndex(row, site), state.mps, state.operator, state.envs;
                it.backend, allocator
            )
        end
        circshift!(C, 1)
        dst_ACs[site] = regauge!(AC, C; alg = alg_orth)
    end

    return ACs
end
function localupdate_step!(
        it::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, scheduler
    )
    alg_gauge = adapt_solver(it.alg_gauge; iter = state.iter, g_global = state.ϵ)
    alg_orth = alg_gauge.alg_orth

    ACs = similar(state.mps.AC)
    dst_ACs = state.mps isa Multiline ? eachcol(ACs) : ACs

    # every site - and the AC and C projections within a site - runs concurrently, so the allocator
    # is shared and has to be one that tolerates that
    allocator = default_allocator(state.mps, scheduler)
    tforeach(eachsite(state.mps); scheduler) do site
        local AC, C
        @sync begin
            Threads.@spawn begin
                AC = map(1:size(state.mps, 1)) do row
                    AC_projection(
                        CartesianIndex(row, site), state.mps, state.operator, state.envs;
                        it.backend, allocator
                    )
                end
                circshift!(AC, 1)
            end
            Threads.@spawn begin
                C = map(1:size(state.mps, 1)) do row
                    C_projection(
                        CartesianIndex(row, site), state.mps, state.operator, state.envs;
                        it.backend, allocator
                    )
                end
                circshift!(C, 1)
            end
        end
        dst_ACs[site] = regauge!(AC, C; alg = alg_orth)
        return nothing
    end

    return ACs
end

function envs_step!(it::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, mps)
    alg_environments = adapt_solver(it.alg_environments; iter = state.iter, g_global = state.ϵ)
    return recalculate!(state.envs, mps, state.operator..., alg_environments)
end
