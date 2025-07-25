Base.@deprecate(
    approximate(
        ψ::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS}, alg::VUMPS, envs...;
        kwargs...
    ),
    approximate(
        ψ, toapprox,
        VOMPS(; alg.tol, alg.maxiter, alg.finalize, alg.verbosity, alg.alg_gauge, alg.alg_environments),
        envs...; kwargs...
    )
)

function approximate(
        mps::MultilineMPS, toapprox::Tuple{<:MultilineMPO, <:MultilineMPS}, alg::VOMPS,
        envs = environments(mps, toapprox)
    )
    log = IterLog("VOMPS")
    iter = 0
    ϵ = calc_galerkin(mps, toapprox..., envs)
    alg_environments = updatetol(alg.alg_environments, iter, ϵ)
    recalculate!(envs, mps, toapprox...; alg_environments.tol)

    state = VOMPSState(mps, toapprox, envs, iter, ϵ)
    it = IterativeSolver(alg, state)

    return LoggingExtras.withlevel(; alg.verbosity) do
        @infov 2 loginit!(log, ϵ)

        for (mps, envs, ϵ) in it
            if ϵ ≤ alg.tol
                @infov 2 logfinish!(log, it.iter, ϵ)
                return mps, envs, ϵ
            end
            if it.iter ≥ alg.maxiter
                @warnv 1 logcancel!(log, it.iter, ϵ)
                return mps, envs, ϵ
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
    ϵ = calc_galerkin(mps, state.operator..., envs)

    # update state
    it.state = VOMPSState(mps, state.operator, envs, state.iter + 1, ϵ)

    return (mps, envs, ϵ), it.state
end

function localupdate_step!(
        ::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, ::SerialScheduler
    )
    alg_orth = QRpos()

    ACs = similar(state.mps.AC)
    dst_ACs = state.mps isa Multiline ? eachcol(ACs) : ACs

    foreach(eachsite(state.mps)) do site
        AC = map(1:size(state.mps, 1)) do row
            AC_projection(CartesianIndex(row, site), state.mps, state.operator, state.envs)
        end
        circshift!(AC, 1)
        C = map(1:size(state.mps, 1)) do row
            C_projection(CartesianIndex(row, site), state.mps, state.operator, state.envs)
        end
        circshift!(C, 1)
        dst_ACs[site] = regauge!(AC, C; alg = alg_orth)
        return nothing
    end

    return ACs
end
function localupdate_step!(
        ::IterativeSolver{<:VOMPS}, state::VOMPSState{<:Any, <:Tuple}, scheduler
    )
    alg_orth = QRpos()

    ACs = similar(state.mps.AC)
    dst_ACs = state.mps isa Multiline ? eachcol(ACs) : ACs

    tforeach(eachsite(state.mps); scheduler) do site
        local AC, C
        @sync begin
            Threads.@spawn begin
                AC = map(1:size(state.mps, 1)) do row
                    AC_projection(CartesianIndex(row, site), state.mps, state.operator, state.envs)
                end
                circshift!(AC, 1)
            end
            Threads.@spawn begin
                C = map(1:size(state.mps, 1)) do row
                    C_projection(CartesianIndex(row, site), state.mps, state.operator, state.envs)
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
    alg_environments = updatetol(it.alg_environments, state.iter, state.ϵ)
    return recalculate!(state.envs, mps, state.operator...; alg_environments.tol)
end
