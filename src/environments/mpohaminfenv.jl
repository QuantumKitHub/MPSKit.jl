"
    This object manages the hamiltonian environments for an InfiniteMPS
"
# mutable struct MPOHamInfEnv{H<:MPOHamiltonian,V,S<:InfiniteMPS,A} <: AbstractInfEnv
#     opp::H
#
#     dependency::S
#     solver::A
#
#     lw::PeriodicArray{V,2}
#     rw::PeriodicArray{V,2}
#
#     lock::ReentrantLock
# end

# function Base.getproperty(env::MPOHamInfEnv, name::Symbol)
#     return name === :operator ? getfield(env, :opp) : getfield(env, name)
# end

mutable struct InfiniteEnvironments{O,V,S,A} <: AbstractInfEnv
    operator::O
    dependency::S

    solver::A

    leftenvs::PeriodicVector{V}
    rightenvs::PeriodicVector{V}

    lock::ReentrantLock
end

# function Base.copy(p::MPOHamInfEnv)
#     return MPOHamInfEnv(p.opp, p.dependency, p.solver, copy(p.lw), copy(p.rw))
# end;

# function gen_lw_rw(ψ::InfiniteMPS{A}, O::Union{SparseMPO,MPOHamiltonian}) where {A}
#     lw = PeriodicArray{A,2}(undef, O.odim, length(ψ))
#     rw = PeriodicArray{A,2}(undef, O.odim, length(ψ))
#
#     for i in 1:length(ψ), j in 1:(O.odim)
#         lw[j, i] = similar(ψ.AL[1],
#                            _firstspace(ψ.AL[i]) * O[i].domspaces[j]' ←
#                            _firstspace(ψ.AL[i]))
#         rw[j, i] = similar(ψ.AL[1],
#                            _lastspace(ψ.AR[i])' * O[i].imspaces[j]' ←
#                            _lastspace(ψ.AR[i])')
#     end
#
#     randomize!.(lw)
#     randomize!.(rw)
#
#     return (lw, rw)
# end

function initialize_environments(state::InfiniteMPS, H::InfiniteMPOHamiltonian)
    # allocate
    GL = PeriodicArray([allocate_GL(state, H, state, i) for i in 1:length(state)])
    GR = PeriodicArray([allocate_GR(state, H, state, i) for i in 1:length(state)])

    # initialize:
    # GL = (1, 0, 0)
    for i in 1:length(GL[1])
        if i == 1
            GL[1][i] = isomorphism(storagetype(eltype(GL)), space(GL[1][i]))
        else
            fill!(GL[1][i], zero(scalartype(GL)))
        end
    end
    # GR = (0, 0, 1)^T
    for i in 1:length(GR[end])
        if i == length(GR[end])
            GR[end][i] = isomorphism(storagetype(eltype(GR)), space(GR[end][i]))
        else
            fill!(GR[end][i], zero(scalartype(GR)))
        end
    end

    return GL, GR
end

#randomly initialize envs
# function environments(ψ::InfiniteMPS, H::MPOHamiltonian; solver=Defaults.linearsolver)
#     (lw, rw) = gen_lw_rw(ψ, H)
#     envs = MPOHamInfEnv(H, similar(ψ), solver, lw, rw, ReentrantLock())
#     return recalculate!(envs, ψ)
# end

function environments(state::InfiniteMPS, H::InfiniteMPOHamiltonian;
                      solver=Defaults.linearsolver)
    GL, GR = initialize_environments(state, H)
    envs = InfiniteEnvironments(H, state, solver, GL, GR, ReentrantLock())
    return recalculate!(envs, state)
end

# function leftenv(envs::MPOHamInfEnv, pos::Int, ψ)
#     check_recalculate!(envs, ψ)
#     return envs.lw[:, pos]
# end
function leftenv(envs::InfiniteEnvironments, pos::Int, ψ)
    check_recalculate!(envs, ψ)
    return envs.leftenvs[pos]
end

# function rightenv(envs::MPOHamInfEnv, pos::Int, ψ)
#     check_recalculate!(envs, ψ)
#     return envs.rw[:, pos]
# end
function rightenv(envs::InfiniteEnvironments, pos::Int, ψ)
    check_recalculate!(envs, ψ)
    return envs.rightenvs[pos]
end

# function recalculate!(envs::MPOHamInfEnv, nstate; tol=envs.solver.tol)
#     sameDspace = reduce(&, _lastspace.(envs.lw[1, :]) .== _firstspace.(nstate.CR))
#
#     if !sameDspace
#         envs.lw, envs.rw = gen_lw_rw(nstate, envs.opp)
#     end
#
#     solver = envs.solver
#     solver = solver.tol == tol ? solver : @set solver.tol = tol
#     @sync begin
#         Threads.@spawn calclw!(envs.lw, nstate, envs.opp; solver)
#         Threads.@spawn calcrw!(envs.rw, nstate, envs.opp; solver)
#     end
#
#     envs.dependency = nstate
#     envs.solver = solver
#
#     return envs
# end

function recalculate!(envs::InfiniteEnvironments, state::InfiniteMPS; tol=envs.solver.tol)
    # check if the virtual spaces have changed and reallocate if necessary
    samespaces = all(only(_lastspace(envs.leftenvs[i])) == left_virtualspace(state, i)
                     for i in 1:length(state))
    if !samespaces
        envs.leftenvs, envs.rightenvs = initialize_environments(state, envs.operator)
    end

    calclw!(envs.leftenvs, state, envs.operator; solver=envs.solver)
    calcrw!(envs.rightenvs, state, envs.operator; solver=envs.solver)

    envs.dependency = state
    return envs
end

# function calclw!(fixpoints, st::InfiniteMPS, H::MPOHamiltonian;
#                  solver=Defaults.linearsolver)
#     len = length(st)
#     @assert len == length(H)
#
#     #the start element
#     leftutil = similar(st.AL[1], H[1].domspaces[1])
#     fill_data!(leftutil, one)
#
#     @plansor fixpoints[1, 1][-1 -2; -3] = l_LL(st)[-1; -3] * conj(leftutil[-2])
#     (len > 1) && left_cyclethrough!(1, fixpoints, H, st)
#     for i in 2:size(fixpoints, 1)
#         prev = copy(fixpoints[i, 1])
#
#         rmul!(fixpoints[i, 1], 0)
#         left_cyclethrough!(i, fixpoints, H, st)
#
#         if isid(H, i) # identity matrices; do the hacky renormalization
#             tm = regularize(TransferMatrix(st.AL, st.AL), l_LL(st), r_LL(st))
#             fixpoints[i, 1], convhist = linsolve(flip(tm), fixpoints[i, 1], prev, solver,
#                                                  1, -1)
#             convhist.converged == 0 &&
#                 @warn "GL$i failed to converge: normres = $(convhist.normres)"
#
#             (len > 1) && left_cyclethrough!(i, fixpoints, H, st)
#
#             #go through the unitcell, again subtracting fixpoints
#             for potato in 1:len
#                 @plansor fixpoints[i, potato][-1 -2; -3] -= fixpoints[i, potato][1 -2; 2] *
#                                                             r_LL(st, potato - 1)[2; 1] *
#                                                             l_LL(st, potato)[-1; -3]
#             end
#
#         else
#             if reduce(&, contains.(H.data, i, i))
#                 diag = map(b -> b[i, i], H[:])
#                 tm = TransferMatrix(st.AL, diag, st.AL)
#                 fixpoints[i, 1], convhist = linsolve(flip(tm), fixpoints[i, 1], prev,
#                                                      solver, 1, -1)
#                 convhist.converged == 0 &&
#                     @warn "GL$i failed to converge: normres = $(convhist.normres)"
#             end
#             (len > 1) && left_cyclethrough!(i, fixpoints, H, st)
#         end
#     end
#
#     return fixpoints
# end

function calclw!(GL, state::InfiniteMPS, H::InfiniteMPOHamiltonian;
                 solver=Defaults.linearsolver)
    L = check_length(state, H, GL)

    # the start element
    # TODO: check if this is necessary
    leftutil = similar(state.AL[1], space(GL[1], 2)[1])
    fill_data!(leftutil, one)
    ρ_left = l_LL(state)
    @plansor GL[1][1][-1 -2; -3] = ρ_left[-1; -3] * leftutil[-2]

    (L > 1) && left_cyclethrough!(1, GL, H, state)

    for i in 2:length(GL[1])
        prev = copy(GL[1][i])
        zerovector!(GL[1][i])
        left_cyclethrough!(i, GL, H, state)

        if isidentitylevel(H, i) # identity matrices; do the hacky renormalization
            T = regularize(TransferMatrix(state.AL, state.AL), ρ_left, r_LL(state))
            GL[1][i], convhist = linsolve(flip(T), GL[1][i], prev, solver, 1, -1)
            convhist.converged == 0 &&
                @warn "GL$i failed to converge: normres = $(convhist.normres)"

            (L > 1) && left_cyclethrough!(i, GL, H, state)

            # go through the unitcell, again subtracting fixpoints
            for site in 1:L
                @plansor GL[site][i][-1 -2; -3] -= GL[site][i][1 -2; 2] *
                                                   r_LL(state, site - 1)[2; 1] *
                                                   l_LL(state, site)[-1; -3]
            end

        else
            if !isemptylevel(H, i)
                diag = map(h -> h[i, 1, 1, i], H[:])
                T = TransferMatrix(state.AL, diag, state.AL)
                GL[1][i], convhist = linsolve(flip(T), GL[1][i], prev, solver, 1, -1)
                convhist.converged == 0 &&
                    @warn "GL$i failed to converge: normres = $(convhist.normres)"
            end
            (L > 1) && left_cyclethrough!(i, GL, H, state)
        end
    end

    return GL
end

# function calcrw!(fixpoints, st::InfiniteMPS, H::MPOHamiltonian;
#                  solver=Defaults.linearsolver)
#     len = length(st)
#     odim = size(fixpoints, 1)
#     @assert len == length(H)
#
#     #the start element
#     rightutil = similar(st.AL[1], H[len].imspaces[1])
#     fill_data!(rightutil, one)
#     @plansor fixpoints[end, end][-1 -2; -3] = r_RR(st)[-1; -3] * conj(rightutil[-2])
#     (len > 1) && right_cyclethrough!(odim, fixpoints, H, st) #populate other sites
#
#     for i in (odim - 1):-1:1
#         prev = copy(fixpoints[i, end])
#         rmul!(fixpoints[i, end], 0)
#         right_cyclethrough!(i, fixpoints, H, st)
#
#         if isid(H, i) #identity matrices; do the hacky renormalization
#
#             #subtract fixpoints
#             tm = regularize(TransferMatrix(st.AR, st.AR), l_RR(st), r_RR(st))
#             fixpoints[i, end], convhist = linsolve(tm, fixpoints[i, end], prev, solver, 1,
#                                                    -1)
#             convhist.converged == 0 &&
#                 @warn "GR$i failed to converge: normres = $(convhist.normres)"
#
#             len > 1 && right_cyclethrough!(i, fixpoints, H, st)
#
#             #go through the unitcell, again subtracting fixpoints
#             for potatoe in 1:len
#                 @plansor fixpoints[i, potatoe][-1 -2; -3] -= fixpoints[i, potatoe][1 -2;
#                                                                                    2] *
#                                                              l_RR(st, potatoe + 1)[2; 1] *
#                                                              r_RR(st, potatoe)[-1; -3]
#             end
#         else
#             if reduce(&, contains.(H.data, i, i))
#                 diag = map(b -> b[i, i], H[:])
#                 tm = TransferMatrix(st.AR, diag, st.AR)
#                 fixpoints[i, end], convhist = linsolve(tm, fixpoints[i, end], prev,
#                                                        solver, 1, -1)
#                 convhist.converged == 0 &&
#                     @warn "GR$i failed to converge: normres = $(convhist.normres)"
#             end
#
#             (len > 1) && right_cyclethrough!(i, fixpoints, H, st)
#         end
#     end
#
#     return fixpoints
# end

function calcrw!(GR, state::InfiniteMPS, H::InfiniteMPOHamiltonian;
                 solver=Defaults.linearsolver)
    L = check_length(GR, state)
    odim = length(GR[end])

    # the start element
    rightutil = similar(state.AL[1], space(GR[end], 2)[end])
    fill_data!(rightutil, one)
    @plansor GR[end][end][-1 -2; -3] = r_RR(state)[-1; -3] * rightutil[-2]

    (L > 1) && right_cyclethrough!(odim, GR, H, state) # populate other sites

    for i in (odim - 1):-1:1
        prev = copy(GR[end][i])
        zerovector!(GR[end][i])
        right_cyclethrough!(i, GR, H, state)

        if isidentitylevel(H, i) # identity matrices; do the hacky renormalization
            # subtract fixpoints
            T = regularize(TransferMatrix(state.AR, state.AR), l_RR(state), r_RR(state))
            GR[end][i], convhist = linsolve(T, GR[end][i], prev, solver, 1, -1)
            convhist.converged == 0 &&
                @warn "GR$i failed to converge: normres = $(convhist.normres)"

            L > 1 && right_cyclethrough!(i, GR, H, state)

            # go through the unitcell, again subtracting fixpoints
            for site in 1:L
                @plansor GR[site][i][-1 -2; -3] -= GR[site][i][1 -2; 2] *
                                                   l_RR(state, site + 1)[2; 1] *
                                                   r_RR(state, site)[-1; -3]
            end
        else
            if !isemptylevel(H, i)
                diag = map(b -> b[i, 1, 1, i], H[:])
                T = TransferMatrix(state.AR, diag, state.AR)
                GR[end][i], convhist = linsolve(T, GR[end][i], prev, solver, 1, -1)
                convhist.converged == 0 &&
                    @warn "GR$i failed to converge: normres = $(convhist.normres)"
            end

            (L > 1) && right_cyclethrough!(i, GR, H, state)
        end
    end

    return GR
end
function left_cyclethrough!(index::Int, fp, H, st)
    for i in 1:size(fp, 2)
        rmul!(fp[index, i + 1], 0)

        for j in index:-1:1
            contains(H[i], j, index) || continue

            if isscal(H[i], j, index)
                axpy!(H.Os[i, j, index],
                      fp[j, i] * TransferMatrix(st.AL[i], st.AL[i]),
                      fp[index, i + 1])
            else
                axpy!(true,
                      fp[j, i] * TransferMatrix(st.AL[i], H[i][j, index], st.AL[i]),
                      fp[index, i + 1])
            end
        end
    end
end

function left_cyclethrough!(index::Int, GL, H::InfiniteMPOHamiltonian, state)
    # TODO: efficient transfer matrix slicing for large unitcells
    for site in 1:length(GL)
        leftinds = 1:index
        GL[site + 1][index] = GL[site][leftinds] * TransferMatrix(state.AL[site],
                                                                  H[site][leftinds, 1, 1, index],
                                                                  state.AL[site])
    end
    return GL
end

function right_cyclethrough!(index::Int, fp, H, st)
    for i in size(fp, 2):(-1):1
        rmul!(fp[index, i - 1], 0)

        for j in index:size(fp, 1)
            contains(H[i], index, j) || continue

            if isscal(H[i], index, j)
                axpy!(H.Os[i, index, j],
                      TransferMatrix(st.AR[i], st.AR[i]) * fp[j, i],
                      fp[index, i - 1])
            else
                axpy!(true,
                      TransferMatrix(st.AR[i], H[i][index, j], st.AR[i]) * fp[j, i],
                      fp[index, i - 1])
            end
        end
    end
end

function right_cyclethrough!(index::Int, GR, H::InfiniteMPOHamiltonian, state)
    # TODO: efficient transfer matrix slicing for large unitcells
    L = length(GR)
    for site in reverse(1:L)
        rightinds = index:length(GR[site])
        GR[site - 1][index] = TransferMatrix(state.AR[site],
                                             H[site][index, 1, 1, rightinds],
                                             state.AR[site]) * GR[site][rightinds]
    end
    return GR
end
