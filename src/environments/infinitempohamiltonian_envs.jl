"""
    InfiniteMPOHamiltonianEnvironments{O<:InfiniteMPOHamiltonian,V,S<:InfiniteMPS,A} <: AbstractMPSEnvironments

Environment manager for `InfiniteMPOHamiltonian`.
"""
mutable struct InfiniteMPOHamiltonianEnvironments{O,V,S,A} <: AbstractInfiniteEnvironments
    operator::O
    dependency::S

    solver::A

    leftenvs::PeriodicVector{V}
    rightenvs::PeriodicVector{V}

    lock::ReentrantLock
end
function InfiniteMPOHamiltonianEnvironments(operator, dependency, solver, leftenvs,
                                            rightenvs)
    return InfiniteMPOHamiltonianEnvironments(operator, dependency, solver, leftenvs,
                                              rightenvs, ReentrantLock)
end

# Constructors
# ------------
function environments(state::InfiniteMPS, H::InfiniteMPOHamiltonian;
                      solver=Defaults.linearsolver)
    GL, GR = initialize_environments(state, H)
    envs = InfiniteMPOHamiltonianEnvironments(H, state, solver, GL, GR, ReentrantLock())
    return recalculate!(envs, state)
end

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

# Getter/Setters
# --------------
function leftenv(envs::InfiniteMPOHamiltonianEnvironments, pos::Int, ψ)
    check_recalculate!(envs, ψ)
    return envs.leftenvs[pos]
end

function rightenv(envs::InfiniteMPOHamiltonianEnvironments, pos::Int, ψ)
    check_recalculate!(envs, ψ)
    return envs.rightenvs[pos]
end

# Utility
# -------
function check_dependency(envs::InfiniteMPOHamiltonianEnvironments, state::InfiniteMPS)
    return envs.dependency === state
end

function issamespace(envs::InfiniteMPOHamiltonianEnvironments, state::InfiniteMPS)
    return all(1:length(state)) do i
        return left_virtualspace(envs.dependency, i) == left_virtualspace(state, i)
    end
end

# Calculation
# -----------
function compute_leftenv!(envs::InfiniteMPOHamiltonianEnvironments)
    state = envs.dependency
    H = envs.operator
    GL = envs.leftenvs
    solver = envs.solver
    L = length(state)

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

function compute_rightenv!(envs::InfiniteMPOHamiltonianEnvironments)
    GR = envs.rightenvs
    H = envs.operator
    state = envs.dependency
    solver = envs.solver
    L = length(state)

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

# no normalization here, but need this for consistent interface
TensorKit.normalize!(envs::InfiniteMPOHamiltonianEnvironments) = envs
