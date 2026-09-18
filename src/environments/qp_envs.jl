"""
    InfiniteQPEnvironments <: AbstractMPSEnvironments

Environments for an infinite QP-MPO-QP combination. These solve the corresponding fixedpoint equations:
```math
GLs[i] * T_BL[i] + GBLs[i] * T_RL[i] = GBLs[i + 1]
T_BR[i] * GRs[i] + T_LR[i] * GBRs[i] = GBRs[i - 1]
```
where `T_BL`, `T_BR`, `T_RL` and `T_LR` are the (regularized) transfer matrix operators on a given site for `B-O-AL`, `B-O-AR`, `AR-O-AL` and `AL-O-AR` respectively.
"""
struct InfiniteQPEnvironments{A, B} <: AbstractMPSEnvironments
    leftBenvs::PeriodicVector{A}
    rightBenvs::PeriodicVector{A}

    leftenvs::B
    rightenvs::B
end

Base.length(envs::InfiniteQPEnvironments) = length(envs.leftenvs)

function leftenv(envs::InfiniteQPEnvironments, site::Int, state; kwargs...)
    return leftenv(envs.leftenvs, site, state; kwargs...)
end
function rightenv(envs::InfiniteQPEnvironments, site::Int, state; kwargs...)
    return rightenv(envs.rightenvs, site, state; kwargs...)
end

function environments(
        exci::Union{InfiniteQP, MultilineQP}, operator::Union{InfiniteMPO, InfiniteMPOHamiltonian, MultilineMPO}, above = exci;
        lenvs = environments(exci.left_gs, operator, exci.left_gs), renvs = istopological(exci) ? environments(exci.right_gs, operator, exci.right_gs) : lenvs,
        backend::AbstractBackend = DefaultBackend(), scheduler = Defaults.scheduler[],
        kwargs...
    )
    # `backend` and `scheduler` are named explicitly so that they are not swept into
    # `environment_alg`'s keyword arguments, which describe the linear solver. Same
    # convention as `recalculate!`.
    alg = environment_alg(exci, operator, above; kwargs...)
    return environments(exci, operator, above, alg; lenvs, renvs, backend, scheduler)
end

function environments(
        qp::MultilineQP, operator::MultilineMPO, above, alg; lenvs, renvs = lenvs,
        backend::AbstractBackend = DefaultBackend(), scheduler = Defaults.scheduler[]
    )
    (rows = size(qp, 1)) == size(operator, 1) || throw(ArgumentError("Incompatible sizes"))
    envs = map(1:rows) do row
        return environments(
            qp[row], operator[row], qp[row], alg; lenvs = lenvs[row], renvs = renvs[row],
            backend, scheduler
        )
    end
    return Multiline(PeriodicVector(envs))
end

function environments(
        exci::InfiniteQP, H::InfiniteMPOHamiltonian, above, alg; lenvs, renvs = lenvs,
        backend::AbstractBackend = DefaultBackend(), scheduler = Defaults.scheduler[]
    )
    solver = resolve_environment_solver(alg, exci, H, exci)

    lBs = PeriodicVector([allocate_GBL(exci, H, exci, i) for i in 1:length(exci)])
    rBs = PeriodicVector([allocate_GBR(exci, H, exci, i) for i in 1:length(exci)])
    envs = InfiniteQPEnvironments(lBs, rBs, lenvs, renvs)

    # concurrency depends on scheduler, but each half takes its own dedicated scratch allocator
    tforeach(1:2; scheduler) do half
        allocator = default_allocator(exci.left_gs, SerialScheduler())
        if isone(half)
            compute_leftenvs!(envs, exci, H, above, solver; backend, allocator)
        else
            compute_rightenvs!(envs, exci, H, above, solver; backend, allocator)
        end
        return nothing
    end

    return envs
end

# regularization of trivial excitations
function regularize_GBL!(GBL, exci::InfiniteQP, ids, pos::Int)
    (istrivial(exci) && !isempty(ids)) || return GBL
    ρ_left = l_RL(exci.left_gs, pos + 1)
    ρ_right = r_RL(exci.left_gs, pos)
    for i in ids
        regularize!(GBL[i], ρ_right, ρ_left)
    end
    return GBL
end
function regularize_GBR!(GBR, exci::InfiniteQP, ids, pos::Int)
    (istrivial(exci) && !isempty(ids)) || return GBR
    ρ_left = l_LR(exci.left_gs, pos)
    ρ_right = r_LR(exci.left_gs, pos - 1)
    for i in ids
        regularize!(GBR[i], ρ_left, ρ_right)
    end
    return GBR
end

function compute_leftenvs!(
        envs::InfiniteQPEnvironments, exci::InfiniteQP, H::InfiniteMPOHamiltonian, above, alg;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    lBs, lenvs = envs.leftBenvs, envs.leftenvs
    ids = findall(Base.Fix1(isidentitylevel, H), 2:(size(H[1], 1) - 1)) .+ 1
    AL, AR = exci.left_gs.AL, exci.right_gs.AR

    # push through unitcell to obtain the inhomogeneity
    zerovector!(lBs[1])
    for pos in 1:length(exci)
        lBs[pos + 1] = lBs[pos] * TransferMatrix(AR[pos], H[pos], AL[pos]; backend, allocator) /
            cis(exci.momentum)
        lBs[pos + 1] += leftenv(lenvs, pos, exci.left_gs; backend, allocator) *
            TransferMatrix(exci[pos], H[pos], AL[pos]; backend, allocator) / cis(exci.momentum)
        regularize_GBL!(lBs[pos + 1], exci, ids, pos)
    end

    # solve the fixed point equation
    lBs[1] = left_excitation_transfer_system(
        lBs[1], H, exci; solver = alg, backend, allocator
    )

    # push the solution through the unitcell
    lB_cur = lBs[1]
    for i in 1:(length(exci) - 1)
        lB_cur = lB_cur * TransferMatrix(AR[i], H[i], AL[i]; backend, allocator) / cis(exci.momentum)
        regularize_GBL!(lB_cur, exci, ids, i)
        lBs[i + 1] += lB_cur
    end

    return envs
end

function compute_rightenvs!(
        envs::InfiniteQPEnvironments, exci::InfiniteQP, H::InfiniteMPOHamiltonian, above, alg;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    rBs, renvs = envs.rightBenvs, envs.rightenvs
    ids = findall(Base.Fix1(isidentitylevel, H), 2:(size(H[1], 1) - 1)) .+ 1
    AL, AR = exci.left_gs.AL, exci.right_gs.AR

    # push through unitcell to obtain the inhomogeneity
    zerovector!(rBs[end])
    for pos in length(exci):-1:1
        rBs[pos - 1] = TransferMatrix(AL[pos], H[pos], AR[pos]; backend, allocator) *
            rBs[pos] * cis(exci.momentum)
        rBs[pos - 1] += TransferMatrix(exci[pos], H[pos], AR[pos]; backend, allocator) *
            rightenv(renvs, pos, exci.right_gs; backend, allocator) * cis(exci.momentum)
        regularize_GBR!(rBs[pos - 1], exci, ids, pos)
    end

    # solve the fixed point equation
    rBs[end] = right_excitation_transfer_system(
        rBs[end], H, exci; solver = alg, backend, allocator
    )

    # push the solution through the unitcell
    rB_cur = rBs[end]
    for i in length(exci):-1:2
        rB_cur = TransferMatrix(AL[i], H[i], AR[i]; backend, allocator) * rB_cur * cis(exci.momentum)
        regularize_GBR!(rB_cur, exci, ids, i)
        rBs[i - 1] += rB_cur
    end

    return envs
end

function environments(
        exci::FiniteQP, H::FiniteMPOHamiltonian, above = exci, alg = nothing;
        lenvs = environments(exci.left_gs, H, exci.left_gs),
        renvs = istopological(exci) ? environments(exci.right_gs, H, exci.right_gs) : lenvs,
        backend::AbstractBackend = DefaultBackend()
    )
    # the sweeps below are serial, so a single allocator serves the whole chain
    allocator = default_allocator(exci.left_gs, SerialScheduler())

    AL = exci.left_gs.AL
    AR = exci.right_gs.AR

    #construct lBE
    # TODO: should not have to be periodic
    lBs = PeriodicVector([allocate_GBL(exci, H, exci, i) for i in 1:length(exci)])
    rBs = PeriodicVector([allocate_GBR(exci, H, exci, i) for i in 1:length(exci)])

    zerovector!(lBs[1])
    for pos in 1:(length(exci) - 1)
        lBs[pos + 1] = lBs[pos] * TransferMatrix(AR[pos], H[pos], AL[pos]; backend, allocator)
        lBs[pos + 1] += leftenv(lenvs, pos, exci.left_gs) *
            TransferMatrix(exci[pos], H[pos], AL[pos]; backend, allocator)
    end

    zerovector!(rBs[end])
    for pos in length(exci):-1:2
        rBs[pos - 1] = TransferMatrix(AL[pos], H[pos], AR[pos]; backend, allocator) * rBs[pos]
        rBs[pos - 1] += TransferMatrix(exci[pos], H[pos], AR[pos]; backend, allocator) *
            rightenv(renvs, pos, exci.right_gs)
    end

    return InfiniteQPEnvironments(lBs, rBs, lenvs, renvs)
end

function environments(
        exci::InfiniteQP, O::InfiniteMPO, above, alg; lenvs, renvs,
        backend::AbstractBackend = DefaultBackend(), scheduler = Defaults.scheduler[]
    )
    istopological(exci) &&
        @warn "there is a phase ambiguity in topologically nontrivial statmech excitations"
    solver = resolve_environment_solver(alg, exci, O, exci)

    GBL = PeriodicVector([allocate_GBL(exci, O, exci, i) for i in 1:length(exci)])
    GBR = PeriodicVector([allocate_GBR(exci, O, exci, i) for i in 1:length(exci)])
    envs = InfiniteQPEnvironments(GBL, GBR, lenvs, renvs)

    # Which of the two halves run concurrently is the scheduler's call, but neither half fans out
    # any further and they never share their scratch, so each takes a buffer of its own instead of
    # the whole computation falling back on a shared allocator as soon as the scheduler spawns.
    tforeach(1:2; scheduler) do half
        allocator = default_allocator(exci.left_gs, SerialScheduler())
        if isone(half)
            compute_leftenvs!(envs, exci, O, above, solver; backend, allocator)
        else
            compute_rightenvs!(envs, exci, O, above, solver; backend, allocator)
        end
        return nothing
    end

    return envs
end

function compute_leftenvs!(
        envs::InfiniteQPEnvironments, exci::InfiniteQP, O::InfiniteMPO, above, alg;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    GBL, lenvs = envs.leftBenvs, envs.leftenvs
    left_gs, right_gs = exci.left_gs, exci.right_gs

    regularization = map(1:length(exci)) do site
        GL = leftenv(lenvs, site, left_gs)
        GR = rightenv(lenvs, site, left_gs)
        return inv(contract_mpo_expval(left_gs.AC[site], GL, O[site], GR))
    end

    # push through unitcell to obtain the inhomogeneity
    # GBL[i] lives on the left MPO bond of site i. Applying site i therefore
    # produces an object in GBL[i + 1]. This matters when the MPO bond spaces
    # vary within the unit cell, as for a finite-ring shift MPO.
    gbl = zerovector!(GBL[1])
    for col in 1:length(exci)
        gbl = gbl * TransferMatrix(right_gs.AR[col], O[col], left_gs.AL[col]; backend, allocator)
        gbl += leftenv(lenvs, col, left_gs) *
            TransferMatrix(exci[col], O[col], left_gs.AL[col]; backend, allocator)
        gbl *= regularization[col] * cis(-exci.momentum)
        GBL[col + 1] = gbl
    end

    # solve the fixed point equation
    T_RL = TransferMatrix(right_gs.AR, O, left_gs.AL; backend, allocator)
    if istrivial(exci)
        @plansor rvec[-1 -2; -3] := rightenv(lenvs, 0, left_gs)[-1 -2; 1] *
            conj(left_gs.C[0][-3; 1])
        @plansor lvec[-1 -2; -3] := leftenv(lenvs, 1, left_gs)[-1 -2; 1] *
            left_gs.C[0][1; -3]
        T_RL = regularize(T_RL, lvec, rvec)
    end

    GBL[1], convhist = linsolve(
        flip(T_RL), gbl, gbl, alg, 1,
        -cis(-length(exci) * exci.momentum) * prod(regularization)
    )
    convhist.converged == 0 &&
        @warn "GBL failed to converge: normres = $(convhist.normres)"

    # push the solution through the unitcell
    left_cur = GBL[1]
    for col in 1:(length(exci) - 1)
        left_cur = regularization[col] * left_cur *
            TransferMatrix(right_gs.AR[col], O[col], left_gs.AL[col]; backend, allocator) *
            cis(-exci.momentum)
        GBL[col + 1] += left_cur
    end

    return envs
end

function compute_rightenvs!(
        envs::InfiniteQPEnvironments, exci::InfiniteQP, O::InfiniteMPO, above, alg;
        backend::AbstractBackend = DefaultBackend(), allocator = DefaultAllocator()
    )
    GBR, renvs = envs.rightBenvs, envs.rightenvs
    left_gs, right_gs = exci.left_gs, exci.right_gs

    regularization = map(1:length(exci)) do site
        GL = leftenv(renvs, site, right_gs)
        GR = rightenv(renvs, site, right_gs)
        return inv(contract_mpo_expval(right_gs.AC[site], GL, O[site], GR))
    end

    # push through unitcell to obtain the inhomogeneity
    # GBR[i] lives on the right MPO bond of site i. Applying site i from the
    # right therefore produces an object in GBR[i - 1].
    gbr = zerovector!(GBR[end])
    for col in reverse(1:length(exci))
        gbr = TransferMatrix(left_gs.AL[col], O[col], right_gs.AR[col]; backend, allocator) * gbr
        gbr += TransferMatrix(exci[col], O[col], right_gs.AR[col]; backend, allocator) *
            rightenv(renvs, col, right_gs)
        gbr *= regularization[col] * cis(exci.momentum)
        GBR[col - 1] = gbr
    end

    # solve the fixed point equation
    T_LR = TransferMatrix(left_gs.AL, O, right_gs.AR; backend, allocator)
    if istrivial(exci)
        @plansor rvec[-1 -2; -3] := rightenv(renvs, 0, right_gs)[1 -2; -3] *
            right_gs.C[0][-1; 1]
        @plansor lvec[-1 -2; -3] := conj(right_gs.C[0][-3; 1]) *
            leftenv(renvs, 1, right_gs)[-1 -2; 1]
        T_LR = regularize(T_LR, lvec, rvec)
    end

    GBR[end], convhist = linsolve(
        T_LR, gbr, gbr, alg, 1,
        -cis(length(exci) * exci.momentum) * prod(regularization)
    )
    convhist.converged == 0 &&
        @warn "GBR failed to converge: normres = $(convhist.normres)"

    # push the solution through the unitcell
    right_cur = GBR[end]
    for col in reverse(2:length(exci))
        right_cur = TransferMatrix(left_gs.AL[col], O[col], right_gs.AR[col]; backend, allocator) *
            right_cur * cis(exci.momentum) * regularization[col]
        GBR[col - 1] += right_cur
    end

    return envs
end
