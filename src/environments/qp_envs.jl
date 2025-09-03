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

function leftenv(envs::InfiniteQPEnvironments, site::Int, state)
    return leftenv(envs.leftenvs, site, state)
end
function rightenv(envs::InfiniteQPEnvironments, site::Int, state)
    return rightenv(envs.rightenvs, site, state)
end

# Explicitly define optional arguments as these depend on kwargs,
# which needs to come after these arguments.

function environments(exci::Union{InfiniteQP, MultilineQP}, H; kwargs...)
    lenvs = environments(exci.left_gs, H; kwargs...)
    return environments(exci, H, lenvs; kwargs...)
end
function environments(exci::Union{InfiniteQP, MultilineQP}, H, lenvs; kwargs...)
    renvs = !istopological(exci) ? lenvs : environments(exci.right_gs, H; kwargs...)
    return environments(exci, H, lenvs, renvs; kwargs...)
end

function environments(qp::MultilineQP, operator::MultilineMPO, lenvs, renvs; kwargs...)
    (rows = size(qp, 1)) == size(operator, 1) || throw(ArgumentError("Incompatible sizes"))
    envs = map(1:rows) do row
        return environments(qp[row], operator[row], lenvs[row], renvs[row]; kwargs...)
    end
    return Multiline(PeriodicVector(envs))
end

function environments(exci::InfiniteQP, H::InfiniteMPOHamiltonian, lenvs, renvs; kwargs...)
    ids = findall(Base.Fix1(isidentitylevel, H), 2:(size(H[1], 1) - 1)) .+ 1
    solver = environment_alg(exci, H, exci; kwargs...)

    AL = exci.left_gs.AL
    AR = exci.right_gs.AR

    lBs = PeriodicVector([allocate_GBL(exci, H, exci, i) for i in 1:length(exci)])
    rBs = PeriodicVector([allocate_GBR(exci, H, exci, i) for i in 1:length(exci)])

    zerovector!(lBs[1])
    for pos in 1:length(exci)
        lBs[pos + 1] = lBs[pos] * TransferMatrix(AR[pos], H[pos], AL[pos]) /
            cis(exci.momentum)
        lBs[pos + 1] += leftenv(lenvs, pos, exci.left_gs) *
            TransferMatrix(exci[pos], H[pos], AL[pos]) / cis(exci.momentum)

        if istrivial(exci) && !isempty(ids) # regularization of trivial excitations
            ρ_left = l_RL(exci.left_gs, pos + 1)
            ρ_right = r_RL(exci.left_gs, pos)
            for i in ids
                regularize!(lBs[pos + 1][i], ρ_right, ρ_left)
            end
        end
    end

    zerovector!(rBs[end])
    for pos in length(exci):-1:1
        rBs[pos - 1] = TransferMatrix(AL[pos], H[pos], AR[pos]) *
            rBs[pos] * cis(exci.momentum)
        rBs[pos - 1] += TransferMatrix(exci[pos], H[pos], AR[pos]) *
            rightenv(renvs, pos, exci.right_gs) * cis(exci.momentum)

        if istrivial(exci) && !isempty(ids)
            ρ_left = l_LR(exci.left_gs, pos)
            ρ_right = r_LR(exci.left_gs, pos - 1)
            for i in ids
                regularize!(rBs[pos - 1][i], ρ_left, ρ_right)
            end
        end
    end

    @sync begin
        Threads.@spawn $lBs[1] = left_excitation_transfer_system(
            $lBs[1], $H, $exci; solver = $solver
        )
        Threads.@spawn $rBs[end] = right_excitation_transfer_system(
            $rBs[end], $H, $exci; solver = $solver
        )
    end

    lB_cur = lBs[1]
    for i in 1:(length(exci) - 1)
        lB_cur = lB_cur * TransferMatrix(AR[i], H[i], AL[i]) / cis(exci.momentum)

        if istrivial(exci) && !isempty(ids)
            ρ_left = l_RL(exci.left_gs, i + 1)
            ρ_right = r_RL(exci.left_gs, i)
            for k in ids
                regularize!(lB_cur[k], ρ_right, ρ_left)
            end
        end

        lBs[i + 1] += lB_cur
    end

    rB_cur = rBs[end]
    for i in length(exci):-1:2
        rB_cur = TransferMatrix(AL[i], H[i], AR[i]) * rB_cur * cis(exci.momentum)

        if istrivial(exci) && !isempty(ids)
            ρ_left = l_LR(exci.left_gs, i)
            ρ_right = r_LR(exci.left_gs, i - 1)
            for k in ids
                regularize!(rB_cur[k], ρ_left, ρ_right)
            end
        end

        rBs[i - 1] += rB_cur
    end

    return InfiniteQPEnvironments(lBs, rBs, lenvs, renvs)
end

function environments(
        exci::FiniteQP, H::FiniteMPOHamiltonian,
        lenvs = environments(exci.left_gs, H),
        renvs = !istopological(exci) ? lenvs : environments(exci.right_gs, H);
        kwargs...
    )
    AL = exci.left_gs.AL
    AR = exci.right_gs.AR

    #construct lBE
    # TODO: should not have to be periodic
    lBs = PeriodicVector([allocate_GBL(exci, H, exci, i) for i in 1:length(exci)])
    rBs = PeriodicVector([allocate_GBR(exci, H, exci, i) for i in 1:length(exci)])

    zerovector!(lBs[1])
    for pos in 1:(length(exci) - 1)
        lBs[pos + 1] = lBs[pos] * TransferMatrix(AR[pos], H[pos], AL[pos])
        lBs[pos + 1] += leftenv(lenvs, pos, exci.left_gs) *
            TransferMatrix(exci[pos], H[pos], AL[pos])
    end

    zerovector!(rBs[end])
    for pos in length(exci):-1:2
        rBs[pos - 1] = TransferMatrix(AL[pos], H[pos], AR[pos]) * rBs[pos]
        rBs[pos - 1] += TransferMatrix(exci[pos], H[pos], AR[pos]) *
            rightenv(renvs, pos, exci.right_gs)
    end

    return InfiniteQPEnvironments(lBs, rBs, lenvs, renvs)
end

function environments(exci::InfiniteQP, O::InfiniteMPO, lenvs, renvs; kwargs...)
    istopological(exci) &&
        @warn "there is a phase ambiguity in topologically nontrivial statmech excitations"
    solver = environment_alg(exci, O, exci; kwargs...)

    left_gs = exci.left_gs
    right_gs = exci.right_gs

    GBL = PeriodicVector([allocate_GBL(exci, O, exci, i) for i in 1:length(exci)])
    GBR = PeriodicVector([allocate_GBR(exci, O, exci, i) for i in 1:length(exci)])

    left_regularization = map(1:length(exci)) do site
        GL = leftenv(lenvs, site, left_gs)
        GR = rightenv(lenvs, site, left_gs)
        return inv(contract_mpo_expval(left_gs.AC[site], GL, O[site], GR))
    end
    right_regularization = map(1:length(exci)) do site
        GL = leftenv(renvs, site, right_gs)
        GR = rightenv(renvs, site, right_gs)
        return inv(contract_mpo_expval(right_gs.AC[site], GL, O[site], GR))
    end

    gbl = zerovector!(GBL[end])
    for col in 1:length(exci)
        gbl = gbl * TransferMatrix(right_gs.AR[col], O[col], left_gs.AL[col])
        gbl += leftenv(lenvs, col, left_gs) *
            TransferMatrix(exci[col], O[col], left_gs.AL[col])
        gbl *= left_regularization[col] * cis(-exci.momentum)
        GBL[col] = gbl
    end

    gbr = zerovector!(GBR[end])
    for col in reverse(1:length(exci))
        gbr = TransferMatrix(left_gs.AL[col], O[col], right_gs.AR[col]) * gbr
        gbr += TransferMatrix(exci[col], O[col], right_gs.AR[col]) *
            rightenv(renvs, col, right_gs)
        gbr *= right_regularization[col] * cis(exci.momentum)
        GBR[col] = gbr
    end

    T_RL = TransferMatrix(right_gs.AR, O, left_gs.AL)
    T_LR = TransferMatrix(left_gs.AL, O, right_gs.AR)

    if istrivial(exci)
        @plansor rvec[-1 -2; -3] := rightenv(lenvs, 0, left_gs)[-1 -2; 1] *
            conj(left_gs.C[0][-3; 1])
        @plansor lvec[-1 -2; -3] := leftenv(lenvs, 1, left_gs)[-1 -2; 1] *
            left_gs.C[0][1; -3]

        T_RL = regularize(T_RL, lvec, rvec)

        @plansor rvec[-1 -2; -3] := rightenv(renvs, 0, right_gs)[1 -2; -3] *
            right_gs.C[0][-1; 1]
        @plansor lvec[-1 -2; -3] := conj(right_gs.C[0][-3; 1]) *
            leftenv(renvs, 1, right_gs)[-1 -2; 1]

        T_LR = regularize(T_LR, lvec, rvec)
    end

    GBL[end], convhist = linsolve(
        flip(T_RL), gbl, gbl, solver, 1,
        -cis(-length(exci) * exci.momentum) * prod(left_regularization)
    )

    convhist.converged == 0 &&
        @warn "GBL failed to converge: normres = $(convhist.normres)"

    GBR[1], convhist = linsolve(
        T_LR, gbr, gbr, GMRES(), 1,
        -cis(length(exci) * exci.momentum) * prod(right_regularization)
    )
    convhist.converged == 0 &&
        @warn "GBR failed to converge: normres = $(convhist.normres)"

    left_cur = GBL[end]
    right_cur = GBR[1]
    for col in 1:(length(exci) - 1)
        left_cur = left_regularization[col] * left_cur *
            TransferMatrix(right_gs.AR[col], O[col], left_gs.AL[col]) *
            cis(-exci.momentum)
        GBL[col] += left_cur

        col = length(exci) - col + 1
        right_cur = TransferMatrix(left_gs.AL[col], O[col], right_gs.AR[col]) * right_cur *
            cis(exci.momentum) * right_regularization[col]
        GBR[col] += right_cur
    end

    return InfiniteQPEnvironments(GBL, GBR, lenvs, renvs)
end
