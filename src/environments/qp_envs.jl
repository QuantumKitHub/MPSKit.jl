#=
nothing fancy - only used internally (and therefore cryptic) - stores some partially contracted things
seperates out this bit of logic from effective_excitation_hamiltonian (now more readable)
can also - potentially - partially reuse this in other algorithms
=#
struct QPEnv{A,B} <: AbstractMPSEnvironments
    lBs::PeriodicArray{A,2}
    rBs::PeriodicArray{A,2}

    lenvs::B
    renvs::B
end

struct QuasiparticleEnvironments{A,B} <: AbstractMPSEnvironments
    leftBenvs::PeriodicVector{A}
    rightBenvs::PeriodicVector{A}

    leftenvs::B
    rightenvs::B
end

function environments(exci::Union{InfiniteQP,Multiline{<:InfiniteQP}}, H;
                      solver=Defaults.linearsolver)
    # Explicitly define optional arguments as these depend on solver,
    # which needs to come after these arguments.
    lenvs = environments(exci.left_gs, H; solver=solver)

    return environments(exci, H, lenvs; solver=solver)
end

function environments(exci::Union{InfiniteQP,Multiline{<:InfiniteQP}}, H, lenvs;
                      solver=Defaults.linearsolver)
    # Explicitly define optional arguments as these depend on solver,
    # which needs to come after these arguments.
    renvs = exci.trivial ? lenvs : environments(exci.right_gs, H; solver=solver)

    return environments(exci, H, lenvs, renvs; solver=solver)
end

function gen_exci_lw_rw(left_gs::InfiniteMPS, H::Union{InfiniteMPO,InfiniteMPOHamiltonian},
                        right_gs,
                        excileg)
    B = tensormaptype(spacetype(left_gs), 2, 2, storagetype(eltype(left_gs)))
    BB = tensormaptype(sumspacetype(spacetype(B)), 2, 2, B)

    GBL = PeriodicVector{BB}(undef, length(left_gs))
    GBR = PeriodicVector{BB}(undef, length(left_gs))

    for site in 1:length(left_gs)
        GBL[site] = BB(undef,
                       left_virtualspace(left_gs, site - 1) ⊗ left_virtualspace(H, site)' ←
                       excileg' ⊗ left_virtualspace(right_gs, site - 1))
        fill!(GBL[site], zero(scalartype(B)))

        GBR[site] = BB(undef,
                       right_virtualspace(left_gs, site)' ⊗ right_virtualspace(H, site)' ←
                       excileg' ⊗ right_virtualspace(right_gs, site)')
        fill!(GBR[site], zero(scalartype(B)))
    end

    return GBL, GBR
end

function environments(exci::InfiniteQP, H::InfiniteMPOHamiltonian, lenvs, renvs;
                      solver=Defaults.linearsolver)
    ids = findall(Base.Fix1(isidentitylevel, H), 2:(size(H[1], 1) - 1))
    # ids = collect(Iterators.filter(x -> isidentitylevel(H, x), 2:(H.odim - 1)))

    AL = exci.left_gs.AL
    AR = exci.right_gs.AR

    lBs = PeriodicVector([allocate_GBL(exci, H, exci, i) for i in 1:length(exci)])
    rBs = PeriodicVector([allocate_GBR(exci, H, exci, i) for i in 1:length(exci)])

    # lBs, rBs = gen_exci_lw_rw(exci.left_gs, H, exci.right_gs, space(exci[1], 3))
    zerovector!(lBs[1])
    for pos in 1:length(exci)
        lBs[pos + 1] = lBs[pos] * TransferMatrix(AR[pos], H[pos], AL[pos]) /
                       cis(exci.momentum)
        lBs[pos + 1] += leftenv(lenvs, pos, exci.left_gs) *
                        TransferMatrix(exci[pos], H[pos], AL[pos]) /
                        cis(exci.momentum)

        if exci.trivial # regularization of trivial excitations
            for i in ids
                @plansor lBs[pos + 1][i][-1 -2; -3 -4] -= lBs[pos + 1][i][1 4; -3 2] *
                                                          r_RL(exci.left_gs, pos)[2; 3] *
                                                          τ[3 4; 5 1] *
                                                          l_RL(exci.left_gs, pos + 1)[-1;
                                                                                      6] *
                                                          τ[5 6; -4 -2]
            end
        end
    end

    zerovector!(rBs[end])
    for pos in length(exci):-1:1
        rBs[pos - 1] = TransferMatrix(AL[pos], H[pos], AR[pos]) *
                       rBs[pos] * cis(exci.momentum)
        rBs[pos - 1] += TransferMatrix(exci[pos], H[pos], AR[pos]) *
                        rightenv(renvs, pos, exci.right_gs) * cis(exci.momentum)

        if exci.trivial
            for i in ids
                ρ_left = l_LR(exci.left_gs, pos)
                ρ_right = r_LR(exci.left_gs, pos - 1)
                @plansor rBs[pos - 1][i][-1 -2; -3 -4] -= τ[6 4; 1 3] *
                                                          rBs[pos - 1][i][1 3; -3 2] *
                                                          ρ_left[2; 4] *
                                                          ρ_right[-1; 5] *
                                                          τ[-2 -4; 5 6]
            end
        end
    end

    @sync begin
        Threads.@spawn $lBs[1] = left_excitation_transfer_system($lBs[1], $H, $exci;
                                                                 solver=$solver)
        Threads.@spawn $rBs[end] = right_excitation_transfer_system($rBs[end], $H,
                                                                    $exci;
                                                                    solver=$solver)
    end

    lB_cur = lBs[1]
    for i in 1:(length(exci) - 1)
        lB_cur = lB_cur * TransferMatrix(AR[i], H[i], AL[i]) / cis(exci.momentum)

        if exci.trivial
            for k in ids
                ρ_left = l_RL(exci.left_gs, i + 1)
                ρ_right = r_RL(exci.left_gs, i)
                @plansor lB_cur[k][-1 -2; -3 -4] -= lB_cur[k][1 4; -3 2] *
                                                    ρ_right[2; 3] *
                                                    τ[3 4; 5 1] *
                                                    ρ_left[-1; 6] *
                                                    τ[5 6; -4 -2]
            end
        end

        lBs[i + 1] += lB_cur
    end

    rB_cur = rBs[end]
    for i in length(exci):-1:2
        rB_cur = TransferMatrix(AL[i], H[i], AR[i]) * rB_cur * cis(exci.momentum)

        if exci.trivial
            for k in ids
                ρ_left = l_LR(exci.left_gs, i)
                ρ_right = r_LR(exci.left_gs, i - 1)
                @plansor rB_cur[k][-1 -2; -3 -4] -= τ[6 4; 1 3] *
                                                    rB_cur[k][1 3; -3 2] *
                                                    ρ_left[2; 4] *
                                                    ρ_right[-1; 5] *
                                                    τ[-2 -4; 5 6]
            end
        end

        rBs[i - 1] += rB_cur
    end

    return QuasiparticleEnvironments(lBs, rBs, lenvs, renvs)
end

function environments(exci::FiniteQP,
                      H::FiniteMPOHamiltonian,
                      lenvs=environments(exci.left_gs, H),
                      renvs=exci.trivial ? lenvs : environments(exci.right_gs, H))
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

    return QuasiparticleEnvironments(lBs, rBs, lenvs, renvs)
end

function environments(exci::Multiline{<:InfiniteQP},
                      ham::MultilineMPO,
                      lenvs,
                      renvs;
                      solver=Defaults.linearsolver)
    exci.trivial ||
        @warn "there is a phase ambiguity in topologically nontrivial statmech excitations"

    left_gs = exci.left_gs
    right_gs = exci.right_gs

    exci_space = space(exci[1][1], 3)

    (numrows, numcols) = size(left_gs)

    st = site_type(typeof(left_gs))
    B_type = tensormaptype(spacetype(st), 2, 2, storagetype(st))

    lBs = PeriodicArray{B_type,2}(undef, size(left_gs, 1), size(left_gs, 2))
    rBs = PeriodicArray{B_type,2}(undef, size(left_gs, 1), size(left_gs, 2))

    for row in 1:numrows
        c_lenvs = broadcast(col -> leftenv(lenvs, col, left_gs)[row], 1:numcols)
        c_renvs = broadcast(col -> rightenv(renvs, col, right_gs)[row], 1:numcols)

        hamrow = ham[row, :]

        left_above = left_gs[row]
        left_below = left_gs[row + 1]
        right_above = right_gs[row]
        right_below = right_gs[row + 1]

        left_renorms = fill(zero(scalartype(B_type)), numcols)
        right_renorms = fill(zero(scalartype(B_type)), numcols)

        for col in 1:numcols
            lv = leftenv(lenvs, col, left_gs)[row]
            rv = rightenv(lenvs, col, left_gs)[row]
            left_renorms[col] = @plansor lv[1 2; 3] *
                                         left_above.AC[col][3 4; 5] *
                                         hamrow[col][2 6; 4 7] *
                                         rv[5 7; 8] *
                                         conj(left_below.AC[col][1 6; 8])

            lv = leftenv(renvs, col, right_gs)[row]
            rv = rightenv(renvs, col, right_gs)[row]
            right_renorms[col] = @plansor lv[1 2; 3] *
                                          right_above.AC[col][3 4; 5] *
                                          hamrow[col][2 6; 4 7] *
                                          rv[5 7; 8] *
                                          conj(right_below.AC[col][1 6; 8])
        end

        left_renorms = left_renorms .^ -1
        right_renorms = right_renorms .^ -1

        lB_cur = fill_data!(similar(left_below.AL[1],
                                    left_virtualspace(left_below, 0) *
                                    _firstspace(hamrow[1])',
                                    exci_space' * right_virtualspace(right_above, 0)),
                            zero)
        rB_cur = fill_data!(similar(left_below.AL[1],
                                    left_virtualspace(left_below, 0) *
                                    _firstspace(hamrow[1]),
                                    exci_space' * right_virtualspace(right_above, 0)),
                            zero)
        for col in 1:numcols
            lB_cur = lB_cur *
                     TransferMatrix(right_above.AR[col], hamrow[col], left_below.AL[col])
            lB_cur += c_lenvs[col] *
                      TransferMatrix(exci[row][col], hamrow[col], left_below.AL[col])
            lB_cur *= left_renorms[col] * exp(-1im * exci.momentum)
            lBs[row, col] = lB_cur

            col = numcols - col + 1

            rB_cur = TransferMatrix(left_above.AL[col], hamrow[col], right_below.AR[col]) *
                     rB_cur
            rB_cur += TransferMatrix(exci[row][col], hamrow[col], right_below.AR[col]) *
                      c_renvs[col]
            rB_cur *= exp(1im * exci.momentum) * right_renorms[col]
            rBs[row, col] = rB_cur
        end

        tm_RL = TransferMatrix(right_above.AR, hamrow, left_below.AL)
        tm_LR = TransferMatrix(left_above.AL, hamrow, right_below.AR)

        if exci.trivial
            @plansor rvec[-1 -2; -3] := rightenv(lenvs, 0, left_gs)[row][-1 -2; 1] *
                                        conj(left_below.CR[0][-3; 1])
            @plansor lvec[-1 -2; -3] := leftenv(lenvs, 1, left_gs)[row][-1 -2; 1] *
                                        left_above.CR[0][1; -3]

            tm_RL = regularize(tm_RL, lvec, rvec)

            @plansor rvec[-1 -2; -3] := rightenv(renvs, 0, right_gs)[row][1 -2; -3] *
                                        right_above.CR[0][-1; 1]
            @plansor lvec[-1 -2; -3] := conj(right_below.CR[0][-3; 1]) *
                                        leftenv(renvs, 1, right_gs)[row][-1 -2; 1]

            tm_LR = regularize(tm_LR, lvec, rvec)
        end

        lBs[row, end], convhist = linsolve(flip(tm_RL), lB_cur, lB_cur, solver, 1,
                                           -exp(-1im * numcols * exci.momentum) *
                                           prod(left_renorms))
        convhist.converged == 0 &&
            @warn "GBL[$row] failed to converge: normres = $(convhist.normres)"

        rBs[row, 1], convhist = linsolve(tm_LR, rB_cur, rB_cur, GMRES(), 1,
                                         -exp(1im * numcols * exci.momentum) *
                                         prod(right_renorms))
        convhist.converged == 0 &&
            @warn "GBR[$row] failed to converge: normres = $(convhist.normres)"

        left_cur = lBs[row, end]
        right_cur = rBs[row, 1]
        for col in 1:(numcols - 1)
            left_cur = left_renorms[col] * left_cur *
                       TransferMatrix(right_above.AR[col], hamrow[col],
                                      left_below.AL[col]) * exp(-1im * exci.momentum)
            lBs[row, col] += left_cur

            col = numcols - col + 1
            right_cur = TransferMatrix(left_above.AL[col], hamrow[col],
                                       right_below.AR[col]) * right_cur *
                        exp(1im * exci.momentum) * right_renorms[col]
            rBs[row, col] += right_cur
        end
    end

    return QPEnv(lBs, rBs, lenvs, renvs)
end

function environments(exci::InfiniteQP,
                      O::InfiniteMPO,
                      lenvs,
                      renvs;
                      solver=Defaults.linearsolver)
    exci.trivial ||
        @warn "there is a phase ambiguity in topologically nontrivial statmech excitations"

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

    if exci.trivial
        @plansor rvec[-1 -2; -3] := rightenv(lenvs, 0, left_gs)[-1 -2; 1] *
                                    conj(left_gs.CR[0][-3; 1])
        @plansor lvec[-1 -2; -3] := leftenv(lenvs, 1, left_gs)[-1 -2; 1] *
                                    left_gs.CR[0][1; -3]

        T_RL = regularize(T_RL, lvec, rvec)

        @plansor rvec[-1 -2; -3] := rightenv(renvs, 0, right_gs)[1 -2; -3] *
                                    right_gs.CR[0][-1; 1]
        @plansor lvec[-1 -2; -3] := conj(right_gs.CR[0][-3; 1]) *
                                    leftenv(renvs, 1, right_gs)[-1 -2; 1]

        T_LR = regularize(T_LR, lvec, rvec)
    end

    GBL[end], convhist = linsolve(flip(T_RL), gbl, gbl, solver, 1,
                                  -cis(-length(exci) * exci.momentum) *
                                  prod(left_regularization))

    convhist.converged == 0 &&
        @warn "GBL failed to converge: normres = $(convhist.normres)"

    GBR[1], convhist = linsolve(T_LR, gbr, gbr, GMRES(), 1,
                                -cis(length(exci) * exci.momentum) *
                                prod(right_regularization))
    convhist.converged == 0 &&
        @warn "GBR failed to converge: normres = $(convhist.normres)"

    left_cur = GBL[end]
    right_cur = GBR[1]
    for col in 1:(length(exci) - 1)
        left_cur = left_regularization[col] * left_cur *
                   TransferMatrix(right_gs.AR[col], O[col],
                                  left_gs.AL[col]) * cis(-exci.momentum)
        GBL[col] += left_cur

        col = length(exci) - col + 1
        right_cur = TransferMatrix(left_gs.AL[col], O[col],
                                   right_gs.AR[col]) * right_cur *
                    cis(exci.momentum) * right_regularization[col]
        GBR[col] += right_cur
    end

    return QuasiparticleEnvironments(GBL, GBR, lenvs, renvs)
end
