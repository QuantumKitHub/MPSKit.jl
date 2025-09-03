function left_excitation_transfer_system(
        GBL, H::InfiniteMPOHamiltonian, exci;
        mom = exci.momentum, solver = Defaults.linearsolver
    )
    len = length(H)
    found = zerovector(GBL)
    odim = length(GBL)

    if istrivial(exci)
        ρ_left = l_RL(exci.right_gs)
        ρ_right = r_RL(exci.right_gs)
    end

    for i in 1:odim
        # this operation can in principle be even further optimized for larger unit cells
        # as we only require the terms that end at level i.
        # this would require to check the finite state machine, and discard non-connected
        # terms.
        H_partial = map(h -> getindex(h, 1:i, 1, 1, 1:i), parent(H))
        T = TransferMatrix(exci.right_gs.AR, H_partial, exci.left_gs.AL)
        start = scale!(last(found[1:i] * T), cis(-mom * len))
        if istrivial(exci) && isidentitylevel(H, i)
            regularize!(start, ρ_right, ρ_left)
        if exci.trivial && isidentitylevel(H, i)
            # not using braiding tensors here, leads to extra leg
            util = similar(exci.left_gs.AL[i], first(left_virtualspace(H[i])))
            fill_data!(util, one)
            @plansor start[-1 -2; -3 -4] -= start[2 1; -3 3] *
                util[1] *
                r_RL(exci.right_gs)[3; 2] *
                l_RL(exci.right_gs)[-1; -4] *
                conj(util[-2])
        end

        found[i] = add!(start, GBL[i])

        if !isemptylevel(H, i)
            if isidentitylevel(H, i)
                T = TransferMatrix(exci.right_gs.AR, exci.left_gs.AL)
                if istrivial(exci)
                    T = regularize(T, ρ_left, ρ_right)
                end
            else
                T = TransferMatrix(
                    exci.right_gs.AR, map(h -> h[i, 1, 1, i], parent(H)), exci.left_gs.AL
                )
            end

            found[i], convhist = linsolve(
                flip(T), found[i], found[i], solver, 1, -cis(-mom * len)
            )
            convhist.converged == 0 &&
                @warn "GBL$i failed to converge: normres = $(convhist.normres)"
        end
    end
    return found
end

function right_excitation_transfer_system(
        GBR, H::InfiniteMPOHamiltonian, exci;
        mom = exci.momentum,
        solver = Defaults.linearsolver
    )
    len = length(H)
    found = zerovector(GBR)
    odim = length(GBR)

    if istrivial(exci)
        ρ_left = l_LR(exci.right_gs)
        ρ_right = r_LR(exci.right_gs)
    end

    for i in odim:-1:1
        # this operation can in principle be even further optimized for larger unit cells
        # as we only require the terms that end at level i.
        # this would require to check the finite state machine, and discard non-connected
        # terms.
        H_partial = map(h -> h[i:end, 1, 1, i:end], parent(H))
        T = TransferMatrix(exci.left_gs.AL, H_partial, exci.right_gs.AR)
        start = scale!(first(T * found[i:odim]), cis(mom * len))
        if istrivial(exci) && isidentitylevel(H, i)
            regularize!(start, ρ_left, ρ_right)
        if exci.trivial && isidentitylevel(H, i)
            # not using braiding tensors here, leads to extra leg
            util = similar(exci.right_gs.AL[i], first(left_virtualspace(H[i])))
            fill_data!(util, one)
            @plansor start[-1 -2; -3 -4] -= start[2 1; -3 3] *
                conj(util[1]) *
                l_LR(exci.right_gs)[3; 2] *
                r_LR(exci.right_gs)[-1; -4] *
                util[-2]
        end

        found[i] = add!(start, GBR[i])

        if !isemptylevel(H, i)
            if isidentitylevel(H, i)
                tm = TransferMatrix(exci.left_gs.AL, exci.right_gs.AR)
                if istrivial(exci)
                    tm = regularize(tm, ρ_left, ρ_right)
                end
            else
                tm = TransferMatrix(
                    exci.left_gs.AL, map(h -> h[i, 1, 1, i], parent(H)), exci.right_gs.AR
                )
            end

            found[i], convhist = linsolve(
                tm, found[i], found[i], solver, 1, -cis(mom * len)
            )
            convhist.converged < 1 &&
                @warn "GBR$i failed to converge: normres = $(convhist.normres)"
        end
    end
    return found
end
