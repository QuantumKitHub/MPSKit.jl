function left_excitation_transfer_system(lBs, H, exci; mom=exci.momentum,
                                         solver=Defaults.linearsolver)
    len = H.period
    found = zero.(lBs)
    odim = length(lBs)

    for i in 1:odim
        # this operation can in principle be even further optimized for larger unit cells
        # as we only require the terms that end at level i.
        # this would require to check the finite state machine, and discard non-connected
        # terms.
        H_partial = map(site -> H.data[site, 1:i, 1:i], 1:len)
        T = TransferMatrix(exci.right_gs.AR, H_partial, exci.left_gs.AL)
        start = scale!(last(found[1:i] * T), cis(-mom * len))
        if exci.trivial && isid(H, i)
            @plansor start[-1 -2; -3 -4] -= start[1 4; -3 2] *
                                            r_RL(exci.right_gs)[2; 3] *
                                            τ[3 4; 5 1] *
                                            l_RL(exci.right_gs)[-1; 6] *
                                            τ[5 6; -4 -2]
        end

        found[i] = add!(start, lBs[i])

        if reduce(&, contains.(H.data, i, i))
            if isid(H, i)
                tm = TransferMatrix(exci.right_gs.AR, exci.left_gs.AL)
                if exci.trivial
                    tm = regularize(tm, l_RL(exci.right_gs), r_RL(exci.right_gs))
                end
            else
                tm = TransferMatrix(exci.right_gs.AR, getindex.(H.data, i, i),
                                    exci.left_gs.AL)
            end

            found[i], convhist = linsolve(flip(tm), found[i], found[i], solver, 1,
                                          -cis(-mom * len))
            convhist.converged == 0 &&
                @warn "GBL$i failed to converge: normres = $(convhist.normres)"
        end
    end
    return found
end

function right_excitation_transfer_system(rBs, H, exci; mom=exci.momentum,
                                          solver=Defaults.linearsolver)
    len = H.period
    found = zero.(rBs)
    odim = length(rBs)

    for i in odim:-1:1
        # this operation can in principle be even further optimized for larger unit cells
        # as we only require the terms that end at level i.
        # this would require to check the finite state machine, and discard non-connected
        # terms.
        H_partial = map(site -> H.data[site, i:odim, i:odim], 1:len)
        T = TransferMatrix(exci.left_gs.AL, H_partial, exci.right_gs.AR)
        start = scale!(first(T * found[i:odim]), cis(mom * len))
        if exci.trivial && isid(H, i)
            @plansor start[-1 -2; -3 -4] -= τ[6 2; 3 4] * start[3 4; -3 5] *
                                            l_LR(exci.right_gs)[5; 2] *
                                            r_LR(exci.right_gs)[-1; 1] * τ[-2 -4; 1 6]
        end

        found[i] = add!(start, rBs[i])

        if reduce(&, contains.(H.data, i, i))
            if isid(H, i)
                tm = TransferMatrix(exci.left_gs.AL, exci.right_gs.AR)
                if exci.trivial
                    tm = regularize(tm, l_LR(exci.left_gs), r_LR(exci.right_gs))
                end
            else
                tm = TransferMatrix(exci.left_gs.AL, getindex.(H.data, i, i),
                                    exci.right_gs.AR)
            end

            found[i], convhist = linsolve(tm, found[i], found[i], solver, 1,
                                          -cis(mom * len))
            convhist.converged < 1 &&
                @warn "GBR$i failed to converge: normres = $(convhist.normres)"
        end
    end
    return found
end
