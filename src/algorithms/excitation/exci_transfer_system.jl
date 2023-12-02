function left_excitation_transfer_system(lBs, H, exci; mom=exci.momentum,
                                         solver=Defaults.linearsolver)
    len = H.period
    found = zero.(lBs)
    odim = length(lBs)

    for i in 1:odim
        #this operation can be sped up by at least a factor 2;  found mostly consists of zeros
        start = found * TransferMatrix(exci.right_gs.AR, H[:], exci.left_gs.AL) /
                exp(1im * mom * len)
        if exci.trivial && isid(H, i)
            @plansor start[i][-1 -2; -3 -4] -= start[i][1 4; -3 2] *
                                               r_RL(exci.right_gs)[2; 3] *
                                               τ[3 4; 5 1] *
                                               l_RL(exci.right_gs)[-1; 6] *
                                               τ[5 6; -4 -2]
        end

        found[i] = lBs[i] + start[i]

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

            (found[i], convhist) = linsolve(flip(tm), found[i], found[i], solver, 1,
                                            -1 / exp(1im * mom * len))
            convhist.converged < 1 &&
                @info "left $(i) excitation inversion failed normres $(convhist.normres)"
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
        start = TransferMatrix(exci.left_gs.AL, H[:], exci.right_gs.AR) *
                found *
                exp(1im * mom * len)

        if exci.trivial && isid(H, i)
            @plansor start[i][-1 -2; -3 -4] -= τ[6 2; 3 4] * start[i][3 4; -3 5] *
                                               l_LR(exci.right_gs)[5; 2] *
                                               r_LR(exci.right_gs)[-1; 1] * τ[-2 -4; 1 6]
        end

        found[i] = rBs[i] + start[i]

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

            (found[i], convhist) = linsolve(tm, found[i], found[i], solver, 1,
                                            -exp(1im * mom * len))
            convhist.converged < 1 &&
                @info "right $(i) excitation inversion failed normres $(convhist.normres)"
        end
    end
    return found
end
