
function effective_excitation_hamiltonian(ham::MPOHamiltonian, exci::InfiniteQP,pars=params(exci,ham))
    #does not "know" that B is left gauged, so it's possible to speed this up further
    Bs = [exci[i] for i in 1:length(exci)];
    toret = zero.(Bs);

    ids = collect(Iterators.filter(x->isid(ham,x),2:ham.odim-1));

    # B in same unit cell as B'
    # this is the only point where we have to take renorm into account (a constant shift in the hamiltonian will only affect the terms where both Bs are at the same position)
    for i = 1:length(exci)
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(pars.lpars,i,exci.left_gs)[j][-1,1,2]*
                                                Bs[i][2,3,-3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(pars.rpars,i,exci.right_gs)[k][4,5,-4]

            # <B|H|B>-<H>
            en = @tensor    conj(exci.left_gs.AC[i][11,12,13])*
                            leftenv(pars.lpars,i,exci.left_gs)[j][11,1,2]*
                            exci.left_gs.AC[i][2,3,4]*
                            ham[i,j,k][1,12,5,3]*
                            rightenv(pars.lpars,i,exci.left_gs)[k][4,5,13]
            toret[i] -= Bs[i]*en

            if i>1
                @tensor toret[i][-1,-2,-3,-4] +=    pars.lBsc[i-1][j][-1,1,-3,2]*
                                                    exci.right_gs.AR[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rightenv(pars.rpars,i,exci.right_gs)[k][4,5,-4]
            end
            if i<length(exci)
                @tensor toret[i][-1,-2,-3,-4] +=    leftenv(pars.lpars,i,exci.left_gs)[j][-1,1,2]*
                                                    exci.left_gs.AL[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    pars.rBsc[i+1][k][4,-3,5,-4]
            end
        end

    end

    #B left to B'; outside the unit cell
    lBsE = pars.lBsE
    for i=1:length(exci)
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    lBsE[j][-1,1,-3,2]*
                                                exci.right_gs.AR[i][2,3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(pars.rpars,i,exci.right_gs)[k][4,5,-4]
        end

        lBsE = exci_transfer_left(lBsE,ham,i,exci.right_gs.AR[i],exci.left_gs.AL[i])*exp(conj(1im*exci.momentum))

        exci.trivial && for k in ids
            @tensor lBsE[k][-1,-2,-3,-4] -= lBsE[k][1,-2,-3,2]*r_RL(exci.left_gs,i)[2,1]*l_RL(exci.left_gs,i)[-1,-4]
        end

    end

    #B right to B'; outside the unit cell
    rBsE = pars.rBsE
    for i=length(exci):-1:1
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(pars.lpars,i,exci.left_gs)[j][-1,1,2]*
                                                exci.left_gs.AL[i][2,3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rBsE[k][4,-3,5,-4]
        end

        rBsE = exci_transfer_right(rBsE,ham,i,exci.left_gs.AL[i],exci.right_gs.AR[i])*exp(1im*exci.momentum)

        exci.trivial && for k in ids
            @tensor rBsE[k][-1,-2,-3,-4]-=rBsE[k][1,-2,-3,2]*l_LR(exci.left_gs,i)[2,1]*r_LR(exci.left_gs,i)[-1,-4]
        end
    end

    toret_vec = similar(exci);
    for i in 1:length(exci)
        toret_vec[i] = toret[i]
    end
    return toret_vec
end
