function effective_excitation_hamiltonian(ham::MPOHamiltonian, exci::FiniteQP,pars=params(exci,ham))
    odim = ham.odim;

    Bs = [exci[i] for i in 1:length(exci)];
    toret = zero.(Bs);

    #do necessary contractions
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
            if i<length(exci.left_gs)
                @tensor toret[i][-1,-2,-3,-4] +=    leftenv(pars.lpars,i,exci.left_gs)[j][-1,1,2]*
                                                    exci.left_gs.AL[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    pars.rBsc[i+1][k][4,-3,5,-4]
            end
        end

    end

    toret_vec = similar(exci);
    for i in 1:length(exci)
        toret_vec[i] = toret[i]
    end
    return toret_vec

end
