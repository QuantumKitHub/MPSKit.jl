function effective_excitation_hamiltonian(ham::MPOHamiltonian, exci::FiniteQP,leftpars,rightpars=leftpars)
    odim = ham.odim;

    Bs = [exci[i] for i in 1:length(exci)];
    toret = zero.(Bs);

    #construct lBsE
    lBs = [ TensorMap(zeros,eltype(exci),
                    space(leftenv(leftpars,1,exci.left_gs)[k],1)*space(leftenv(leftpars,1,exci.left_gs)[k],2),
                    space(Bs[1],3)'*space(leftenv(leftpars,1,exci.left_gs)[k],3)') for k in 1:ham.odim]
    lBsc = typeof(lBs)[]
    for pos = 1:length(exci)
        lBs = exci_transfer_left(lBs,ham,pos,exci.right_gs.AR[pos],exci.left_gs.AL[pos])
        lBs += exci_transfer_left(leftenv(leftpars,pos,exci.left_gs),ham,pos,Bs[pos],exci.left_gs.AL[pos])
        push!(lBsc,lBs)
    end

    #build rBs(c)
    rBs = [ TensorMap(zeros,eltype(exci),
                    space(rightenv(leftpars,length(exci.left_gs),exci.left_gs)[k],1)*space(Bs[1],3),
                    space(rightenv(leftpars,length(exci.left_gs),exci.left_gs)[k],2)'
                    *space(rightenv(leftpars,length(exci.left_gs),exci.left_gs)[k],3)') for k in 1:ham.odim]
    rBsc = typeof(rBs)[]
    for pos=length(exci):-1:1
        rBs = exci_transfer_right(rBs,ham,pos,exci.left_gs.AL[pos],exci.right_gs.AR[pos])
        rBs += exci_transfer_right(rightenv(rightpars,pos,exci.right_gs),ham,pos,Bs[pos],exci.right_gs.AR[pos])
        push!(rBsc,rBs)
    end
    rBsc=reverse(rBsc)

    #do necessary contractions
    for i = 1:length(exci)
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(leftpars,i,exci.left_gs)[j][-1,1,2]*
                                                Bs[i][2,3,-3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(rightpars,i,exci.right_gs)[k][4,5,-4]

            # <B|H|B>-<H>
            en = @tensor    conj(exci.left_gs.AC[i][11,12,13])*
                            leftenv(leftpars,i,exci.left_gs)[j][11,1,2]*
                            exci.left_gs.AC[i][2,3,4]*
                            ham[i,j,k][1,12,5,3]*
                            rightenv(leftpars,i,exci.left_gs)[k][4,5,13]

            toret[i] -= Bs[i]*en
            if i>1
                @tensor toret[i][-1,-2,-3,-4] +=    lBsc[i-1][j][-1,1,-3,2]*
                                                    exci.right_gs.AR[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rightenv(rightpars,i,exci.right_gs)[k][4,5,-4]
            end
            if i<length(exci.left_gs)
                @tensor toret[i][-1,-2,-3,-4] +=    leftenv(leftpars,i,exci.left_gs)[j][-1,1,2]*
                                                    exci.left_gs.AL[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rBsc[i+1][k][4,-3,5,-4]
            end
        end

    end

    toret_vec = similar(exci);
    for i in 1:length(exci)
        toret_vec[i] = toret[i]
    end
    return toret_vec

end
