function effective_excitation_hamiltonian(ham::MPOHamiltonian, Bs, leftmps::FiniteMPS{A}, leftpars,rightmps::FiniteMPS{A}=leftmps,rightpars=leftpars) where A
    odim = ham.odim;
    toret = zero.(Bs)

    #construct lBsE
    lBs = [ TensorMap(zeros,eltype(A),
                    space(leftenv(leftpars,1,leftmps)[k],1)*space(leftenv(leftpars,1,leftmps)[k],2),
                    space(Bs[1],3)'*space(leftenv(leftpars,1,leftmps)[k],3)') for k in 1:ham.odim]
    lBsc = typeof(lBs)[]
    for pos = 1:length(leftmps)
        lBs = exci_transfer_left(lBs,ham,pos,rightmps.AR[pos],leftmps.AL[pos])
        lBs += exci_transfer_left(leftenv(leftpars,pos,leftmps),ham,pos,Bs[pos],leftmps.AL[pos])
        push!(lBsc,lBs)
    end

    #build rBs(c)
    rBs = [ TensorMap(zeros,eltype(A),
                    space(rightenv(leftpars,length(leftmps),leftmps)[k],1)*space(Bs[1],3),
                    space(rightenv(leftpars,length(leftmps),leftmps)[k],2)'
                    *space(rightenv(leftpars,length(leftmps),leftmps)[k],3)') for k in 1:ham.odim]
    rBsc = typeof(rBs)[]
    for pos=length(leftmps):-1:1
        rBs = exci_transfer_right(rBs,ham,pos,leftmps.AL[pos],rightmps.AR[pos])
        rBs += exci_transfer_right(rightenv(rightpars,pos,rightmps),ham,pos,Bs[pos],rightmps.AR[pos])
        push!(rBsc,rBs)
    end
    rBsc=reverse(rBsc)

    #do necessary contractions
    for i = 1:length(leftmps)
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(leftpars,i,leftmps)[j][-1,1,2]*
                                                Bs[i][2,3,-3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(rightpars,i,rightmps)[k][4,5,-4]

            # <B|H|B>-<H>
            en = @tensor    conj(leftmps.AC[i][11,12,13])*
                            leftenv(leftpars,i,leftmps)[j][11,1,2]*
                            leftmps.AC[i][2,3,4]*
                            ham[i,j,k][1,12,5,3]*
                            rightenv(leftpars,i,leftmps)[k][4,5,13]
            
            toret[i] -= Bs[i]*en
            if i>1
                @tensor toret[i][-1,-2,-3,-4] +=    lBsc[i-1][j][-1,1,-3,2]*
                                                    rightmps.AR[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rightenv(rightpars,i,rightmps)[k][4,5,-4]
            end
            if i<length(leftmps)
                @tensor toret[i][-1,-2,-3,-4] +=    leftenv(leftpars,i,leftmps)[j][-1,1,2]*
                                                    leftmps.AL[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rBsc[i+1][k][4,-3,5,-4]
            end
        end

    end

    return toret

end
