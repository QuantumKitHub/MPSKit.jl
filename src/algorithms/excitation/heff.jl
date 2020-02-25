@bm function effective_excitation_hamiltonian(trivial::Bool, ham::MPOHamiltonian, Bs, p::Float64, mpsleft::InfiniteMPS, paramsleft, mpsright::InfiniteMPS, paramsright; renorm=true)
    #does not "know" that B is left gauged, so it's possible to speed this up further
    toret = zero.(Bs)
    len = length(mpsleft);

    #precached version if isid
    pisid = [isid(ham,i) for i in 1:ham.odim]
    ids = collect(map(x->x[1],Iterators.filter(x->x[2],enumerate(pisid))))

    #validity checks
    @assert pisid[1] && pisid[end]
    @assert length(Bs) == len
    @assert length(mpsright) == len
    @assert ham.period == len
    @assert eltype(mpsleft)==eltype(mpsright)

    #build lBs(c)
    lBs = [ TensorMap(zeros,eltype(mpsleft),
                    space(paramsleft.lw[1,k],1)*space(paramsleft.lw[1,k],2),
                    space(Bs[1],3)'*space(paramsright.lw[1,k],3)') for k in 1:ham.odim]
    lBsc = []

    for pos = 1:len
        lBs = exci_transfer_left(lBs,ham,pos,mpsright.AR[pos],mpsleft.AL[pos])*exp(-1im*p)
        lBs += exci_transfer_left(leftenv(paramsleft,pos,mpsleft),ham,pos,Bs[pos],mpsleft.AL[pos])*exp(-1im*p)

        if trivial
            for i in ids[2:end-1]
                @tensor lBs[i][-1,-2,-3,-4] -= lBs[i][1,-2,-3,2]*r_RL(mpsleft,pos)[2,1]*l_RL(mpsleft,pos)[-1,-4]
            end
        end

        push!(lBsc,lBs)
    end

    #build rBs(c)
    rBs = [ TensorMap(zeros,eltype(mpsleft),
                    space(paramsleft.rw[end,k],1)*space(Bs[1],3),
                    space(paramsright.rw[end,k],2)'*space(paramsright.rw[end,k],3)') for k in 1:ham.odim]
    rBsc = []

    for pos=len:-1:1
        rBs = exci_transfer_right(rBs,ham,pos,mpsleft.AL[pos],mpsright.AR[pos])*exp(1im*p)
        rBs += exci_transfer_right(rightenv(paramsright,pos,mpsright),ham,pos,Bs[pos],mpsright.AR[pos])*exp(1im*p)

        if trivial
            for i in ids[2:end-1]
                @tensor rBs[i][-1,-2,-3,-4] -= rBs[i][1,-2,-3,2]*l_LR(mpsleft,pos)[2,1]*r_LR(mpsleft,pos)[-1,-4]
            end
        end

        push!(rBsc,rBs)
    end
    rBsc=reverse(rBsc)



    # B in same unit cell as B'
    # this is the only point where we have to take renorm into account (a constant shift in the hamiltonian will only affect the terms where both Bs are at the same position)
    for i = 1:len
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(paramsleft,i,mpsleft)[j][-1,1,2]*
                                                Bs[i][2,3,-3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(paramsright,i,mpsright)[k][4,5,-4]

            if (renorm) # <B|H|B>-<H>
                en = @tensor    conj(mpsleft.AC[i][11,12,13])*
                                leftenv(paramsleft,i,mpsleft)[j][11,1,2]*
                                mpsleft.AC[i][2,3,4]*
                                ham[i,j,k][1,12,5,3]*
                                rightenv(paramsleft,i,mpsleft)[k][4,5,13]
                toret[i] -= Bs[i]*en
            end
            if i>1
                @tensor toret[i][-1,-2,-3,-4] +=    lBsc[i-1][j][-1,1,-3,2]*
                                                    mpsright.AR[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rightenv(paramsright,i,mpsright)[k][4,5,-4]
            end
            if i<len
                @tensor toret[i][-1,-2,-3,-4] +=    leftenv(paramsleft,i,mpsleft)[j][-1,1,2]*
                                                    mpsleft.AL[i][2,3,4]*
                                                    ham[i,j,k][1,-2,5,3]*
                                                    rBsc[i+1][k][4,-3,5,-4]
            end
        end

    end

    #B left to B'; outside the unit cell
    lBsE = left_excitation_transfer_system(lBs,ham,mpsleft,mpsright,trivial,ids,p)

    for i=1:len
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    lBsE[j][-1,1,-3,2]*
                                                mpsright.AR[i][2,3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rightenv(paramsright,i,mpsright)[k][4,5,-4]
        end

        lBsE = exci_transfer_left(lBsE,ham,i,mpsright.AR[i],mpsleft.AL[i])*exp(-1im*p)

        if trivial
            for k in ids[2:end-1]
                @tensor lBsE[k][-1,-2,-3,-4] -= lBsE[k][1,-2,-3,2]*r_RL(mpsleft,i)[2,1]*l_RL(mpsleft,i)[-1,-4]
            end
        end
    end

    #B right to B'; outside the unit cell
    rBsE = right_excitation_transfer_system(rBs,ham,mpsleft,mpsright,trivial,ids,p)

    for i=len:-1:1
        for (j,k) in keys(ham,i)
            @tensor toret[i][-1,-2,-3,-4] +=    leftenv(paramsleft,i,mpsleft)[j][-1,1,2]*
                                                mpsleft.AL[i][2,3,4]*
                                                ham[i,j,k][1,-2,5,3]*
                                                rBsE[k][4,-3,5,-4]
        end

        rBsE=exci_transfer_right(rBsE,ham,i,mpsleft.AL[i],mpsright.AR[i])*exp(1im*p)

        if trivial
            for k in ids[2:end-1]
                @tensor rBsE[k][-1,-2,-3,-4]-=rBsE[k][1,-2,-3,2]*l_LR(mpsleft,i)[2,1]*r_LR(mpsleft,i)[-1,-4]
            end
        end
    end

    return toret
end
