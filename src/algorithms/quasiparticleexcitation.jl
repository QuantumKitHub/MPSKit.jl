#=
    an excitation tensor has 4 legs (1,2),(3,4)
    the first and the last are virtual, the second is physical, the third is the utility leg
=#

"
    quasiparticle_excitation calculates the energy of the first excited state at momentum 'moment'
"
function quasiparticle_excitation(hamiltonian::Hamiltonian, moment::Float64, mpsleft::MpsCenterGauged, paramsleft, mpsright::MpsCenterGauged=mpsleft, paramsright=paramsleft; excitation_space=oneunit(space(mpsleft.AL[1],1)), trivial=true, num=1 ,X_initial=nothing, toler = 1e-10,krylovdim=30)
    #check whether the provided mps is sensible
    @assert length(mpsleft) == length(mpsright)

    #find the left null spaces for the TNS
    LeftNullSpace = [adjoint(rightnull(adjoint(v))) for v in mpsleft.AL]

    #we need an initial array of Xs, one for every site in unit cell which the user may of may not have provided by the user
    if X_initial == nothing
        X_initial=[TensorMap(rand,eltype(mpsleft),space(LeftNullSpace[loc],3)',excitation_space'*space(mpsright.AR[ loc+1],1)) for loc in 1:length(mpsleft)]
    end

    #the function that maps x->B and then places this in the excitation hamiltonian
    function eigEx(x)
        x=[x.vecs...]

        B = [ln*cx for (ln,cx) in zip(LeftNullSpace,x)]

        Bseff = effective_excitation_hamiltonian(trivial, hamiltonian, B, moment, mpsleft, paramsleft, mpsright, paramsright)

        out = [adjoint(ln)*cB for (ln,cB) in zip(LeftNullSpace,Bseff)]

        return RecursiveVec(out...)
    end
    Es,Vs,convhist = eigsolve(eigEx, RecursiveVec(X_initial...), num, :SR, tol=toler,krylovdim=krylovdim)
    convhist.converged<num && @info "quasiparticle didn't converge k=$(moment) $(convhist.normres)"

    #we dont want to return a RecursiveVec to the user so upack it
    Xs=[[v.vecs...] for v in Vs]
    Bs=[[ln*cx for (ln,cx) in zip(LeftNullSpace,x)] for x in Xs]

    return Es,Bs, Xs
end

function quasiparticle_excitation(hamiltonian::Hamiltonian, momenta::AbstractVector, mpsleft::MpsCenterGauged, paramsleft, mpsright::MpsCenterGauged=mpsleft, paramsright=paramsleft; excitation_space=oneunit(space(mpsleft.AL[1],1)), trivial=true, num=1 ,X_initial=nothing,verbose=Defaults.verbose,krylovdim=30)
    Ep = Vector(undef,length(momenta))
    Bp = Vector(undef,length(momenta))
    Xp = Vector(undef,length(momenta))

    @threads for i in 1:length(momenta)
        verbose && println("Finding excitations for p = $(momenta[i])")
        Ep[i],Bp[i], Xp[i] = quasiparticle_excitation(hamiltonian, momenta[i], mpsleft, paramsleft, mpsright, paramsright, excitation_space=excitation_space, trivial=trivial, num=num ,X_initial= X_initial,krylovdim = krylovdim)
    end
    return Ep,Bp,Xp
end

function effective_excitation_hamiltonian(trivial::Bool, ham::MpoHamiltonian, Bs, p::Float64, mpsleft::MpsCenterGauged, paramsleft, mpsright::MpsCenterGauged, paramsright; renorm=true)
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
        lBs = transfer_left(lBs,ham,pos,mpsright.AR[pos],mpsleft.AL[pos])*exp(-1im*p)
        lBs += transfer_left(leftenv(paramsleft,pos,mpsleft),ham,pos,Bs[pos],mpsleft.AL[pos])*exp(-1im*p)

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
        rBs = transfer_right(rBs,ham,pos,mpsleft.AL[pos],mpsright.AR[pos])*exp(1im*p)
        rBs += transfer_right(rightenv(paramsright,pos,mpsright),ham,pos,Bs[pos],mpsright.AR[pos])*exp(1im*p)

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

        lBsE = transfer_left(lBsE,ham,i,mpsright.AR[i],mpsleft.AL[i])*exp(-1im*p)

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

        rBsE=transfer_right(rBsE,ham,i,mpsleft.AL[i],mpsright.AR[i])*exp(1im*p)

        if trivial
            for k in ids[2:end-1]
                @tensor rBsE[k][-1,-2,-3,-4]-=rBsE[k][1,-2,-3,2]*l_LR(mpsleft,i)[2,1]*r_LR(mpsleft,i)[-1,-4]
            end
        end
    end

    return toret
end

#=
original code :
(lBsEr,convhist)=linsolve(RecursiveVec(lBs...),RecursiveVec(lBs...),GMRES()) do y
    x=collect(y.vecs)
    tor=reduce((a,b)->transfer_left(a,ham,b,mpsright.AR[b],mpsleft.AL[b])*exp(-1im*p),1:len,init=x)

    if(trivial)
        for i in ids
            @tensor tor[i][-1,-2,-3,-4]-=tor[i][1,-2,2,-4]*r_RL(mpsleft)[2,1]*l_RL(mpsleft)[-1,-3]
        end
    end

    tor=x-tor
    return  RecursiveVec(tor...)
end
lBsE=lBsEr.vecs

but this can be made faster; using the fact that the hamiltonion is upper-triangular, which is what we do here
=#

function left_excitation_transfer_system(lBs,ham,mpsleft::MpsCenterGauged,mpsright::MpsCenterGauged,trivial,ids,p)
    len = ham.period
    found=zero.(lBs)

    for i in 1:ham.odim


        #this operation can be sped up by at least a factor 2;  found mostly consists of zeros
        start = found
        for k in 1:len
            start = transfer_left(start,ham,k,mpsright.AR[k],mpsleft.AL[k])*exp(-1im*p)

            if trivial
                for l in ids[2:end-1]
                    @tensor start[l][-1,-2,-3,-4]-=start[l][1,-2,-3,2]*r_RL(mpsright,k)[2,1]*l_RL(mpsright,k)[-1,-4]
                end
            end
        end

        #either the element ham_ii exists; in which case we have to solve a linear system
        #otherwise it's easy and we already know found[i]
        if reduce((a,b)->contains(ham,b,i,i),1:len,init=true)
            (found[i],convhist)=linsolve(lBs[i]+start[i],lBs[i]+start[i],GMRES()) do y
                x=reduce((a,b)->transfer_left(a,ham[b,i,i],mpsright.AR[b],mpsleft.AL[b])*exp(-1im*p),1:len,init=y)

                if trivial && i in ids
                    @tensor x[-1,-2,-3,-4]-=x[1,-2,-3,2]*r_RL(mpsleft)[2,1]*l_RL(mpsleft)[-1,-4]
                end

                return y-x
            end
            if convhist.converged<1
                @info "left $(i) excitation inversion failed normres $(convhist.normres)"
            end
        else
            found[i]=lBs[i]+start[i]
        end
    end
    return found
end

function right_excitation_transfer_system(rBs,ham,mpsleft,mpsright::MpsCenterGauged,trivial,ids,p)
    len = ham.period
    found=zero.(rBs)

    for i in ham.odim:-1:1

        start = found
        for k in len:-1:1
            start = transfer_right(start,ham,k,mpsleft.AL[k],mpsright.AR[k])*exp(1im*p)

            if trivial
                for l in ids[2:end-1]
                    @tensor start[l][-1,-2,-3,-4]-=start[l][1,-2,-3,2]*l_LR(mpsright,k)[2,1]*r_LR(mpsright,k)[-1,-4]
                end
            end
        end

        if reduce((a,b)->contains(ham,b,i,i),1:len,init=true)
            (found[i],convhist)=linsolve(rBs[i]+start[i],rBs[i]+start[i],GMRES()) do y
                x=reduce((a,b)->transfer_right(a,ham[b,i,i],mpsleft.AL[b],mpsright.AR[b])*exp(1im*p),len:-1:1,init=y)

                if trivial && i in ids
                    @tensor x[-1,-2,-3,-4]-=x[1,-2,-3,2]*l_LR(mpsright)[2,1]*r_LR(mpsright)[-1,-4]
                end

                return y-x
            end
            if convhist.converged<1
                @info "right $(i) excitation inversion failed normres $(convhist.normres)"
            end
        else
            found[i]=rBs[i]+start[i]
        end
    end
    return found
end
