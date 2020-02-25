#excitation transfers - we default to regular transfers when no better candidate is found
exci_transfer_left(v,A,B=A) = transfer_left(v,A,B)
exci_transfer_right(v,A,B=A) = transfer_right(v,A,B)
exci_transfer_left(v,A,B,C) = transfer_left(v,A,B,C)
exci_transfer_right(v,A,B,C) = transfer_right(v,A,B,C)
exci_transfer_left(v,A,B,C,D) = transfer_left(v,A,B,C,D)
exci_transfer_right(v,A,B,C,D) = transfer_right(v,A,B,C,D)

#transfer, but the upper A is an excited tensor
exci_transfer_left(v::MPSVecType, A::MPOType, Ab::MPSType) = @tensor t[-1 -2;-3] := v[1,2]*A[2,3,-2,-3]*conj(Ab[1,3,-1])
exci_transfer_right(v::MPSVecType, A::MPOType, Ab::MPSType) = @tensor t[-1 -2;-3] := A[-1,3,-2,1]*v[1,2]*conj(Ab[-3,3,2])

#transfer, but there is both a utility leg and an mpo leg that is passed through
exci_transfer_left(v::MPOType, A::MPSType, Ab::MPSType=A) = @tensor v[-1 -2;-3 -4] := v[1,-2,-3,2]*A[2,3,-4]*conj(Ab[1,3,-1])
exci_transfer_right(v::MPOType, A::MPSType, Ab::MPSType=A) = @tensor v[-1 -2;-3 -4] := A[-1,3,1]*v[1,-2,-3,2]*conj(Ab[-4,3,2])

#transfer, but the upper A is an excited tensor and there is an mpo leg being passed through
exci_transfer_left(v::MPSType, A::MPOType, Ab::MPSType) = @tensor t[-1 -2;-3 -4] := v[1,-2,2]*A[2,3,-3,-4]*conj(Ab[1,3,-1])
exci_transfer_right(v::MPSType, A::MPOType, Ab::MPSType) = @tensor t[-1 -2;-3 -4] := A[-1,3,-2,1]*v[1,-3,2]*conj(Ab[-4,3,2])

#mpo transfer, but with A an excitation-tensor
exci_transfer_left(v::MPSType,O::MPOType,A::MPOType,Ab::MPSType) = @tensor t[-1 -2;-3 -4] := v[4,5,1]*A[1,3,-3,-4]*O[5,2,-2,3]*conj(Ab[4,2,-1])
exci_transfer_right(v::MPSType,O::MPOType,A::MPOType,Ab::MPSType) = @tensor t[-1 -2;-3 -4] := A[-1,1,-2,5]*O[-3,3,4,1]*conj(Ab[-4,3,2])*v[5,4,2]

#mpo transfer, with an excitation leg
exci_transfer_left(v::MPOType,O::MPOType,A::MPSType,Ab::MPSType=A) = @tensor v[-1 -2;-3 -4] := v[4,5,-3,1]*A[1,3,-4]*O[5,2,-2,3]*conj(Ab[4,2,-1])
exci_transfer_right(v::MPOType,O::MPOType,A::MPSType,Ab::MPSType=A) = @tensor v[-1 -2;-3 -4] := A[-1,1,5]*O[-3,3,4,1]*conj(Ab[-4,3,2])*v[5,-2,4,2]

#A is an excitation tensor; with an excitation leg
exci_transfer_left(vec::Array{V,1},ham::MPOHamiltonian,pos::Int,A::M,Ab::V=A) where V<:MPSType where M <:MPOType = exci_transfer_left(M,vec,ham,pos,A,Ab)
exci_transfer_right(vec::Array{V,1},ham::MPOHamiltonian,pos::Int,A::M,Ab::V=A) where V<:MPSType where M <:MPOType = exci_transfer_right(M,vec,ham,pos,A,Ab)

#v has an extra excitation leg
exci_transfer_left(vec::Array{V,1},ham::MPOHamiltonian,pos::Int,A::M,Ab::M=A) where V<:MPOType where M <:MPSType = exci_transfer_left(V,vec,ham,pos,A,Ab)
exci_transfer_right(vec::Array{V,1},ham::MPOHamiltonian,pos::Int,A::M,Ab::M=A) where V<:MPOType where M <:MPSType = exci_transfer_right(V,vec,ham,pos,A,Ab)

function exci_transfer_left(RetType,vec,ham::MPOHamiltonian,pos,A,Ab=A)
    toreturn = Array{RetType,1}(undef,length(vec));
    assigned = [false for i in 1:ham.odim]

    for (j,k) in keys(ham,pos)
        if assigned[k]
            if j==k && isscal(ham,pos,j)
                toreturn[k]+=ham.scalars[pos][j]*exci_transfer_left(vec[j],A,Ab)
            else
                toreturn[k]+=exci_transfer_left(vec[j],ham[pos,j,k],A,Ab)
            end
        else
            if j==k && isscal(ham,pos,j)
                toreturn[k]=ham.scalars[pos][j]*exci_transfer_left(vec[j],A,Ab)
            else
                toreturn[k]=exci_transfer_left(vec[j],ham[pos,j,k],A,Ab)
            end
            assigned[k]=true
        end
    end


    for k in 1:ham.odim
        if !assigned[k]
            #prefereably this never happens, because it's a wasted step
            #it's also avoideable with a little bit more code
            toreturn[k]=exci_transfer_left(vec[1],ham[pos,1,k],A,Ab)
        end
    end

    return toreturn
end
function exci_transfer_right(RetType,vec,ham::MPOHamiltonian,pos,A,Ab=A)
    toreturn = Array{RetType,1}(undef,length(vec));
    assigned = [false for i in 1:ham.odim]

    for (j,k) in keys(ham,pos)
        if assigned[j]
            if j==k && isscal(ham,pos,j)
                toreturn[j]+=ham.scalars[pos][j]*exci_transfer_right(vec[k],A,Ab)
            else
                toreturn[j]+=exci_transfer_right(vec[k],ham[pos,j,k],A,Ab)
            end

        else
            if j==k && isscal(ham,pos,j)
                toreturn[j]=ham.scalars[pos][j]*exci_transfer_right(vec[k],A,Ab)
            else
                toreturn[j]=exci_transfer_right(vec[k],ham[pos,j,k],A,Ab)
            end
            assigned[j]=true
        end
    end

    for j in 1:ham.odim
        if !assigned[j]
            toreturn[j]=exci_transfer_right(vec[1],ham[pos,j,1],A,Ab)
        end
    end

    return toreturn
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

@bm function left_excitation_transfer_system(lBs,ham,mpsleft::InfiniteMPS,mpsright::InfiniteMPS,trivial,ids,p)
    len = ham.period
    found=zero.(lBs)

    for i in 1:ham.odim


        #this operation can be sped up by at least a factor 2;  found mostly consists of zeros
        start = found
        for k in 1:len
            start = exci_transfer_left(start,ham,k,mpsright.AR[k],mpsleft.AL[k])*exp(-1im*p)

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
                x=reduce((a,b)->exci_transfer_left(a,ham[b,i,i],mpsright.AR[b],mpsleft.AL[b])*exp(-1im*p),1:len,init=y)

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

@bm function right_excitation_transfer_system(rBs,ham,mpsleft,mpsright::InfiniteMPS,trivial,ids,p)
    len = ham.period
    found=zero.(rBs)

    for i in ham.odim:-1:1

        start = found
        for k in len:-1:1
            start = exci_transfer_right(start,ham,k,mpsleft.AL[k],mpsright.AR[k])*exp(1im*p)

            if trivial
                for l in ids[2:end-1]
                    @tensor start[l][-1,-2,-3,-4]-=start[l][1,-2,-3,2]*l_LR(mpsright,k)[2,1]*r_LR(mpsright,k)[-1,-4]
                end
            end
        end

        if reduce((a,b)->contains(ham,b,i,i),1:len,init=true)
            (found[i],convhist)=linsolve(rBs[i]+start[i],rBs[i]+start[i],GMRES()) do y
                x=reduce((a,b)->exci_transfer_right(a,ham[b,i,i],mpsleft.AL[b],mpsright.AR[b])*exp(1im*p),len:-1:1,init=y)

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
