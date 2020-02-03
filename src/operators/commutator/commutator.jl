#implements 'effecient' code for the commutator acting on an mpo
#code is a bit ugly, and can be speed up

struct ComAct{T1<:MpoHamiltonian,T2<:MpoHamiltonian} <: Hamiltonian
    below::T1
    above::T2
end

ComAct(H::MpoHamiltonian,commutator=true) = ComAct(H,(commutator ? -1 : 1)*H)

function Base.getproperty(h::ComAct,f::Symbol)
    if f==:odim
        return (h.below.odim+h.above.odim)::Int
    elseif f==:domspaces
        return Periodic([[h.below.domspaces[i];h.above.domspaces[i]] for i in 1:h.period])
    elseif f==:imspaces
        return Periodic([[h.below.imspaces[i];h.above.imspaces[i]] for i in 1:h.period])
    elseif f==:period
        @assert h.above.period == h.below.period
        return h.below.period::Int
    else
        return getfield(h,f)
    end
end

summap(a::ComAct,isbelow,k) = isbelow ? k : a.below.odim+k
isbelow(a::ComAct,k) = k<=a.below.odim
keys(a::ComAct,pos) = [[(summap(a,true,i[1]),summap(a,true,i[2])) for i in keys(a.below,pos)];[(summap(a,false,i[1]),summap(a,false,i[2])) for i in keys(a.above,pos)]]

function Base.getindex(a::ComAct,pos,i,j)
    if i<=a.below.odim && j<=a.below.odim
        return a.below[pos,i,j]
    elseif i>a.below.odim && j>a.below.odim
        return a.above[pos,i-a.below.odim,j-a.below.odim]
    else
        @assert false
        #return TensorMap(zeros,eltype(a.above[1,1,1]),a.domspaces[pos][i]*a.pspaces[pos],a.imspaces[pos][j]'*a.pspaces[pos])
    end
end

include("fmcache.jl")

function ac_prime(x::MpoType,pos,mpo,cache::FiniteMpoCache)
    ham=cache.opp

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]

        if isbelow(ham,i)
            #for vub project
            #@tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mpo)[i][-1,8,7]*x[7,1,2,-4]*opp[8,-2,3,1]*rightenv(cache,pos,mpo)[j][2,3,-3]
            @tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mpo)[i][-1,8,7]*x[7,2,1,-4]*opp[8,-2,3,2]*rightenv(cache,pos,mpo)[j][1,3,-3]
        else
            #@tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mpo)[i][-1,7,6]*x[7,-2,2,4]*opp[6,4,5,-4]*rightenv(cache,pos,mpo)[j][5,2,-3]
            @tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mpo)[i][-1,7,6]*x[7,-2,2,4]*opp[6,4,5,-4]*rightenv(cache,pos,mpo)[j][5,2,-3]
        end
    end

    return toret
end

function ac2_prime(x,pos,mpo,cache::FiniteMpoCache)
    ham=cache.opp

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        for (k,l) in keys(ham,pos+1)
            if j!=k
                continue
            end
            opp1 = ham[pos,i,j]
            opp2 = ham[pos+1,k,l]

            if isbelow(ham,i)
                @tensor toret[-1,-2,-3,-4,-5,-6] += leftenv(cache,pos,mpo)[i][-1,2,1]*x[1,3,5,7,-5,-6]*opp1[2,-2,4,3]*opp2[4,-3,6,5]*rightenv(cache,pos+1,mpo)[l][7,6,-4]
            else
                @tensor toret[-1,-2,-3,-4,-5,-6] += leftenv(cache,pos,mpo)[i][-1,1,2]*x[1,-2,-3,7,5,3]*opp1[2,3,4,-6]*opp2[4,5,6,-5]*rightenv(cache,pos+1,mpo)[l][6,7,-4]
            end
        end
    end

    return toret
end

#C to the right of pos
function c_prime(x::MpsVecType,pos,mpo,cache::FiniteMpoCache)
    toret=zero(x)
    ham=cache.opp

    for i in 1:ham.odim
        if isbelow(ham,i)
            @tensor toret[-1,-2]+=leftenv(cache,pos+1,mpo)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mpo)[i][3,2,-2]
        else
            @tensor toret[-1,-2]+=leftenv(cache,pos+1,mpo)[i][-1,1,2]*x[1,3]*rightenv(cache,pos,mpo)[i][2,3,-2]
        end
    end

    return toret
end

#specialized mpohamiltonian transfer function (not yet typestable)
#¢an be cleaned up quite easily but I'm lazy
#also not yet optimal
function mps_apply_transfer_left(vec,ham::ComAct,pos,A,Ab=A)
    toreturn = similar(vec)
    assigned=[false for i in vec]

    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]

        if assigned[j]
            if isbelow(ham,i)
                @tensor toreturn[j][-1 -2;-3] += vec[i][1,3,4]*conj(Ab[1,2,-1,6])*opp[3,2,-2,5]*A[4,5,-3,6]
            else
                @tensor toreturn[j][-1 -2;-3] += vec[i][6,1,2]*conj(Ab[6,5,-1,4])*A[1,5,-2,3]*opp[2,3,-3,4]
            end
        else
            assigned[j]=true
            if isbelow(ham,i)
                @tensor toreturn[j][-1 -2;-3] := vec[i][1,3,4]*conj(Ab[1,2,-1,6])*opp[3,2,-2,5]*A[4,5,-3,6]
            else
                @tensor toreturn[j][-1 -2;-3] := vec[i][6,1,2]*conj(Ab[6,5,-1,4])*A[1,5,-2,3]*opp[2,3,-3,4]
            end
        end
    end

    for k in 1:ham.odim
        if !assigned[k]
            opp = ham[pos,k,k]
            if isbelow(ham,k)
                @tensor toreturn[k][-1 -2;-3] := vec[k][1,3,4]*conj(Ab[1,2,-1,6])*opp[3,2,-2,5]*A[4,5,-3,6]
            else
                @tensor toreturn[k][-1 -2;-3] := vec[k][6,1,2]*conj(Ab[6,5,-1,4])*A[1,5,-2,3]*opp[2,3,-3,4]
            end
        end
    end

    return toreturn
end
function mps_apply_transfer_right(vec,ham::ComAct,pos,A,Ab=A)
    toreturn = similar(vec)
    assigned=[false for i in vec]

    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]

        if assigned[i]
            if isbelow(ham,i)
                @tensor toreturn[i][-1 -2;-3] += vec[j][4,3,1]*conj(Ab[-3,2,1,6])*opp[-2,2,3,5]*A[-1,5,4,6]
            else
                @tensor toreturn[i][-1 -2;-3] += vec[j][2,1,6]*conj(Ab[-3,5,6,4])*A[-2,5,1,3]*opp[-1,3,2,4]
            end
        else
            assigned[i]=true
            if isbelow(ham,i)
                @tensor toreturn[i][-1 -2;-3] := vec[j][4,3,1]*conj(Ab[-3,2,1,6])*opp[-2,2,3,5]*A[-1,5,4,6]
            else
                @tensor toreturn[i][-1 -2;-3] := vec[j][2,1,6]*conj(Ab[-3,5,6,4])*A[-2,5,1,3]*opp[-1,3,2,4]
            end
        end
    end

    for k in 1:ham.odim
        if !assigned[k]
            opp = ham[pos,k,k]
            if isbelow(ham,k)
                @tensor toreturn[k][-1 -2;-3] := vec[k][4,3,1]*conj(Ab[-3,2,1,6])*opp[-2,2,3,5]*A[-1,5,4,6]
            else
                @tensor toreturn[k][-1 -2;-3] := vec[k][2,1,6]*conj(Ab[-3,5,6,4])*A[-2,5,1,3]*opp[-1,3,2,4]
            end
        end
    end

    return toreturn
end
