# Given a state and it's environments, we can act on it

"""
    One-site derivative
"""
function ac_prime(x::MPSTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache)
    ham=cache.opp

    toret=zero(x)
    for (i,j) in opkeys(ham,pos)
        @tensor toret[-1,-2,-3]+=leftenv(cache,pos,mps)[i][-1,5,4]*x[4,2,1]*ham[pos,i,j][5,-2,3,2]*rightenv(cache,pos,mps)[j][1,3,-3]
    end
    for (i,j) in scalkeys(ham,pos)
        scal = ham.Os[pos,i,j];
        @tensor toret[-1,-2,-3]+=leftenv(cache,pos,mps)[i][-1,5,4]*(scal*x)[4,-2,1]*rightenv(cache,pos,mps)[j][1,5,-3]
    end

    return toret
end

function ac_prime(x::GenericMPSTensor{S,3},pos::Int,mpo,cache) where S
    ham=cache.opp

    toret=zero(x)
    for (i,j) in keys(ham,pos)
        opp = ham[pos,i,j]

        if isbelow(ham,i)
            @tensor toret[-1,-2,-3,-4] +=   leftenv(cache,pos,mpo)[i][-1,8,7]*
                                            x[7,2,-3,1]*
                                            opp[8,-2,3,2]*
                                            rightenv(cache,pos,mpo)[j][1,3,-4]
        else
            @tensor toret[-1,-2,-3,-4] +=   leftenv(cache,pos,mpo)[i][-1,6,7]*
                                            x[7,-2,4,2]*
                                            opp[6,4,5,-3]*
                                            rightenv(cache,pos,mpo)[j][2,5,-4]
        end
    end

    return toret
end
function ac_prime(x::MPSTensor, row::Int,col::Int,mps::Union{InfiniteMPS,MPSMultiline}, pars::PerMPOInfEnv)
    @tensor toret[-1 -2;-3]:=leftenv(pars,row,col,mps)[-1,2,1]*x[1,3,4]*(pars.opp[row,col])[2,-2,5,3]*rightenv(pars,row,col,mps)[4,5,-3]
end

"""
    Two-site derivative
"""
function ac2_prime(x::MPOTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache)
    ham=cache.opp

    toret=zero(x)

    for (i,j) in keys(ham,pos)
        for k in 1:ham.odim
            if contains(ham,pos+1,j,k)
                #can be sped up for scalar fields
                @tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mps)[i][-1,7,6]*x[6,5,3,1]*ham[pos,i,j][7,-2,4,5]*ham[pos+1,j,k][4,-3,2,3]*rightenv(cache,pos+1,mps)[k][1,2,-4]
            end
        end

    end

    return toret
end
function ac2_prime(x::AbstractTensorMap,pos::Int,mpo,cache::FinEnv{<:ComAct})
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
                @tensor toret[-1,-2,-3,-4,-5,-6] += leftenv(cache,pos,mpo)[i][-1,2,1]*
                                                    x[1,3,-3,5,-5,7]*
                                                    opp1[2,-2,4,3]*
                                                    opp2[4,-4,6,5]*
                                                    rightenv(cache,pos+1,mpo)[l][7,6,-6]
            else
                @tensor toret[-1,-2,-3,-4,-5,-6] += leftenv(cache,pos,mpo)[i][-1,2,1]*
                                                    x[1,-2,3,-4,5,7]*
                                                    opp1[2,3,4,-3]*
                                                    opp2[4,5,6,-5]*
                                                    rightenv(cache,pos+1,mpo)[l][7,6,-6]
            end
        end
    end

    return toret
end
function ac2_prime(x::MPOTensor, row::Int,col::Int,mps::Union{InfiniteMPS,MPSMultiline}, pars::PerMPOInfEnv)
    @tensor toret[-1 -2;-3 -4]:=leftenv(pars,row,col,mps)[-1,2,1]*
                                x[1,3,4,5]*
                                pars.opp[row,col][2,-2,6,3]*
                                pars.opp[row,col+1][6,-3,7,4]*
                                rightenv(pars,row,col+1,mps)[5,7,-4]
end

"""
    Zero-site derivative (the C matrix to the right of pos)
"""
function c_prime(x::MPSBondTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache)
    toret=zero(x)
    ham=cache.opp

    for i in 1:ham.odim
        @tensor toret[-1,-2]+=leftenv(cache,pos+1,mps)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mps)[i][3,2,-2]
    end

    return toret
end
function c_prime(x::TensorMap, row::Int,col::Int, mps::Union{InfiniteMPS,MPSMultiline}, pars::PerMPOInfEnv)
    @tensor toret[-1;-2] := leftenv(pars,row,col+1,mps)[-1,3,1]*x[1,2]*rightenv(pars,row,col,mps)[2,3,-2]
end
