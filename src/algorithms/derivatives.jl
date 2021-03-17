# Given a state and it's environments, we can act on it

#allow calling them with CartesianIndices
ac_prime(x,pos::CartesianIndex,mps,envs) = ac_prime(x,Tuple(pos)...,mps,envs)
ac2_prime(x,pos::CartesianIndex,mps,envs) = ac2_prime(x,Tuple(pos)...,mps,envs)
c_prime(x,pos::CartesianIndex,mps,envs) = c_prime(x,Tuple(pos)...,mps,envs)

"""
    One-site derivative
"""
function ac_prime(x::MPSTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache::Union{FinEnv,MPOHamInfEnv,IdmrgEnv})
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

function ac_prime(x::MPSTensor, row::Int,col::Int,mps::Union{InfiniteMPS,MPSMultiline}, envs::Union{MixPerMPOInfEnv,PerMPOInfEnv})
    @tensor toret[-1 -2;-3]:=leftenv(envs,row,col,mps)[-1,2,1]*x[1,3,4]*(envs.opp[row,col])[2,-2,5,3]*rightenv(envs,row,col,mps)[4,5,-3]
end

"""
    Two-site derivative
"""
function ac2_prime(x::MPOTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache::Union{FinEnv,MPOHamInfEnv,IdmrgEnv})
    ham=cache.opp

    toret=zero(x)

    for (i,j) in keys(ham,pos)
        for k in 1:ham.odim
            contains(ham,pos+1,j,k) || continue

            if isscal(ham,pos,i,j) && isscal(ham,pos+1,j,k)
                scal = ham.Os[pos,i,j]*ham.Os[pos+1,j,k]
                @tensor toret[-1,-2,-3,-4] += (scal*leftenv(cache,pos,mps)[i])[-1,7,6]*x[6,-2,-3,1]*rightenv(cache,pos+1,mps)[k][1,7,-4]
            elseif isscal(ham,pos,i,j)
                scal = ham.Os[pos,i,j]
                @tensor toret[-1,-2,-3,-4]+=(scal*leftenv(cache,pos,mps)[i])[-1,7,6]*x[6,-2,3,1]*ham[pos+1,j,k][7,-3,2,3]*rightenv(cache,pos+1,mps)[k][1,2,-4]
            elseif isscal(ham,pos+1,j,k)
                scal = ham.Os[pos+1,j,k]
                @tensor toret[-1,-2,-3,-4]+=(scal*leftenv(cache,pos,mps)[i])[-1,7,6]*x[6,5,-3,1]*ham[pos,i,j][7,-2,2,5]*rightenv(cache,pos+1,mps)[k][1,2,-4]
            else
                @tensor toret[-1,-2,-3,-4]+=leftenv(cache,pos,mps)[i][-1,7,6]*x[6,5,3,1]*ham[pos,i,j][7,-2,4,5]*ham[pos+1,j,k][4,-3,2,3]*rightenv(cache,pos+1,mps)[k][1,2,-4]
            end
        end

    end

    return toret
end
function ac2_prime(x::MPOTensor, row::Int,col::Int,mps::Union{InfiniteMPS,MPSMultiline}, envs::Union{MixPerMPOInfEnv,PerMPOInfEnv})
    @tensor toret[-1 -2;-3 -4]:=leftenv(envs,row,col,mps)[-1,2,1]*
                                x[1,3,4,5]*
                                envs.opp[row,col][2,-2,6,3]*
                                envs.opp[row,col+1][6,-3,7,4]*
                                rightenv(envs,row,col+1,mps)[5,7,-4]
end

"""
    Zero-site derivative (the C matrix to the right of pos)
"""
function c_prime(x::MPSBondTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving,IdmrgEnv},cache)
    toret=zero(x)
    ham=cache.opp

    for i in 1:ham.odim
        @tensor toret[-1,-2]+=leftenv(cache,pos+1,mps)[i][-1,2,1]*x[1,3]*rightenv(cache,pos,mps)[i][3,2,-2]
    end

    return toret
end
function c_prime(x::TensorMap, row::Int,col::Int, mps::Union{InfiniteMPS,MPSMultiline}, envs::Union{MixPerMPOInfEnv,PerMPOInfEnv})
    @tensor toret[-1;-2] := leftenv(envs,row,col+1,mps)[-1,3,1]*x[1,2]*rightenv(envs,row,col,mps)[2,3,-2]
end
