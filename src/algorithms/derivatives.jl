# Given a state and it's environments, we can act on it

"""
    One-site derivative
"""
function ac_prime(x::MPSTensor,ham::SparseMPOSlice,leftenv,rightenv)
    local toret

    @floop for (i,j) in keys(ham)
        if isscal(ham,i,j)
            scal = ham.Os[i,j];
            @plansor t[-1 -2;-3] := leftenv[i][-1 5;4]*(scal*x)[4 6;1]*τ[6 5;7 -2]*rightenv[j][1 7;-3]
        else
            @plansor t[-1 -2;-3] := leftenv[i][-1 5;4]*x[4 2;1]*ham[i,j][5 -2;2 3]*rightenv[j][1 3;-3]
        end

        @reduce(toret += t)
    end

    return toret
end

function ac_prime(x::MPSTensor,opp::MPOTensor,leftenv,rightenv)
    @plansor toret[-1 -2;-3] := leftenv[-1 2;1]*x[1 3;4]*opp[2 -2; 3 5]*rightenv[4 5;-3]
end

ac_prime(x::MPSTensor,::Nothing,leftenv,rightenv) = _transpose_front(leftenv*_transpose_tail(x*rightenv))


"""
    Two-site derivative
"""
function ac2_prime(x::MPOTensor,h1::SparseMPOSlice,h2::SparseMPOSlice,leftenv,rightenv)
    local toret

    @floop for (i,j) in collect(keys(h1)), k in 1:h1.odim
        contains(h2,j,k) || continue

        if isscal(h1,i,j) && isscal(h2,j,k)
            scal = h1.Os[i,j]*h2.Os[j,k]
            @plansor t[-1 -2;-3 -4] := (scal*leftenv[i])[-1 7;6]*x[6 5;1 3]*τ[7 -2;5 4]*τ[4 -4;3 2]*rightenv[k][1 2;-3]
        elseif isscal(h1,i,j)
            scal = h1.Os[i,j]
            @plansor t[-1 -2;-3 -4] := (scal*leftenv[i])[-1 7;6]*x[6 5;1 3]*τ[7 -2;5 4]*h2[j,k][4 -4;3 2]*rightenv[k][1 2;-3]
        elseif isscal(h2,j,k)
            scal = h2.Os[j,k]
            @plansor t[-1 -2;-3 -4] := (scal*leftenv[i])[-1 7;6]*x[6 5;1 3]*h1[i,j][7 -2;5 4]*τ[4 -4;3 2]*rightenv[k][1 2;-3]
        else
            @plansor t[-1 -2;-3 -4] := leftenv[i][-1 7;6]*x[6 5;1 3]*h1[i,j][7 -2;5 4]*h2[j,k][4 -4;3 2]*rightenv[k][1 2;-3]
        end

        @reduce(toret+=t)
    end

    return toret
end
function ac2_prime(x::MPOTensor,opp1::MPOTensor,opp2::MPOTensor,leftenv,rightenv)
    @plansor toret[-1 -2;-3 -4] := leftenv[-1 7;6]*x[6 5;1 3]*opp1[7 -2;5 4]*opp2[4 -4;3 2]*rightenv[1 2;-3]
end
function ac2_prime(x::MPOTensor,::Nothing,::Nothing,leftenv,rightenv)
    @plansor y[-1 -2;-3 -4] := x[1 -2;2 -4]*leftenv[-1;1]*rightenv[2;-3]
end
"""
    Zero-site derivative (the C matrix to the right of pos)
"""
function c_prime(x,leftenv::AbstractVector,rightenv::AbstractVector)
    sum(zip(leftenv,rightenv)) do (le,re)
        c_prime(x,le,re)
    end
end

function c_prime(x, leftenv::MPSTensor,rightenv::MPSTensor)
    @plansor toret[-1;-2] := leftenv[-1 3;1]*x[1;2]*rightenv[2 3;-2]
end

function c_prime(x, leftenv::MPSBondTensor,rightenv::MPSBondTensor)
    @plansor toret[-1;-2] := leftenv[-1;1]*x[1;2]*rightenv[2;-2]
end

#not breaking everything immediatly
function ac_prime(x::MPSTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache::Union{FinEnv,MPOHamInfEnv,IDMRGEnv})
    ac_prime(x,cache.opp[pos],leftenv(cache,pos,mps),rightenv(cache,pos,mps))
end
function ac_prime(x::MPSTensor, row::Int,col::Int,mps::Union{InfiniteMPS,MPSMultiline}, envs::PerMPOInfEnv)
    ac_prime(x,envs.opp[row,col],leftenv(envs,row,col,mps),rightenv(envs,row,col,mps));
end
function ac2_prime(x::MPOTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache::Union{FinEnv,MPOHamInfEnv,IDMRGEnv})
    ac2_prime(x,cache.opp[pos],cache.opp[pos+1],leftenv(cache,pos,mps),rightenv(cache,pos+1,mps));
end
function ac2_prime(x::MPOTensor, row::Int,col::Int,mps::Union{InfiniteMPS,MPSMultiline}, envs::PerMPOInfEnv)
    ac2_prime(x,envs.opp[row,col],envs.opp[row,col+1],leftenv(envs,row,col,mps),rightenv(envs,row,col+1,mps))
end
function c_prime(x::MPSBondTensor,pos::Int,mps::Union{FiniteMPS,InfiniteMPS,MPSComoving},cache)
    c_prime(x,leftenv(cache,pos+1,mps),rightenv(cache,pos,mps))
end
function c_prime(x::TensorMap, row::Int,col::Int, mps::Union{InfiniteMPS,MPSMultiline}, envs::PerMPOInfEnv)
    c_prime(x,leftenv(envs,row,col+1,mps),rightenv(envs,row,col,mps))
end

#allow calling them with CartesianIndices
ac_prime(x,pos::CartesianIndex,mps,envs) = ac_prime(x,Tuple(pos)...,mps,envs)
ac2_prime(x,pos::CartesianIndex,mps,envs) = ac2_prime(x,Tuple(pos)...,mps,envs)
c_prime(x,pos::CartesianIndex,mps,envs) = c_prime(x,Tuple(pos)...,mps,envs)

#downproject for approximate
c_proj(pos,below,envs::FinEnv) = c_prime(envs.above.CR[pos],leftenv(envs,pos+1,below),rightenv(envs,pos,below))

function c_proj(row,col,below,envs::PerMPOInfEnv)
    c_prime(envs.above.CR[row,col],leftenv(envs,row,col+1,below),rightenv(envs,row,col,below))
end

function ac_proj(pos,below,envs)
    le = leftenv(envs,pos,below)
    re = rightenv(envs,pos,below)

    ac_prime(envs.above.AC[pos],envs.opp[pos],le,re)
end
function ac_proj(row,col,below,envs::PerMPOInfEnv)
    ac_prime(envs.above.AC[row,col],envs.opp[row,col],leftenv(envs,row,col,below),rightenv(envs,row,col,below))
end
function ac2_proj(pos,below,envs)
    le = leftenv(envs,pos,below)
    re = rightenv(envs,pos+1,below)

    ac2_prime(envs.above.AC[pos]*_transpose_tail(envs.above.AR[pos+1]),envs.opp[pos],envs.opp[pos+1],le,re)
end
function ac2_proj(row,col,below,envs::PerMPOInfEnv)
    @tensor ac2[-1 -2;-3 -4] := envs.above.AC[row,col][-1 -2;1]*envs.above.AR[row,col+1][1 -4;-3]
    ac2_prime(ac2,leftenv(envs,row,col+1,below),rightenv(envs,row,col+1,below))
end
