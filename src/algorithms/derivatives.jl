# Given a state and it's environments, we can act on it

"""
    Draft operators
"""
struct MPO_∂∂C{L,R}
    leftenv::L
    rightenv::R
end

struct MPO_∂∂AC{O,L,R}
    o::O
    leftenv::L
    rightenv::R
end


struct MPO_∂∂AC2{O,L,R}
    o1::O
    o2::O
    leftenv::L
    rightenv::R
end

Base.:*(h::Union{<:MPO_∂∂C,<:MPO_∂∂AC,<:MPO_∂∂AC2},v) = h(v);

(h::MPO_∂∂C)(x) = ∂C(x,h.leftenv,h.rightenv);
(h::MPO_∂∂AC)(x) = ∂AC(x,h.o,h.leftenv,h.rightenv);
(h::MPO_∂∂AC2)(x) = ∂AC2(x,h.o1,h.o2,h.leftenv,h.rightenv);

# draft operator constructors
∂∂C(pos::Int,mps,opp::Union{MPOHamiltonian,SparseMPO,DenseMPO},cache) =
    MPO_∂∂C(leftenv(cache,pos+1,mps),rightenv(cache,pos,mps))
∂∂C(col::Int, mps, opp::MPOMultiline, envs) =
    MPO_∂∂C(leftenv(envs,col+1,mps),rightenv(envs,col,mps))
∂∂C(row::Int,col::Int, mps, opp::MPOMultiline, envs) =
    MPO_∂∂C(leftenv(envs,row,col+1,mps),rightenv(envs,row,col,mps))

∂∂AC(pos::Int,mps,opp::Union{MPOHamiltonian,SparseMPO,DenseMPO},cache) =
    MPO_∂∂AC(cache.opp[pos],leftenv(cache,pos,mps),rightenv(cache,pos,mps))
∂∂AC(row::Int,col::Int,mps, opp::MPOMultiline, envs) =
    MPO_∂∂AC(envs.opp[row,col],leftenv(envs,row,col,mps),rightenv(envs,row,col,mps));
∂∂AC(col::Int,mps, opp::MPOMultiline, envs) =
    MPO_∂∂AC(envs.opp[:,col],leftenv(envs,col,mps),rightenv(envs,col,mps));

∂∂AC2(pos::Int,mps,opp::Union{MPOHamiltonian,SparseMPO,DenseMPO},cache) =
    MPO_∂∂AC2(cache.opp[pos],cache.opp[pos+1],leftenv(cache,pos,mps),rightenv(cache,pos+1,mps));
∂∂AC2(col::Int,mps, opp::MPOMultiline,envs) =
    MPO_∂∂AC2(envs.opp[:,col],envs.opp[:,col+1],leftenv(envs,col,mps),rightenv(envs,col+1,mps))
∂∂AC2(row::Int,col::Int,mps, opp::MPOMultiline,envs) =
    MPO_∂∂AC2(envs.opp[row,col],envs.opp[row,col+1],leftenv(envs,row,col,mps),rightenv(envs,row,col+1,mps))

#allow calling them with CartesianIndices
∂∂C(pos::CartesianIndex,mps,opp,envs) = ∂∂C(Tuple(pos)...,mps,opp,envs)
∂∂AC(pos::CartesianIndex,mps,opp,envs) = ∂∂AC(Tuple(pos)...,mps,opp,envs)
∂∂AC2(pos::CartesianIndex,mps,opp,envs) = ∂∂AC2(Tuple(pos)...,mps,opp,envs)



"""
    One-site derivative
"""

function ∂AC(x::MPSTensor,ham::SparseMPOSlice,leftenv,rightenv) :: typeof(x)
    local toret

    @floop WorkStealingEx() for (i,j) in keys(ham)
        if isscal(ham,i,j)
            @plansor t[-1 -2;-3] := leftenv[i][-1 5;4]*x[4 6;1]*τ[6 5;7 -2]*rightenv[j][1 7;-3]
            lmul!(ham.Os[i,j],t);
        else
            @plansor t[-1 -2;-3] := leftenv[i][-1 5;4]*x[4 2;1]*ham[i,j][5 -2;2 3]*rightenv[j][1 3;-3]
        end

        @reduce(toret = inplace_add!(nothing,t))
    end

    return toret
end

function ∂AC(x::MPSTensor,opp::MPOTensor,leftenv,rightenv) :: typeof(x)
    @plansor toret[-1 -2;-3] := leftenv[-1 2;1]*x[1 3;4]*opp[2 -2; 3 5]*rightenv[4 5;-3]
end

# mpo multiline
∂AC(x::RecursiveVec,opp,leftenv,rightenv) =
    RecursiveVec(circshift(map(t->∂AC(t...),zip(x.vecs,opp,leftenv,rightenv)),1))

∂AC(x::MPSTensor,::Nothing,leftenv,rightenv) = _transpose_front(leftenv*_transpose_tail(x*rightenv))


"""
    Two-site derivative
"""
function ∂AC2(x::MPOTensor,h1::SparseMPOSlice,h2::SparseMPOSlice,leftenv,rightenv) :: typeof(x)
    local toret

    tl = tensormaptype(spacetype(x),2,3,storagetype(x));
    hl = Vector{Union{Nothing,tl}}(undef,h1.odim);
    @threads for j in 1:h1.odim
        @floop WorkStealingEx() for i in keys(h1,:,j)
            if isscal(h1,i,j)
                @plansor t[-1 -2;-3 -4 -5] := (h1.Os[i,j]*leftenv[i])[-1 1;2]*τ[1 -2;3 -5]*x[2 3;-3 -4]
            else
                @plansor t[-1 -2;-3 -4 -5] := leftenv[i][-1 1;2]*h1[i,j][1 -2;3 -5]*x[2 3;-3 -4]
            end
            @reduce(curel = inplace_add!(nothing,t))
        end
        hl[j] = curel; 
    end

    @floop WorkStealingEx() for (j,k) in keys(h2)
        isnothing(hl[j]) && continue

        if isscal(h2,j,k)
            @plansor t[-1 -2;-3 -4] := (h2.Os[j,k]*hl[j])[-1 -2;5 3 4]*τ[4 -4;3 6]*rightenv[k][5 6;-3]
        else
            @plansor t[-1 -2;-3 -4] := hl[j][-1 -2;5 3 4]*h2[j,k][4 -4;3 6]*rightenv[k][5 6;-3]
        end

        @reduce(toret = inplace_add!(nothing,t))
    end

    return toret
end
function ∂AC2(x::MPOTensor,opp1::MPOTensor,opp2::MPOTensor,leftenv,rightenv)
    @plansor toret[-1 -2;-3 -4] := leftenv[-1 7;6]*x[6 5;1 3]*opp1[7 -2;5 4]*opp2[4 -4;3 2]*rightenv[1 2;-3]
end
function ∂AC2(x::MPOTensor,::Nothing,::Nothing,leftenv,rightenv)
    @plansor y[-1 -2;-3 -4] := x[1 -2;2 -4]*leftenv[-1;1]*rightenv[2;-3]
end

∂AC2(x::RecursiveVec,opp1,opp2,leftenv,rightenv) =
    RecursiveVec(circshift(map(t->∂AC2(t...),zip(x.vecs,opp1,opp2,leftenv,rightenv)),1))


"""
    Zero-site derivative (the C matrix to the right of pos)
"""
function ∂C(x::MPSBondTensor,leftenv::AbstractVector,rightenv::AbstractVector) :: typeof(x)
    @floop WorkStealingEx() for (le,re) in zip(leftenv,rightenv)
        t = ∂C(x,le,re)

        @reduce(s = inplace_add!(nothing,t))
    end

    s
end

function ∂C(x::MPSBondTensor, leftenv::MPSTensor,rightenv::MPSTensor)
    @plansor toret[-1;-2] := leftenv[-1 3;1]*x[1;2]*rightenv[2 3;-2]
end

function ∂C(x::MPSBondTensor, leftenv::MPSBondTensor,rightenv::MPSBondTensor)
    @plansor toret[-1;-2] := leftenv[-1;1]*x[1;2]*rightenv[2;-2]
end

∂C(x::RecursiveVec,leftenv,rightenv) =
    RecursiveVec(circshift(map(t->∂C(t...),zip(x.vecs,leftenv,rightenv)),1))


#downproject for approximate
c_proj(pos,below,envs::FinEnv) = ∂C(envs.above.CR[pos],leftenv(envs,pos+1,below),rightenv(envs,pos,below))

function c_proj(row,col,below,envs::PerMPOInfEnv)
    ∂C(envs.above.CR[row,col],leftenv(envs,row,col+1,below),rightenv(envs,row,col,below))
end

function ac_proj(pos,below,envs)
    le = leftenv(envs,pos,below)
    re = rightenv(envs,pos,below)

    ∂AC(envs.above.AC[pos],envs.opp[pos],le,re)
end
function ac_proj(row,col,below,envs::PerMPOInfEnv)
    ∂AC(envs.above.AC[row,col],envs.opp[row,col],leftenv(envs,row,col,below),rightenv(envs,row,col,below))
end
function ac2_proj(pos,below,envs)
    le = leftenv(envs,pos,below)
    re = rightenv(envs,pos+1,below)

    ∂AC2(envs.above.AC[pos]*_transpose_tail(envs.above.AR[pos+1]),envs.opp[pos],envs.opp[pos+1],le,re)
end
function ac2_proj(row,col,below,envs::PerMPOInfEnv)
    @plansor ac2[-1 -2;-3 -4] := envs.above.AC[row,col][-1 -2;1]*envs.above.AR[row,col+1][1 -4;-3]
    ∂AC2(ac2,leftenv(envs,row,col+1,below),rightenv(envs,row,col+1,below))
end

#=
∂∂C(pos::Int,mps,opp::LinearCombination,cache) =
    LinearCombination(broadcast((h,e) -> ∂∂C(pos,mps,h,e),opp.opps,cache.envs),opp.coeffs)

∂∂AC(pos::Int,mps,opp::LinearCombination,cache) =
    LinearCombination(broadcast((h,e) -> ∂∂AC(pos,mps,h,e),opp.opps,cache.envs),opp.coeffs)


∂∂AC2(pos::Int,mps,opp::LinearCombination,cache) =
    LinearCombination(broadcast((h,e) -> ∂∂AC2(pos,mps,h,e),opp.opps,cache.envs),opp.coeffs)
=#
struct AC_EffProj{A,L}
    a1::A
    le::L
    re::L
end
struct AC2_EffProj{A,L}
    a1::A
    a2::A
    le::L
    re::L
end
Base.:*(h::Union{<:AC_EffProj,AC2_EffProj},v) = h(v);

function (h::AC_EffProj)(x::MPSTensor)
    @plansor v[-1;-2 -3 -4] := h.le[4;-1 -2 5]*h.a1[5 2;1]*h.re[1;-3 -4 3]*conj(x[4 2;3])
    @plansor y[-1 -2;-3] := conj(v[1;2 5 6])*h.le[-1;1 2 4]*h.a1[4 -2;3]*h.re[3;5 6 -3]
end
function (h::AC2_EffProj)(x::MPOTensor)
    @plansor v[-1;-2 -3 -4] := h.le[6;-1 -2 7]*h.a1[7 4;5]*h.a2[5 2;1]*h.re[1;-3 -4 3]*conj(x[6 4;3 2])
    @plansor y[-1 -2;-3 -4] := conj(v[2;3 5 6])*h.le[-1;2 3 4]*h.a1[4 -2;7]*h.a2[7 -4;1]*h.re[1;5 6 -3]
end

∂∂AC(pos::Int,state,opp::ProjectionOperator,env) =
    AC_EffProj(opp.ket.AC[pos],leftenv(env,pos,state),rightenv(env,pos,state));
∂∂AC2(pos::Int,state,opp::ProjectionOperator,env) =
    AC2_EffProj(opp.ket.AC[pos],opp.ket.AR[pos+1],leftenv(env,pos,state),rightenv(env,pos+1,state));

# MultipliedOperator and SumOfOperators
∂∂C(pos::Int,mps,opp::MultipliedOperator,cache) =
MultipliedOperator(∂∂C(pos::Int,mps,opp.op,cache),opp.f)

∂∂AC(pos::Int,mps,opp::MultipliedOperator,cache) =
MultipliedOperator(∂∂AC(pos::Int,mps,opp.op,cache),opp.f)

∂∂AC2(pos::Int,mps,opp::MultipliedOperator,cache) =
MultipliedOperator(∂∂AC2(pos::Int,mps,opp.op,cache),opp.f)

∂∂C(pos::Int,mps,opp::SumOfOperators,cache::MultipleEnvironments) =
    SumOfOperators( map((op,openv)->∂∂C(pos,mps,op,openv),opp.ops,cache.envs) )

∂∂AC(pos::Int,mps,opp::SumOfOperators,cache::MultipleEnvironments) =
    SumOfOperators( map((op,openv)->∂∂AC(pos,mps,op,openv),opp.ops,cache.envs) )

∂∂AC2(pos::Int,mps,opp::SumOfOperators,cache::MultipleEnvironments) =
    SumOfOperators( map((op,openv)->∂∂AC2(pos,mps,op,openv),opp.ops,cache.envs) )

