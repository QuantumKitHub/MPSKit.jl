# Given a state and its environments, we can act on it

"""
    Draft operators
"""
struct MPO_∂∂C{L,R}
    leftenv::L
    rightenv::R
end

struct MPO_∂∂AC{L,R}
    leftblocks::L
    rightblocks::R
end

struct MPO_∂∂AC2{L,R}
    leftblocks::L
    rightblocks::R
end

const DerivativeOperator = Union{MPO_∂∂C,MPO_∂∂AC,MPO_∂∂AC2}

Base.:*(h::Union{MPO_∂∂C,MPO_∂∂AC,MPO_∂∂AC2}, v) = h(v);

(h::MPO_∂∂C)(x) = ∂C(x, h.leftenv, h.rightenv);
(h::MPO_∂∂AC)(x) = ∂AC(x, h.leftblocks, h.rightblocks);
(h::MPO_∂∂AC2)(x) = ∂AC2(x, h.leftblocks, h.rightblocks);
(h::DerivativeOperator)(v, ::Number) = h(v)

function MPO_∂∂AC(opp::SparseMPOSlice,l,r)

    un_r = unique(last.(keys(opp)))

    blocked = map(un_r) do j
        left_blocked = sum(map(keys(opp,:,j)) do i
            @plansor left_blocked[-1 -2 -3;-4 -5] := l[i][-1 1;-4]*opp[i,j][1 -2;-5 -3]
        end)

        (left_blocked,r[j])
    end
    return MPO_∂∂AC(first.(blocked),last.(blocked))
end

function MPO_∂∂AC(opp::MPOTensor,l,r)
    @plansor left_blocked[-1 -2 -3;-4 -5] := l[-1 1;-4]*opp[1 -2;-5 -3]
    return MPO_∂∂AC([left_blocked],[r])
end

function MPO_∂∂AC(opp::AbstractVector{T},l,r) where T<:MPOTensor
    blocked = map(zip(opp,l,r)) do (o,left,right)
        @plansor left_blocked[-1 -2 -3;-4 -5] := left[-1 1;-4]*o[1 -2;-5 -3]

        (left_blocked,right)
    end
    return MPO_∂∂AC(first.(blocked),last.(blocked))
end


function MPO_∂∂AC2(opp1::SparseMPOSlice,opp2::SparseMPOSlice,l,r)

    middle = intersect(unique(last.(keys(opp1))),unique(first.(keys(opp2))))

    lblocked = map(middle) do j
        sum(map(keys(opp1,:,j)) do i
            @plansor left_blocked[-1 -2 -3;-4 -5] := l[i][-1 1;-4]*opp1[i,j][1 -2;-5 -3]
        end)
    end

    rblocked = map(middle) do i
        sum(map(keys(opp2,i,:)) do j
            @plansor right_blocked[-1 -2 -3;-4 -5] := opp2[i,j][-3 -5;-2 1]*r[j][-1 1;-4]
        end)
    end

    MPO_∂∂AC2(lblocked,rblocked)
end

function MPO_∂∂AC2(opp1::MPOTensor,opp2::MPOTensor,l,r)
    @plansor left_blocked[-1 -2 -3;-4 -5] := l[-1 1;-4]*opp1[1 -2;-5 -3]
    @plansor right_blocked[-1 -2 -3;-4 -5] := opp2[-3 -5;-2 1]*r[-1 1;-4]

    MPO_∂∂AC2([left_blocked],[right_blocked])
end

function MPO_∂∂AC2(opp1::AbstractVector{T},opp2::AbstractVector{T},l,r) where T<:MPOTensor
    lblocked = map(zip(l,opp1)) do (cl,o)
        @plansor left_blocked[-1 -2 -3;-4 -5] := cl[-1 1;-4]*o[1 -2;-5 -3]
    end

    rblocked = map(zip(opp2,r)) do (o,cr)
        @plansor right_blocked[-1 -2 -3;-4 -5] := o[-3 -5;-2 1]*cr[-1 1;-4]
    end

    MPO_∂∂AC2(lblocked,rblocked)
end
# draft operator constructors
function ∂∂C(pos::Int, mps, opp::Union{MPOHamiltonian,SparseMPO,DenseMPO}, cache)
    return MPO_∂∂C(leftenv(cache, pos + 1, mps), rightenv(cache, pos, mps))
end
function ∂∂C(col::Int, mps, opp::MPOMultiline, envs)
    return MPO_∂∂C(leftenv(envs, col + 1, mps), rightenv(envs, col, mps))
end
function ∂∂C(row::Int, col::Int, mps, opp::MPOMultiline, envs)
    return MPO_∂∂C(leftenv(envs, row, col + 1, mps), rightenv(envs, row, col, mps))
end

function ∂∂AC(pos::Int, mps, opp::Union{MPOHamiltonian,SparseMPO,DenseMPO}, cache)
    return MPO_∂∂AC(cache.opp[pos], leftenv(cache, pos, mps), rightenv(cache, pos, mps))
end
function ∂∂AC(row::Int, col::Int, mps, opp::MPOMultiline, envs)
    return MPO_∂∂AC(envs.opp[row, col], leftenv(envs, row, col, mps),
                    rightenv(envs, row, col, mps))
end;
function ∂∂AC(col::Int, mps, opp::MPOMultiline, envs)
    return MPO_∂∂AC(envs.opp[:, col], leftenv(envs, col, mps), rightenv(envs, col, mps))
end;

function ∂∂AC2(pos::Int, mps, opp::Union{MPOHamiltonian,SparseMPO,DenseMPO}, cache)
    return MPO_∂∂AC2(cache.opp[pos], cache.opp[pos + 1], leftenv(cache, pos, mps),
                     rightenv(cache, pos + 1, mps))
end;
function ∂∂AC2(col::Int, mps, opp::MPOMultiline, envs)
    return MPO_∂∂AC2(envs.opp[:, col], envs.opp[:, col + 1], leftenv(envs, col, mps),
                     rightenv(envs, col + 1, mps))
end
function ∂∂AC2(row::Int, col::Int, mps, opp::MPOMultiline, envs)
    return MPO_∂∂AC2(envs.opp[row, col], envs.opp[row, col + 1],
                     leftenv(envs, row, col, mps), rightenv(envs, row, col + 1, mps))
end

#allow calling them with CartesianIndices
∂∂C(pos::CartesianIndex, mps, opp, envs) = ∂∂C(Tuple(pos)..., mps, opp, envs)
∂∂AC(pos::CartesianIndex, mps, opp, envs) = ∂∂AC(Tuple(pos)..., mps, opp, envs)
∂∂AC2(pos::CartesianIndex, mps, opp, envs) = ∂∂AC2(Tuple(pos)..., mps, opp, envs)

"""
    One-site derivative
"""

function ∂AC(x::MPSTensor, leftblocks,rightblocks)::typeof(x)
    local y
    @static if Defaults.parallelize_derivatives
        @floop WorkStealingEx() for (lb, rb) in zip(leftblocks,rightblocks)
            @plansor t[-1 -2;-3] := lb[-1 -2 3;1 2]*x[1 2;4]*rb[4 3;-3]
            @reduce(y = inplace_add!(nothing, t))
        end
    else
        @plansor y[-1 -2;-3] := leftblocks[1][-1 -2 3;1 2]*x[1 2;4]*rightblocks[1][4 3;-3]
        for (lb, rb) in zip(leftblocks[2:end],rightblocks[2:end])
            @plansor y[-1 -2;-3] += lb[-1 -2 3;1 2]*x[1 2;4]*rb[4 3;-3]
        end
    end

    return y
end

# mpo multiline
function ∂AC(x::RecursiveVec, leftblocks,rightblocks)::typeof(x)
    return RecursiveVec(circshift(map(zip(x.vecs,leftblocks,rightblocks)) do (x,l,r)
        @plansor y[-1 -2;-3] := l[-1 -2 3;1 2]*x[1 2;4]*r[4 3;-3]
    end,1))
end


∂AC(x::MPSTensor, h, leftenv, rightenv) = MPO_∂∂AC(h,leftenv,rightenv)(x)
function ∂AC(x::MPSTensor, ::Nothing, leftenv, rightenv)
    return _transpose_front(leftenv * _transpose_tail(x * rightenv))
end

"""
    Two-site derivative
"""
function ∂AC2(x::MPOTensor, leftblocks,rightblocks)
    tl = tensormaptype(spacetype(x), 2, 3, storagetype(x))
    hl = Vector{Union{Nothing,tl}}(undef, length(leftblocks))
    @floop WorkStealingEx()  for (i,lb) in enumerate(leftblocks)
        @plansor hl[i][-1 -2;-3 -4 -5] := lb[-1 -2 -5;1 2]*x[1 2;-3 -4]
    end
    
    @floop WorkStealingEx() for (i,rb) in enumerate(rightblocks)
        t = hl[i]*rb
        
        @reduce(toret = inplace_add!(nothing, t))
    end

    return toret
end

function ∂AC2(x::MPOTensor, h1::SparseMPOSlice, h2::SparseMPOSlice, leftenv,
              rightenv)::typeof(x)
    return MPO_∂∂AC2(h1,h2,leftenv,rightenv)(x)
end
function ∂AC2(x::MPOTensor, opp1::MPOTensor, opp2::MPOTensor, leftenv, rightenv)
    return MPO_∂∂AC2(opp1,opp2,leftenv,rightenv)(x)
end
function ∂AC2(x::MPOTensor, ::Nothing, ::Nothing, leftenv, rightenv)
    @plansor y[-1 -2; -3 -4] := x[1 -2; 2 -4] * leftenv[-1; 1] * rightenv[2; -3]
end

function ∂AC2(x::RecursiveVec, opp1, opp2, leftenv, rightenv)
    return RecursiveVec(circshift(map(t -> ∂AC2(t...),
                                      zip(x.vecs, opp1, opp2, leftenv, rightenv)), 1))
end

"""
    Zero-site derivative (the C matrix to the right of pos)
"""
function ∂C(x::MPSBondTensor, leftenv::AbstractVector, rightenv::AbstractVector)::typeof(x)
    if Defaults.parallelize_derivatives
        @floop WorkStealingEx() for (le, re) in zip(leftenv, rightenv)
            t = ∂C(x, le, re)
            @reduce(y = inplace_add!(nothing, t))
        end
    else
        y = ∂C(x, leftenv[1], rightenv[1])
        for (le, re) in zip(leftenv[2:end], rightenv[2:end])
            VectorInterface.add!(y, ∂C(x, le, re))
        end
    end

    return y
end

function ∂C(x::MPSBondTensor, leftenv::MPSTensor, rightenv::MPSTensor)
    @plansor toret[-1; -2] := leftenv[-1 3; 1] * x[1; 2] * rightenv[2 3; -2]
end

function ∂C(x::MPSBondTensor, leftenv::MPSBondTensor, rightenv::MPSBondTensor)
    @plansor toret[-1; -2] := leftenv[-1; 1] * x[1; 2] * rightenv[2; -2]
end

function ∂C(x::RecursiveVec, leftenv, rightenv)
    return RecursiveVec(circshift(map(t -> ∂C(t...), zip(x.vecs, leftenv, rightenv)), 1))
end

#downproject for approximate
function c_proj(pos, below, envs::FinEnv)
    return ∂C(envs.above.CR[pos], leftenv(envs, pos + 1, below), rightenv(envs, pos, below))
end

function c_proj(row, col, below, envs::PerMPOInfEnv)
    return ∂C(envs.above.CR[row, col], leftenv(envs, row, col + 1, below),
              rightenv(envs, row, col, below))
end

# TODO: rewrite this to not use operator from cache?
function ac_proj(pos, below, envs)
    le = leftenv(envs, pos, below)
    re = rightenv(envs, pos, below)

    return ∂AC(envs.above.AC[pos], envs.opp[pos], le, re)
end
function ac_proj(row, col, below, envs::PerMPOInfEnv)
    return ∂AC(envs.above.AC[row, col], envs.opp[row, col], leftenv(envs, row, col, below),
               rightenv(envs, row, col, below))
end
function ac2_proj(pos, below, envs)
    le = leftenv(envs, pos, below)
    re = rightenv(envs, pos + 1, below)

    return ∂AC2(envs.above.AC[pos] * _transpose_tail(envs.above.AR[pos + 1]), envs.opp[pos],
                envs.opp[pos + 1], le, re)
end
function ac2_proj(row, col, below, envs::PerMPOInfEnv)
    @plansor ac2[-1 -2; -3 -4] := envs.above.AC[row, col][-1 -2; 1] *
                                  envs.above.AR[row, col + 1][1 -4; -3]
    return ∂AC2(ac2, leftenv(envs, row, col + 1, below),
                rightenv(envs, row, col + 1, below))
end

function ∂∂C(pos::Int, mps, opp::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂C(pos, mps, h, e), opp.opps, cache.envs),
                             opp.coeffs)
end

function ∂∂AC(pos::Int, mps, opp::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC(pos, mps, h, e), opp.opps,
                                       cache.envs), opp.coeffs)
end

function ∂∂AC2(pos::Int, mps, opp::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC2(pos, mps, h, e), opp.opps,
                                       cache.envs), opp.coeffs)
end

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
Base.:*(h::Union{AC_EffProj,AC2_EffProj}, v) = h(v);

function (h::AC_EffProj)(x::MPSTensor)
    @plansor v[-1; -2 -3 -4] := h.le[4; -1 -2 5] * h.a1[5 2; 1] * h.re[1; -3 -4 3] *
                                conj(x[4 2; 3])
    @plansor y[-1 -2; -3] := conj(v[1; 2 5 6]) * h.le[-1; 1 2 4] * h.a1[4 -2; 3] *
                             h.re[3; 5 6 -3]
end
function (h::AC2_EffProj)(x::MPOTensor)
    @plansor v[-1; -2 -3 -4] := h.le[6; -1 -2 7] * h.a1[7 4; 5] * h.a2[5 2; 1] *
                                h.re[1; -3 -4 3] * conj(x[6 4; 3 2])
    @plansor y[-1 -2; -3 -4] := conj(v[2; 3 5 6]) * h.le[-1; 2 3 4] * h.a1[4 -2; 7] *
                                h.a2[7 -4; 1] * h.re[1; 5 6 -3]
end

function ∂∂AC(pos::Int, state, opp::ProjectionOperator, env)
    return AC_EffProj(opp.ket.AC[pos], leftenv(env, pos, state), rightenv(env, pos, state))
end
function ∂∂AC2(pos::Int, state, opp::ProjectionOperator, env)
    return AC2_EffProj(opp.ket.AC[pos], opp.ket.AR[pos + 1], leftenv(env, pos, state),
                       rightenv(env, pos + 1, state))
end

# time dependent derivate operators
(h::UntimedOperator{<:DerivativeOperator})(y, args...) = h.f * h.op(y)
(h::TimedOperator{<:DerivativeOperator})(y, t::Number) = h.f(t) * h.op(y)
function (x::LazySum{<:Union{MultipliedOperator{D},D} where {D<:DerivativeOperator}})(y,
                                                                                      t::Number)
    return sum(O -> O(y, t), x)
end
function (x::LazySum{<:Union{MultipliedOperator{D},D} where {D<:DerivativeOperator}})(y)
    return sum(O -> O(y), x)
end
function Base.:*(h::LazySum{<:Union{D,MultipliedOperator{D}} where {D<:DerivativeOperator}},
                 v)
    return h(v)
end

function ∂∂C(pos::Int, mps, opp::MultipliedOperator, cache)
    return MultipliedOperator(∂∂C(pos::Int, mps, opp.op, cache), opp.f)
end

function ∂∂AC(pos::Int, mps, opp::MultipliedOperator, cache)
    return MultipliedOperator(∂∂AC(pos::Int, mps, opp.op, cache), opp.f)
end

function ∂∂AC2(pos::Int, mps, opp::MultipliedOperator, cache)
    return MultipliedOperator(∂∂AC2(pos::Int, mps, opp.op, cache), opp.f)
end

function ∂∂C(pos::Int, mps, opp::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂C(pos, mps, op, openv), opp.ops, cache.envs)
    return LazySum{Union{MPO_∂∂C,MultipliedOperator{<:MPO_∂∂C}}}(suboperators)
end

function ∂∂AC(pos::Int, mps, opp::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂AC(pos, mps, op, openv), opp.ops, cache.envs)
    return LazySum{Union{MPO_∂∂AC,MultipliedOperator{<:MPO_∂∂AC}}}(suboperators)
end

function ∂∂AC2(pos::Int, mps, opp::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂AC2(pos, mps, op, openv), opp.ops, cache.envs)
    return LazySum{Union{MPO_∂∂AC2,MultipliedOperator{<:MPO_∂∂AC2}}}(suboperators)
end
