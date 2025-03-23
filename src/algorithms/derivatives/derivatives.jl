# Given a state and it's environments, we can act on it

abstract type DerivativeOperator end

# TODO: do we still need this?
(h::DerivativeOperator)(v, ::Number) = h(v)

# draft operator constructors
# function ∂∂C(pos::Int, mps, operator, envs)
#     return MPO_∂∂C(leftenv(envs, pos + 1, mps), rightenv(envs, pos, mps))
# end
# function ∂∂C(row::Int, col::Int, mps, operator::MultilineMPO, envs)
#     return ∂∂C(col, mps[row], operator[row], envs[row])
# end

# function ∂∂AC(pos::Int, mps, operator, envs)
#     return MPO_∂∂AC(operator[pos], leftenv(envs, pos, mps), rightenv(envs, pos, mps))
# end
function ∂∂AC(row::Int, col::Int, mps, operator::MultilineMPO, envs)
    return ∂∂AC(col, mps[row], operator[row], envs[row])
end
# function ∂∂AC(col::Int, mps, operator::MultilineMPO, envs)
#     return MPO_∂∂AC(operator[:, col], leftenv(envs, col, mps), rightenv(envs, col, mps))
# end

# function ∂∂AC2(pos::Int, mps, operator, envs)
#     return MPO_∂∂AC2(operator[pos], operator[pos + 1], leftenv(envs, pos, mps),
#                      rightenv(envs, pos + 1, mps))
# end
# function ∂∂AC2(col::Int, mps, operator::MultilineMPO, envs)
#     return MPO_∂∂AC2(operator[:, col], operator[:, col + 1], leftenv(envs, col, mps),
#                      rightenv(envs, col + 1, mps))
# end
function ∂∂AC2(row::Int, col::Int, mps, operator::MultilineMPO, envs::MultilineEnvironments)
    return ∂∂AC2(col, mps[row], operator[row], envs[row])
end

# allow calling them with CartesianIndices
∂∂C(pos::CartesianIndex, mps, operator, envs) = ∂∂C(Tuple(pos)..., mps, operator, envs)
∂∂AC(pos::CartesianIndex, mps, operator, envs) = ∂∂AC(Tuple(pos)..., mps, operator, envs)
∂∂AC2(pos::CartesianIndex, mps, operator, envs) = ∂∂AC2(Tuple(pos)..., mps, operator, envs)

# function ∂AC(x::MPSTensor{S}, operator::MPOTensor{S}, leftenv::MPSTensor{S},
#              rightenv::MPSTensor{S})::typeof(x) where {S}
#     @plansor y[-1 -2; -3] := leftenv[-1 5; 4] * x[4 2; 1] * operator[5 -2; 2 3] *
#                              rightenv[1 3; -3]
#     return y isa BlockTensorMap ? only(y) : y
# end
# function ∂AC(x::MPSTensor{S}, operator::Number, leftenv::MPSTensor{S},
#              rightenv::MPSTensor{S})::typeof(x) where {S}
#     @plansor y[-1 -2; -3] := operator * (leftenv[-1 5; 4] * x[4 6; 1] * τ[6 5; 7 -2] *
#                                          rightenv[1 7; -3])
# end
# function ∂AC(x::GenericMPSTensor{S,3}, operator::MPOTensor{S}, leftenv::MPSTensor{S},
#     rightenv::MPSTensor{S})::typeof(x) where {S}
# @plansor y[-1 -2 -3; -4] ≔ leftenv[-1 7; 6] * x[6 4 2; 1] * operator[7 -2; 4 5] *
#                       τ[5 -3; 2 3] * rightenv[1 3; -4]
# end

# mpo multiline
# function ∂AC(x::Vector, opp, leftenv, rightenv)
#     return circshift(map(∂AC, x, opp, leftenv, rightenv), 1)
# end

# function ∂AC(x::MPSTensor, ::Nothing, leftenv, rightenv)
#     return _transpose_front(leftenv * _transpose_tail(x * rightenv))
# end

# function ∂AC2(x::MPOTensor, operator1::MPOTensor, operator2::MPOTensor, leftenv, rightenv)
#     @plansor toret[-1 -2; -3 -4] := leftenv[-1 7; 6] * x[6 5; 1 3] *
#                                     operator1[7 -2; 5 4] *
#                                     operator2[4 -4; 3 2] *
#                                     rightenv[1 2; -3]
#     return toret isa BlockTensorMap ? only(toret) : toret
# end
# function ∂AC2(x::MPOTensor, ::Nothing, ::Nothing, leftenv, rightenv)
#     @plansor y[-1 -2; -3 -4] := x[1 -2; 2 -4] * leftenv[-1; 1] * rightenv[2; -3]
# end
# function ∂AC2(x::AbstractTensorMap{<:Any,<:Any,3,3}, operator1::MPOTensor,
#     operator2::MPOTensor, leftenv::MPSTensor, rightenv::MPSTensor)::typeof(x)
# @plansor y[-1 -2 -3; -4 -5 -6] ≔ leftenv[-1 11; 10] * x[10 8 6; 1 2 4] *
#                            rightenv[1 3; -4] *
#                            operator1[11 -2; 8 9] * τ[9 -3; 6 7] *
#                            operator2[7 -6; 4 5] * τ[5 -5; 2 3]
# end

# function ∂AC2(x::Vector, opp1, opp2, leftenv, rightenv)
#     return circshift(map(∂AC2, x, opp1, opp2, leftenv, rightenv), 1)
# end

# function ∂C(x::MPSBondTensor, leftenv::MPSTensor, rightenv::MPSTensor)
#     @plansor y[-1; -2] := leftenv[-1 3; 1] * x[1; 2] * rightenv[2 3; -2]
#     return y isa BlockTensorMap ? only(y) : y
# end

# function ∂C(x::MPSBondTensor, leftenv::MPSBondTensor, rightenv::MPSBondTensor)
#     @plansor toret[-1; -2] := leftenv[-1; 1] * x[1; 2] * rightenv[2; -2]
# end

# function ∂C(x::Vector, leftenv, rightenv)
#     return circshift(map(t -> ∂C(t...), zip(x, leftenv, rightenv)), 1)
# end

# downproject for approximate
function c_proj(pos::Int, ψ, (operator, ϕ)::Tuple, envs)
    return ∂C(ϕ.C[pos], leftenv(envs, pos + 1, ψ), rightenv(envs, pos, ψ))
end
function c_proj(pos::Int, ψ, ϕ::AbstractMPS, envs)
    return ∂C(ϕ.C[pos], leftenv(envs, pos + 1, ψ), rightenv(envs, pos, ψ))
end
function c_proj(pos::Int, ψ, Oϕs::LazySum, envs)
    return sum(zip(Oϕs.ops, envs.envs)) do x
        return c_proj(pos, ψ, x...)
    end
end
function c_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return c_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

ac_proj(pos::Int, ψ, (O, ϕ)::Tuple, envs) = ∂∂AC(pos, ψ, O, envs) * ϕ.AC[pos]
ac_proj(pos::Int, ψ, ϕ::AbstractMPS, envs) = ∂∂AC(pos, ψ, nothing, envs) * ϕ.AC[pos]

function ac_proj(pos::Int, ψ, Oϕs::LazySum, envs)
    return sum(zip(Oϕs.ops, envs.envs)) do x
        return ac_proj(pos, ψ, x...)
    end
end
function ac_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return ac_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

function ac2_proj(pos::Int, ψ, (O, ϕ)::Tuple, envs)
    AC2 = ϕ.AC[pos] * _transpose_tail(ϕ.AR[pos + 1])
    return ∂∂AC2(pos, ψ, O, envs) * AC2
end
function ac2_proj(pos::Int, ψ, ϕ::AbstractMPS, envs)
    AC2 = ϕ.AC[pos] * _transpose_tail(ϕ.AR[pos + 1])
    return ∂∂AC2(pos, ψ, nothing, envs) * AC2
end
function ac2_proj(row::Int, col::Int, ψ::MultilineMPS, (O, ϕ)::Tuple, envs)
    return ac2_proj(col, ψ[row], (O[row], ϕ[row]), envs[row])
end

function ∂∂C(pos::Int, mps, operator::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂C(pos, mps, h, e), operator.opps,
                                       cache.envs),
                             operator.coeffs)
end

function ∂∂AC(pos::Int, mps, operator::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC(pos, mps, h, e), operator.opps,
                                       cache.envs), operator.coeffs)
end

function ∂∂AC2(pos::Int, mps, operator::LinearCombination, cache)
    return LinearCombination(broadcast((h, e) -> ∂∂AC2(pos, mps, h, e), operator.opps,
                                       cache.envs), operator.coeffs)
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

function ∂∂AC(pos::Int, state, operator::ProjectionOperator, env)
    return AC_EffProj(operator.ket.AC[pos], leftenv(env, pos, state),
                      rightenv(env, pos, state))
end
function ∂∂AC2(pos::Int, state, operator::ProjectionOperator, env)
    return AC2_EffProj(operator.ket.AC[pos], operator.ket.AR[pos + 1],
                       leftenv(env, pos, state),
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

function ∂∂C(pos::Int, mps, operator::MultipliedOperator, cache)
    return MultipliedOperator(∂∂C(pos::Int, mps, operator.op, cache), operator.f)
end

function ∂∂AC(pos::Int, mps, operator::MultipliedOperator, cache)
    return MultipliedOperator(∂∂AC(pos::Int, mps, operator.op, cache), operator.f)
end

function ∂∂AC2(pos::Int, mps, operator::MultipliedOperator, cache)
    return MultipliedOperator(∂∂AC2(pos::Int, mps, operator.op, cache), operator.f)
end

function ∂∂C(pos::Int, mps, operator::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂C(pos, mps, op, openv), operator.ops, cache.envs)
    return LazySum{Union{MPO_∂∂C,MultipliedOperator{<:MPO_∂∂C}}}(suboperators)
end

function ∂∂AC(pos::Int, mps, operator::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂AC(pos, mps, op, openv), operator.ops, cache.envs)
    elT = Union{D,MultipliedOperator{D}} where {D<:Union{MPO_∂∂AC,JordanMPO_∂∂AC}}
    return LazySum{elT}(suboperators)
end

function ∂∂AC2(pos::Int, mps, operator::LazySum, cache::MultipleEnvironments)
    suboperators = map((op, openv) -> ∂∂AC2(pos, mps, op, openv), operator.ops, cache.envs)
    elT = Union{D,MultipliedOperator{D}} where {D<:Union{MPO_∂∂AC2,JordanMPO_∂∂AC2}}
    return LazySum{elT}(suboperators)
end
