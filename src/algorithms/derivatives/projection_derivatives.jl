struct Projection_∂∂ACN{L,O<:Tuple,R} <: DerivativeOperator
    leftenv::L
    As::O
    rightenv::R
end

const Projection_∂∂AC{L,O,R} = Projection_∂∂ACN{L,Tuple{O},R}
Projection_∂∂AC(GL, A, GR) = Projection_∂∂ACN(GL, (A,), GR)

const Projection_∂∂AC2{L,O₁,O₂,R} = Projection_∂∂ACN{L,Tuple{O₁,O₂},R}
Projection_∂∂AC2(GL, A1, A2, GR) = Projection_∂∂ACN(GL, (A1, A2), GR)

struct AC_EffProj{A,L} <: DerivativeOperator
    a1::A
    le::L
    re::L
end
struct AC2_EffProj{A,L} <: DerivativeOperator
    a1::A
    a2::A
    le::L
    re::L
end

# Constructors
# ------------
function ∂∂AC(pos::Int, state, operator::ProjectionOperator, env)
    return Projection_∂∂AC(leftenv(env, pos, state), operator.ket.AC[pos],
                           rightenv(env, pos, state))
end
function ∂∂AC2(pos::Int, state, operator::ProjectionOperator, env)
    return Projection_∂∂AC2(leftenv(env, pos, state), operator.ket.AC[pos],
                            operator.ket.AR[pos + 1],
                            rightenv(env, pos + 1, state))
    return AC2_EffProj(operator.ket.AC[pos], operator.ket.AR[pos + 1],
                       leftenv(env, pos, state),
                       rightenv(env, pos + 1, state))
end

# Actions
# -------
function (h::Projection_∂∂AC)(x::MPSTensor)
    @plansor v[-1; -2 -3 -4] := h.leftenv[4; -1 -2 5] * h.As[1][5 2; 1] *
                                h.rightenv[1; -3 -4 3] * conj(x[4 2; 3])
    @plansor y[-1 -2; -3] := conj(v[1; 2 5 6]) * h.leftenv[-1; 1 2 4] * h.As[1][4 -2; 3] *
                             h.rightenv[3; 5 6 -3]
end
function (h::Projection_∂∂AC2)(x::MPOTensor)
    @plansor v[-1; -2 -3 -4] := h.leftenv[6; -1 -2 7] * h.As[1][7 4; 5] * h.As[2][5 2; 1] *
                                h.rightenv[1; -3 -4 3] * conj(x[6 4; 3 2])
    @plansor y[-1 -2; -3 -4] := conj(v[2; 3 5 6]) * h.leftenv[-1; 2 3 4] *
                                h.As[1][4 -2; 7] * h.As[2][7 -4; 1] * h.rightenv[1; 5 6 -3]
end
