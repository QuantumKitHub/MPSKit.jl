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
    return AC_EffProj(operator.ket.AC[pos], leftenv(env, pos, state),
                      rightenv(env, pos, state))
end
function ∂∂AC2(pos::Int, state, operator::ProjectionOperator, env)
    return AC2_EffProj(operator.ket.AC[pos], operator.ket.AR[pos + 1],
                       leftenv(env, pos, state),
                       rightenv(env, pos + 1, state))
end

# Actions
# -------
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
end
