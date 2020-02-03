#because both A and B can again be of type UnionTrunc, this thing can also represent union(alg1,alg2,alg3)
"
    Take the union of 2 truncation algorithms using &
"
struct UnionTrunc{A,B} <: Algorithm
    alg1 :: A
    alg2 :: B
end

function changebonds(state,alg::UnionTrunc)
    state = changebonds(state,alg.alg1);
    state = changebonds(state,alg.alg2);
    return state
end

function changebonds(state,H,alg::UnionTrunc,pars = nothing)
    (state,pars) = changebonds(state,H,alg.alg1,pars)
    (state,pars) = changebonds(state,H,alg.alg2,pars)
    return (state,pars)
end

Base.:&(alg1::Algorithm,alg2::Algorithm) = UnionTrunc(alg1,alg2)
