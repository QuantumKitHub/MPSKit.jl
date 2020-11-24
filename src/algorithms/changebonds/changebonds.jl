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

function changebonds(state,H,alg::UnionTrunc,envs=environments(state,H))
    (state,envs) = changebonds(state,H,alg.alg1,envs)
    (state,envs) = changebonds(state,H,alg.alg2,envs)
    return (state,envs)
end

Base.:&(alg1::Algorithm,alg2::Algorithm) = UnionTrunc(alg1,alg2)
