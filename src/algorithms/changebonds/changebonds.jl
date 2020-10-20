#in-place fallback
changebonds(state,opperator,algorithm,args...) = changebonds!(copy(state),opperator,algorithm,args...)
changebonds(state,algorithm) = changebonds!(copy(state),algorithm)

"
    Take the union of 2 truncation algorithms using &
"
struct UnionTrunc{A,B} <: Algorithm
    alg1 :: A
    alg2 :: B
end

function changebonds!(state,alg::UnionTrunc)
    changebonds!(state,alg.alg1);
    changebonds!(state,alg.alg2);
    return state
end

function changebonds!(state,H,alg::UnionTrunc,pars = params(state,H))
    changebonds!(state,H,alg.alg1,pars)
    changebonds!(state,H,alg.alg2,pars)
    return (state,pars)
end

Base.:&(alg1::Algorithm,alg2::Algorithm) = UnionTrunc(alg1,alg2)
