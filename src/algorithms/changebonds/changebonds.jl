#in-place fallback
function changebonds(state::Union{InfiniteMPS,MPSMultiline},opperator,algorithm,envs::AbstractInfEnv = environments(state,opperator))
    nenvs = deepcopy(envs);
    nstate = nenvs.dependency;

    changebonds!(nstate,opperator,algorithm,nenvs)
end
function changebonds(state::Union{FiniteMPS,MPSComoving},opperator,algorithm,envs::Union{OvlEnv,FinEnv} = environments(state,opperator))
    changebonds!(copy(state),opperator,algorithm,envs)
end
changebonds(state,algorithm) = changebonds!(deepcopy(state),algorithm)

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

function changebonds!(state,H,alg::UnionTrunc,envs=environments(state,H))
    changebonds!(state,H,alg.alg1,envs)
    changebonds!(state,H,alg.alg2,envs)
    return (state,envs)
end

Base.:&(alg1::Algorithm,alg2::Algorithm) = UnionTrunc(alg1,alg2)
