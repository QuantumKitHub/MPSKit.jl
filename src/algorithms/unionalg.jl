"
    Take the union of 2 algorithms using &
"
struct UnionAlg{A,B} <: Algorithm
    alg1::A
    alg2::B
end

Base.:&(alg1::Algorithm, alg2::Algorithm) = UnionAlg(alg1, alg2)

function changebonds(state, alg::UnionAlg)
    state = changebonds(state, alg.alg1)
    state = changebonds(state, alg.alg2)
    return state
end

function changebonds(state, H, alg::UnionAlg, envs=environments(state, H))
    (state, envs) = changebonds(state, H, alg.alg1, envs)
    (state, envs) = changebonds(state, H, alg.alg2, envs)
    return (state, envs)
end

function find_groundstate(state, H, alg::UnionAlg, envs=environments(state, H))
    (state, envs) = find_groundstate(state, H, alg.alg1, envs)
    return (state, envs, delta) = find_groundstate(state, H, alg.alg2, envs)
end
