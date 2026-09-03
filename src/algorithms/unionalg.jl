"""
$(TYPEDEF)

Algorithm wrapper representing the sequential application of two algorithms, as produced by
`alg1 & alg2`.

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`find_groundstate`](@ref) and [`changebonds`](@ref).
"""
struct UnionAlg{A, B} <: Algorithm
    "first algorithm"
    alg1::A
    "second algorithm"
    alg2::B
end

Base.:&(alg1::Algorithm, alg2::Algorithm) = UnionAlg(alg1, alg2)

function changebonds(state, alg::UnionAlg)
    state = changebonds(state, alg.alg1)
    state = changebonds(state, alg.alg2)
    return state
end

function changebonds(state, H, alg::UnionAlg, envs = environments(state, H, state))
    state, envs = changebonds(state, H, alg.alg1, envs)
    state, envs = changebonds(state, H, alg.alg2, envs)
    return state, envs
end

function find_groundstate(state, H, alg::UnionAlg, envs = environments(state, H, state))
    state, envs = find_groundstate(state, H, alg.alg1, envs)
    return find_groundstate(state, H, alg.alg2, envs)
end

function approximate(state, toapprox, alg::UnionAlg, envs = enviroments(state, toapprox...))
    state, envs = approximate(state, toapprox, alg.alg1, envs)
    return approximate(state, toapprox, alg.alg2, envs)
end
