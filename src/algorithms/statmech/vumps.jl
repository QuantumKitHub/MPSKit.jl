"""
    leading_boundary(ψ, opp, alg, envs=environments(ψ, opp))

Approximate the leading eigenvector for opp.
"""
function leading_boundary(ψ::InfiniteMPS, H, alg, envs = environments(ψ, H))
    st, pr, de = leading_boundary(convert(MultilineMPS, ψ), Multiline([H]), alg, envs)
    return convert(InfiniteMPS, st), pr, de
end

function leading_boundary(
        mps::MultilineMPS, operator, alg::VUMPS, envs = environments(mps, operator)
    )
    return dominant_eigsolve(operator, mps, alg, envs; which = :LM)
end
