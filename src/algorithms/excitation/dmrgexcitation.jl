"""
    FiniteExcited{A} <: Algorithm

Variational optimization algorithm for excitations of finite Matrix Product States by minimizing the energy of ``H - λᵢ |ψᵢ><ψᵢ|``.

# Fields
- `gsalg::A`: optimization algorithm.
- `weight::Float64`: energy penalty for previous states.
"""
@kwdef struct FiniteExcited{A} <: Algorithm
    gsalg::A = DMRG()
    weight::Float64 = 10.0
end

function excitations(H::MPOHamiltonian, alg::FiniteExcited, states::NTuple{N,T};
                     init=FiniteMPS([copy(first(states).AC[i])
                                     for i in 1:length(first(states))]),
                     num=1) where {N,T<:FiniteMPS}
    num == 0 && return (scalartype(T)[], T[])

    super_op = LinearCombination(tuple(H, ProjectionOperator.(states)...),
                                 tuple(1.0, broadcast(x -> alg.weight, states)...))
    envs = environments(init, super_op)
    ne, _ = find_groundstate(init, super_op, alg.gsalg, envs)

    nstates = (states..., ne)
    ens, excis = excitations(H, alg, nstates; init=init, num=num - 1)

    push!(ens, expectation_value(ne, H))
    push!(excis, ne)

    return ens, excis
end
function excitations(H, alg::FiniteExcited, ψ::FiniteMPS; kwargs...)
    return excitations(H, alg, (ψ,); kwargs...)
end
