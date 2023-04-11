"""
    excitations(H, algorithm::QuasiparticleAnsatz, Ψ::FiniteQP, [left_environments],
                [right_environments]; num=1)
    excitations(H, algorithm::QuasiparticleAnsatz, Ψ::InfiniteQP, [left_environments],
                [right_environments]; num=1, solver=Defaults.solver)
    excitations(H, algorithm::FiniteExcited, Ψs::NTuple{<:Any, <:FiniteMPS};
                num=1, init=copy(first(Ψs)))

Compute the first excited states and their energy gap above a groundstate.

# Arguments
- `H::AbstractMPO`: operator for which to find the excitations
- `algorithm`: optimization algorithm
- `Ψ::QP`: initial quasiparticle guess
- `Ψs::NTuple{N, <:FiniteMPS}`: `N` first excited states
- `[left_environments]`: left groundstate environment
- `[right_environments]`: right groundstate environment

# Keywords
- `num::Int`: number of excited states to compute
- `solver`: algorithm for the linear solver of the quasiparticle environments
- `init`: initial excited state guess
"""
function excitations end