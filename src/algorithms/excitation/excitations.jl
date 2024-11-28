"""
    excitations(H, algorithm::QuasiparticleAnsatz, ψ::FiniteQP, [left_environments],
                [right_environments]; num=1)
    excitations(H, algorithm::QuasiparticleAnsatz, ψ::InfiniteQP, [left_environments],
                [right_environments]; num=1, solver=Defaults.solver)
    excitations(H, algorithm::FiniteExcited, ψs::NTuple{<:Any, <:FiniteMPS};
                num=1, init=copy(first(ψs)))
    excitations(H, algorithm::ChepigaAnsatz, ψ::FiniteMPS, [envs];
                num=1, pos=length(ψ)÷2)
    excitations(H, algorithm::ChepigaAnsatz2, ψ::FiniteMPS, [envs];
                num=1, pos=length(ψ)÷2)

Compute the first excited states and their energy gap above a groundstate.

# Arguments
- `H::AbstractMPO`: operator for which to find the excitations
- `algorithm`: optimization algorithm
- `ψ::QP`: initial quasiparticle guess
- `ψs::NTuple{N, <:FiniteMPS}`: `N` first excited states
- `[left_environments]`: left groundstate environment
- `[right_environments]`: right groundstate environment

# Keywords
- `num::Int`: number of excited states to compute
- `solver`: algorithm for the linear solver of the quasiparticle environments
- `init`: initial excited state guess
- `pos`: position of perturbation
"""
function excitations end
