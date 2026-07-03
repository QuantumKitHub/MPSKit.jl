"""
    excitations(H, algorithm::QuasiparticleAnsatz, ψ::FiniteQP, [left_environments],
                [right_environments]; num = 1) -> (energies, states)
    excitations(H, algorithm::QuasiparticleAnsatz, ψ::InfiniteQP, [left_environments],
                [right_environments]; num = 1, solver = Defaults.solver) -> (energies, states)
    excitations(H, algorithm::FiniteExcited, ψs::NTuple{<:Any, <:FiniteMPS};
                num = 1, init = copy(first(ψs))) -> (energies, states)
    excitations(H, algorithm::ChepigaAnsatz, ψ::FiniteMPS, [envs];
                num = 1, pos = length(ψ) ÷ 2) -> (energies, states)
    excitations(H, algorithm::ChepigaAnsatz2, ψ::FiniteMPS, [envs];
                num = 1, pos = length(ψ) ÷ 2) -> (energies, states)

Compute the first excited states and their energy gap above a ground state.

<!-- REVIEW: the InfiniteQP signature lists `solver = Defaults.solver`, but `Defaults.solver` does not exist (the module defines `Defaults.linearsolver` and `Defaults.eigsolver`). Confirm the intended default for the `solver` keyword. -->

# Arguments
- `H::AbstractMPO`: operator for which to find the excitations
- `algorithm`: optimization algorithm
- `ψ::QP`: initial quasiparticle guess
- `ψs::NTuple{N, <:FiniteMPS}`: `N` first excited states
- `[left_environments]`: left ground state environment
- `[right_environments]`: right ground state environment

# Keyword Arguments
- `num::Int`: number of excited states to compute
- `solver`: algorithm for the linear solver of the quasiparticle environments
- `init`: initial excited state guess
- `pos`: position of perturbation
"""
function excitations end
