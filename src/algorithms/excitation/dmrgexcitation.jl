"""
$(TYPEDEF)

Variational optimization algorithm for excitations of finite MPS by minimizing the energy of

```math
H + λᵢ |ψᵢ⟩⟨ψᵢ|
```

## Fields

$(TYPEDFIELDS)
"""
@kwdef struct FiniteExcited{A} <: Algorithm
    "optimization algorithm"
    gsalg::A = DMRG()
    "energy penalty for enforcing orthogonality with previous states"
    weight::Float64 = 10.0
end

# Default initial guess for the next excited state: the previous state's center tensors with a
# small random perturbation added per site. Without the perturbation the guess can already be
# (near) an eigenvector of the shifted operator, so the Galerkin convergence measure is tiny on
# the first sweep and the optimizer returns immediately instead of climbing to the excited state.
# The perturbation preserves the physical/virtual spaces (hence the symmetry sector) and bond
# dimensions; `FiniteMPS` re-gauges and normalizes the result.
function _perturbed_state(ψ::FiniteMPS; ϵ = 1.0e-2)
    return FiniteMPS(
        map(1:length(ψ)) do i
            A = ψ.AC[i]
            noise = randomize!(similar(A))
            return A + (ϵ / norm(noise)) * noise
        end
    )
end

function excitations(
        H::FiniteMPOHamiltonian, alg::FiniteExcited,
        states::Tuple{T, Vararg{T}};
        init = _perturbed_state(first(states)), num = 1
    ) where {T <: FiniteMPS}
    num == 0 && return (scalartype(T)[], T[])

    super_op = LinearCombination(
        tuple(H, ProjectionOperator.(states)...),
        tuple(1.0, broadcast(x -> alg.weight, states)...)
    )
    envs = environments(init, super_op, init)
    ne, _ = find_groundstate(init, super_op, alg.gsalg, envs)

    nstates = (states..., ne)
    ens, excis = excitations(H, alg, nstates; init = init, num = num - 1)

    pushfirst!(ens, expectation_value(ne, H))
    pushfirst!(excis, ne)

    return ens, excis
end
function excitations(H, alg::FiniteExcited, ψ::FiniteMPS; kwargs...)
    return excitations(H, alg, (ψ,); kwargs...)
end
