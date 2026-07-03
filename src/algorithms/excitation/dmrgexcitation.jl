"""
$(TYPEDEF)

Variational optimization algorithm for excitations of finite MPS by minimizing the energy of

```math
H + λᵢ |ψᵢ⟩⟨ψᵢ|
```

# Fields

$(TYPEDFIELDS)

# See also

Used as the `algorithm` argument of [`excitations`](@ref).
"""
@kwdef struct FiniteExcited{A} <: Algorithm
    "optimization algorithm"
    gsalg::A = DMRG()
    "energy penalty for enforcing orthogonality with previous states"
    weight::Float64 = 10.0
end

# Initialize excited state by perturbing the current eigenvector to avoid local minima
function _perturbed_state(ψ::FiniteMPS; atol = 1.0e-2)
    return FiniteMPS(
        map(1:length(ψ)) do i
            A = i == length(ψ) ? ψ.AC[i] : ψ.AL[i]
            noise = randomize!(similar(A))
            return add!(noise, A, 1, atol / norm(noise))
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
