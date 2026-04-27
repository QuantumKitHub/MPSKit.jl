"""
    fidelity_susceptibility(state::Union{FiniteMPS,InfiniteMPS}, H₀::T,
                            Vs::AbstractVector{T}, [henvs=environments(state, H₀)];
                            maxiter=Defaults.maxiter,
                            tol=Defaults.tol) where {T<:MPOHamiltonian}

Computes the fidelity susceptibility of a the ground state `state` of a base Hamiltonian
`H₀` with respect to a set of perturbing Hamiltonians `Vs`. Each of the perturbing
Hamiltonians can be interpreted as corresponding to a tuning parameter ``aᵢ`` in a 'total'
Hamiltonian ``H = H₀ + ∑ᵢ aᵢ Vᵢ``.

Returns a matrix containing the overlaps of the elementary excitations on top of `state`
corresponding to each of the perturbing Hamiltonians.
"""
function fidelity_susceptibility(
        state::Union{FiniteMPS, InfiniteMPS}, H₀::T,
        Vs::AbstractVector{T}, henvs = environments(state, H₀);
        maxiter = Defaults.maxiter, tol = Defaults.tol
    ) where {T <: MPOHamiltonian}
    tangent_vecs = map(Vs) do V
        venvs = environments(state, V)

        Tos = LeftGaugedQP(rand, state)
        for (i, ac) in enumerate(state.AC)
            temp = AC_projection(i, state, V, state, venvs)
            help = fill_data!(similar(ac, auxiliaryspace(Tos)), one)
            @plansor Tos[i][-1 -2; -3 -4] := temp[-1 -2; -4] * help[-3]
        end

        vec, convhist = linsolve(Tos, Tos, GMRES(; maxiter = maxiter, tol = tol)) do x
            return effective_excitation_hamiltonian(H₀, x, environments(x, H₀, henvs))
        end
        convhist.converged == 0 && @warn "failed to converge: normres = $(convhist.normres)"

        return vec
    end

    return map(product(tangent_vecs, tangent_vecs)) do (a, b)
        return dot(a, b)
    end
end
