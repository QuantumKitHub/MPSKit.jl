#=
I don't know if I should rescale by system size / unit cell
=#
function fidelity_susceptibility(state::Union{FiniteMPS,InfiniteMPS}, H₀::T,
                                 Vs::AbstractVector{T}, henvs=environments(state, H₀);
                                 maxiter=Defaults.maxiter,
                                 tol=Defaults.tol) where {T<:MPOHamiltonian}
    tangent_vecs = map(Vs) do V
        venvs = environments(state, V)

        Tos = LeftGaugedQP(rand, state)
        for (i, ac) in enumerate(state.AC)
            temp = ∂∂AC(i, state, H₀, venvs) * ac
            help = fill_data!(similar(ac, utilleg(Tos)), one)
            @plansor Tos[i][-1 -2; -3 -4] := temp[-1 -2; -4] * help[-3]
        end

        (vec, convhist) = linsolve(Tos, Tos, GMRES(; maxiter=maxiter, tol=tol)) do x
            return effective_excitation_hamiltonian(H₀, x, environments(x, H₀, henvs))
        end
        convhist.converged == 0 && @warn "failed to converge: normres = $(convhist.normres)"

        return vec
    end

    map(product(tangent_vecs, tangent_vecs)) do (a, b)
        return dot(a, b)
    end
end
