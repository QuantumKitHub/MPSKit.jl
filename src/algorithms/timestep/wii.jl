"""
$(TYPEDEF)

Generalization of the Euler approximation of the operator exponential for MPOs.

## Fields

$(TYPEDFIELDS)

## References

* [Zaletel et al. Phys. Rev. B 91 (2015)](@cite zaletel2015)
* [Paeckel et al. Ann. of Phys. 411 (2019)](@cite paeckel2019)
"""
@kwdef struct WII <: Algorithm
    "tolerance for convergence criterium"
    tol::Float64 = Defaults.tol
    "maximal number of iterations"
    maxiter::Int = Defaults.maxiter
end

function make_time_mpo(H::InfiniteMPOHamiltonian{T}, dt, alg::WII) where {T}
    WA = H.A
    WB = H.B
    WC = H.C
    WD = H.D

    δ = dt * (-1im)
    Wnew = map(1:length(H)) do i
        for j in 2:(size(H[i], 1) - 1), k in 2:(size(H[i], 4) - 1)
            init_1 = isometry(storagetype(WD[i]), codomain(WD[i]), domain(WD[i]))
            init = [init_1,
                    zero(H[i][1, 1, 1, k]),
                    zero(H[i][j, 1, 1, end]),
                    zero(H[i][j, 1, 1, k])]

            y, convhist = exponentiate(1.0, init,
                                       Arnoldi(; tol=alg.tol, maxiter=alg.maxiter)) do x
                out = similar(x)

                @plansor out[1][-1 -2; -3 -4] := δ * x[1][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]

                @plansor out[2][-1 -2; -3 -4] := δ * x[2][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]
                @plansor out[2][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[1][1 2; -3 4] *
                                                 H[i][1, 1, 1, k][-1 -2; 3 -4] *
                                                 τ[3 4; 1 2]

                @plansor out[3][-1 -2; -3 -4] := δ * x[3][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]
                @plansor out[3][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[1][1 2; -3 4] *
                                                 H[i][j, 1, 1, end][-1 -2; 3 -4] *
                                                 τ[3 4; 1 2]

                @plansor out[4][-1 -2; -3 -4] := δ * x[4][-1 1; -3 -4] *
                                                 H[i][1, 1, 1, end][2 3; 1 4] *
                                                 τ[-2 4; 2 3]
                @plansor out[4][-1 -2; -3 -4] += x[1][1 2; -3 4] *
                                                 H[i][j, 1, 1, k][-1 -2; 3 -4] *
                                                 τ[3 4; 1 2]
                @plansor out[4][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[2][1 2; -3 -4] *
                                                 H[i][j, 1, 1, end][-1 -2; 3 4] *
                                                 τ[3 4; 1 2]
                @plansor out[4][-1 -2; -3 -4] += sqrt(δ) *
                                                 x[3][-1 4; -3 3] *
                                                 H[i][1, 1, 1, k][2 -2; 1 -4] *
                                                 τ[3 4; 1 2]

                return out
            end
            convhist.converged == 0 &&
                @warn "failed to exponentiate $(convhist.normres)"

            WA[i][j - 1, 1, 1, k - 1] = y[4]
            WB[i][j - 1, 1, 1, 1] = y[3]
            WC[i][1, 1, 1, k - 1] = y[2]
            WD[i] = y[1]
        end

        Vₗ = left_virtualspace(H, i)[1:(end - 1)]
        Vᵣ = right_virtualspace(H, i)[1:(end - 1)]
        P = physicalspace(H, i)

        h′ = similar(H[i], Vₗ ⊗ P ← P ⊗ Vᵣ)
        h′[2:end, 1, 1, 2:end] = WA[i]
        h′[2:end, 1, 1, 1] = WB[i]
        h′[1, 1, 1, 2:end] = WC[i]
        h′[1, 1, 1, 1] = WD[i]

        return h′
    end

    return InfiniteMPO(PeriodicArray(Wnew))
end
