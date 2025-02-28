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

function make_time_mpo(H::InfiniteMPOHamiltonian, dt::Number, alg::WII)
    WA = H.A
    WB = H.B
    WC = H.C
    WD = H.D

    δ = dt * (-1im)
    exp_alg = Arnoldi(; alg.tol, alg.maxiter)

    Wnew = map(1:length(H)) do i
        for j in 2:(size(H[i], 1) - 1), k in 2:(size(H[i], 4) - 1)
            init = (isometry(storagetype(WD[i]), codomain(WD[i]), domain(WD[i])),
                    zero(H[i][1, 1, 1, k]),
                    zero(H[i][j, 1, 1, end]),
                    zero(H[i][j, 1, 1, k]))

            y, convhist = exponentiate(1.0, init, exp_alg) do (x1, x2, x3, x4)
                @plansor y1[-1 -2; -3 -4] := δ * x1[-1 1; -3 -4] *
                                             H[i][1, 1, 1, end][2 3; 1 4] *
                                             τ[-2 4; 2 3]

                @plansor y2[-1 -2; -3 -4] := δ * x2[-1 1; -3 -4] *
                                             H[i][1, 1, 1, end][2 3; 1 4] *
                                             τ[-2 4; 2 3] +
                                             sqrt(δ) *
                                             x1[1 2; -3 4] *
                                             H[i][1, 1, 1, k][-1 -2; 3 -4] *
                                             τ[3 4; 1 2]

                @plansor y3[-1 -2; -3 -4] := δ * x3[-1 1; -3 -4] *
                                             H[i][1, 1, 1, end][2 3; 1 4] *
                                             τ[-2 4; 2 3] +
                                             sqrt(δ) *
                                             x1[1 2; -3 4] *
                                             H[i][j, 1, 1, end][-1 -2; 3 -4] *
                                             τ[3 4; 1 2]

                @plansor y4[-1 -2; -3 -4] := (δ * x4[-1 1; -3 -4] *
                                              H[i][1, 1, 1, end][2 3; 1 4] *
                                              τ[-2 4; 2 3] +
                                              x1[1 2; -3 4] *
                                              H[i][j, 1, 1, k][-1 -2; 3 -4] *
                                              τ[3 4; 1 2]) +
                                             (sqrt(δ) *
                                              x2[1 2; -3 -4] *
                                              H[i][j, 1, 1, end][-1 -2; 3 4] *
                                              τ[3 4; 1 2]) +
                                             (sqrt(δ) *
                                              x3[-1 4; -3 3] *
                                              H[i][1, 1, 1, k][2 -2; 1 -4] *
                                              τ[3 4; 1 2])

                return y1, y2, y3, y4
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

# Hack to treat FiniteMPOhamiltonians as Infinite
function make_time_mpo(H::FiniteMPOHamiltonian, dt::Number, alg::WII)
    H′ = copy(parent(H))

    V_left = left_virtualspace(H[1])
    V_left′ = BlockTensorKit.oplus(V_left, oneunit(V_left), oneunit(V_left))
    H′[1] = similar(H[1], V_left′ ⊗ space(H[1], 2) ← domain(H[1]))
    for (I, v) in nonzero_pairs(H[1])
        H′[1][I] = v
    end

    V_right = right_virtualspace(H[end])
    V_right′ = BlockTensorKit.oplus(oneunit(V_right), oneunit(V_right), V_right)
    H′[end] = similar(H[end], codomain(H[end]) ← space(H[end], 3)' ⊗ V_right′)
    for (I, v) in nonzero_pairs(H[end])
        H′[end][I[1], 1, 1, end] = v
    end

    H′[1][end, 1, 1, end] = H′[1][1, 1, 1, 1]
    H′[end][1, 1, 1, 1] = H′[end][end, 1, 1, end]

    mpo = make_time_mpo(InfiniteMPOHamiltonian(H′), dt, alg; tol)

    # Impose boundary conditions
    mpo[1] = mpo[1][1, :, :, :]
    mpo[end] = mpo[end][:, :, :, 1]
    return remove_orphans!(mpo; tol)
end
