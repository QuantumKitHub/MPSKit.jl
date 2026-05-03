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

function make_time_mpo(
        H::InfiniteMPOHamiltonian, dt::Number, alg::WII;
        imaginary_evolution::Bool = false
    )
    T = complex(promote_type(scalartype(H), typeof(dt)))
    δ = imaginary_evolution ? T(-dt) : T(-im * dt)
    exp_alg = Arnoldi(; alg.tol, alg.maxiter)

    O = tmap(parent(parent(H)); scheduler = Defaults.scheduler[]) do W
        return _make_time_mpo(W, δ, exp_alg)
    end
    return InfiniteMPO(PeriodicArray(O))
end

function _make_time_mpo(W::JordanMPOTensor, δ, exp_alg)
    # Allocate output
    Vₗ = left_virtualspace(W)[1:(end - 1)]
    Vᵣ = right_virtualspace(W)[1:(end - 1)]
    P = physicalspace(W)
    O = similar(W, storagetype(W), Vₗ ⊗ P ← P ⊗ Vᵣ)

    # fill onsite first: treated exactly
    expD = exp!(scale(only(W.D), δ))
    O[1, 1, 1, 1] = insertrightunit(insertleftunit(expD, 1), 3)

    # remaining entries:
    D = W[1, 1, 1, end]
    for j in 2:(size(W, 1) - 1), k in 2:(size(W, 4) - 1)
        B = W[j, 1, 1, end]
        C = W[1, 1, 1, k]
        A = W[j, 1, 1, k]
        O[j, 1, 1, k], O[j, 1, 1, 1], O[1, 1, 1, k], _ =
            wii_solve_block!(W, A, B, C, D, δ, exp_alg)
    end

    return O
end

function wii_solve_block!(
        W, A, B, C, D, δ::T, exp_alg
    ) where {T <: Number}
    init = (zero(A), zero(B), zero(C), isometry(storagetype(D), space(D)))
    op = WIIStep(A, B, C, D, δ)
    (yᴬ, yᴮ, yᶜ, yᴰ), convhist = exponentiate(op, one(real(scalartype(W))), init, exp_alg)
    convhist.converged == 0 && @warn "failed to exponentiate $(convhist.normres)"
    return yᴬ, yᴮ, yᶜ, yᴰ
end

# Replaces the `do`-block closure passed to `exponentiate`.
struct WIIStep{HA, HB, HC, HD, T <: Number}
    A::HA  # interior  / propagation block
    B::HB  # right boundary / operator end
    C::HC  # left boundary  / operator start
    D::HD  # onsite / diagonal
    δ::T
end

function (op::WIIStep)((xᴬ, xᴮ, xᶜ, xᴰ))
    δ = op.δ
    sqrtδ = sqrt(δ)

    @plansor yᴰ[-1 -2; -3 -4] := δ * xᴰ[-1 1; -3 -4] *
        op.D[2 3; 1 4] * τ[-2 4; 2 3]

    @plansor yᶜ[-1 -2; -3 -4] := δ * xᶜ[-1 1; -3 -4] *
        op.D[2 3; 1 4] * τ[-2 4; 2 3] +
        sqrtδ * xᴰ[1 2; -3 4] * op.C[-1 -2; 3 -4] * τ[3 4; 1 2]

    @plansor yᴮ[-1 -2; -3 -4] := δ * xᴮ[-1 1; -3 -4] *
        op.D[2 3; 1 4] * τ[-2 4; 2 3] +
        sqrtδ * xᴰ[1 2; -3 4] * op.B[-1 -2; 3 -4] * τ[3 4; 1 2]

    @plansor yᴬ[-1 -2; -3 -4] := (
        δ * xᴬ[-1 1; -3 -4] * op.D[2 3; 1 4] * τ[-2 4; 2 3] +
            xᴰ[1 2; -3 4] * op.A[-1 -2; 3 -4] * τ[3 4; 1 2]
    ) + (
        sqrtδ * xᶜ[1 2; -3 -4] * op.B[-1 -2; 3 4] * τ[3 4; 1 2]
    ) + (sqrtδ * xᴮ[-1 4; -3 3] * op.C[2 -2; 1 -4] * τ[3 4; 1 2])

    return yᴬ, yᴮ, yᶜ, yᴰ
end

# Hack to treat FiniteMPOhamiltonians as Infinite
function make_time_mpo(
        H::FiniteMPOHamiltonian, dt::Number, alg::WII;
        imaginary_evolution::Bool = false
    )
    H′ = copy(parent(H))

    V_left = left_virtualspace(H[1])
    V_left′ = ⊞(V_left, leftunitspace(V_left), leftunitspace(V_left))
    H′[1] = similar(H[1], V_left′ ⊗ space(H[1], 2) ← domain(H[1]))
    for (I, v) in nonzero_pairs(H[1])
        H′[1][I] = v
    end

    V_right = right_virtualspace(H[end])
    V_right′ = ⊞(rightunitspace(V_right), rightunitspace(V_right), V_right)
    H′[end] = similar(H[end], codomain(H[end]) ← space(H[end], 3)' ⊗ V_right′)
    for (I, v) in nonzero_pairs(H[end])
        H′[end][I[1], 1, 1, end] = v
    end

    H′[1][end, 1, 1, end] = H′[1][1, 1, 1, 1]
    H′[end][1, 1, 1, 1] = H′[end][end, 1, 1, end]

    mpo = make_time_mpo(InfiniteMPOHamiltonian(H′), dt, alg; imaginary_evolution)

    # Impose boundary conditions
    mpo_fin = open_boundary_conditions(mpo, length(H))
    return remove_orphans!(mpo_fin; alg.tol)
end
