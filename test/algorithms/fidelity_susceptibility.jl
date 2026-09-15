println("
--------------------------------
|   Fidelity susceptibility     |
--------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "fidelity susceptibility" begin
    X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
    Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

    H_X = InfiniteMPOHamiltonian(X)
    H_ZZ = InfiniteMPOHamiltonian(Z ⊗ Z)

    hamiltonian(λ) = H_ZZ + λ * H_X
    analytical_susceptibility(λ) = abs(1 / (16 * λ^2 * (λ^2 - 1)))

    for λ in [1.05, 2.0, 4.0]
        H = hamiltonian(λ)
        ψ = InfiniteMPS([ℂ^2], [ℂ^16])
        ψ, envs, = find_groundstate(ψ, H, VUMPS(; maxiter = 100, verbosity = 0))

        numerical_susceptibility = fidelity_susceptibility(ψ, H, [H_X], envs; maxiter = 10)
        @test numerical_susceptibility[1, 1] ≈ analytical_susceptibility(λ) atol = 1.0e-2

        # test if the finite fid sus approximates the analytical one with increasing system size
        fin_en = map([20, 15, 10]) do L
            Hfin = open_boundary_conditions(hamiltonian(λ), L)
            H_Xfin = open_boundary_conditions(H_X, L)
            ψ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^16)
            ψ, envs, = find_groundstate(ψ, Hfin, DMRG(; verbosity = 0))
            numerical_susceptibility = fidelity_susceptibility(
                ψ, Hfin, [H_Xfin], envs; maxiter = 10
            )
            return numerical_susceptibility[1, 1] / L
        end
        @test issorted(abs.(fin_en .- analytical_susceptibility(λ)))
    end
end
