println("
-------------------------------------------
|   TaylorCluster time evolution (finite) |
-------------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "TaylorCluster convergence rate" begin
    L = 4
    Hs = [transverse_field_ising(; L), heisenberg_XXX(; L)]

    Ns = [1, 2, 3]
    dts = [1.0e-2, 1.0e-3]
    for H in Hs
        ψ = FiniteMPS(L, physicalspace(H, 1), ℂ^16)
        for N in Ns
            εs = zeros(ComplexF64, 2)
            for (i, dt) in enumerate(dts)
                ψ₀, _ = find_groundstate(ψ, H, DMRG(; verbosity = 0))
                E₀ = expectation_value(ψ₀, H)

                O = make_time_mpo(H, dt, TaylorCluster(; N = N))

                ψ₁, _ = approximate(ψ₀, (O, ψ₀), DMRG(; verbosity = 0))
                εs[i] = norm(dot(ψ₀, ψ₁) - exp(-im * E₀ * dt))
            end
            @test (log(εs[2]) - log(εs[1])) / (log(dts[2]) - log(dts[1])) ≈ N + 1 atol = 0.1
        end

        for N in Ns
            εs = zeros(ComplexF64, 2)
            for (i, dt) in enumerate(dts)
                ψ₀, _ = find_groundstate(ψ, H, DMRG(; verbosity = 0))
                E₀ = expectation_value(ψ₀, H)

                O = make_time_mpo(H, dt, TaylorCluster(; N = N, compression = true))

                ψ₁, _ = approximate(ψ₀, (O, ψ₀), DMRG(; verbosity = 0))
                εs[i] = norm(dot(ψ₀, ψ₁) - exp(-im * E₀ * dt))
            end
            @test (log(εs[2]) - log(εs[1])) / (log(dts[2]) - log(dts[1])) ≈ N + 1 atol = 0.1
        end
    end
end
