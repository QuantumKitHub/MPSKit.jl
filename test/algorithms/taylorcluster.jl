println("
---------------------------------
|   TaylorCluster time evolution |
---------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: fuse_mul_mpo
using TensorKit
using TensorKit: ℙ

@testset "TaylorCluster time evolution" begin
    L = 4
    dt = 0.05
    dτ = -im * dt

    @testset "O(1) exact expression" begin
        alg = TaylorCluster(; N = 1, compression = true, extension = true)
        for H in (transverse_field_ising(), heisenberg_XXX())
            # Infinite
            mpo = make_time_mpo(H, dt, alg)
            O = mpo[1]
            I, A, B, C, D = H[1][1], H.A[1], H.B[1], H.C[1], H.D[1]

            @test size(O, 1) == size(D, 1)^2 + (size(D, 1) * size(B, 1))
            @test size(O, 4) == size(D, 4)^2 + (size(C, 4) * size(D, 4))

            O_exact = similar(O)
            O_exact[1, 1, 1, 1] = I + dτ * D + dτ^2 / 2 * fuse_mul_mpo(D, D)
            O_exact[1, 1, 1, 2] = C + dτ * symm_mul_mpo(C, D)
            O_exact[2, 1, 1, 1] = dτ * (B + dτ * symm_mul_mpo(B, D))
            O_exact[2, 1, 1, 2] = A + dτ * (symm_mul_mpo(A, D) + symm_mul_mpo(C, B))

            @test all(isapprox.(parent(O), parent(O_exact)))

            # Finite
            H_fin = open_boundary_conditions(H, L)
            mpo_fin = make_time_mpo(H_fin, dt, alg)
            mpo_fin2 = open_boundary_conditions(mpo, L)
            for i in 1:L
                @test all(isapprox.(parent(mpo_fin[i]), parent(mpo_fin2[i])))
            end
        end
    end

    @testset "O(2) exact expression" begin
        alg = TaylorCluster(; N = 2, compression = true, extension = true)
        for H in (transverse_field_ising(), heisenberg_XXX())
            # Infinite
            mpo = make_time_mpo(H, dt, alg)
            O = mpo[1]
            I, A, B, C, D = H[1][1], H.A[1], H.B[1], H.C[1], H.D[1]

            @test size(O, 1) ==
                size(D, 1)^3 + (size(D, 1)^2 * size(B, 1)) + (size(D, 1) * size(B, 1)^2)
            @test size(O, 4) ==
                size(D, 4)^3 + (size(C, 4) * size(D, 4)^2) + (size(D, 4) * size(C, 4)^2)

            O_exact = similar(O)
            O_exact[1, 1, 1, 1] = I + dτ * D + dτ^2 / 2 * fuse_mul_mpo(D, D) +
                dτ^3 / 6 * fuse_mul_mpo(fuse_mul_mpo(D, D), D)
            O_exact[1, 1, 1, 2] = C + dτ * symm_mul_mpo(C, D) +
                dτ^2 / 2 * symm_mul_mpo(C, D, D)
            O_exact[1, 1, 1, 3] = fuse_mul_mpo(C, C) + dτ * symm_mul_mpo(C, C, D)
            O_exact[2, 1, 1, 1] = dτ *
                (
                B + dτ * symm_mul_mpo(B, D) +
                    dτ^2 / 2 * symm_mul_mpo(B, D, D)
            )
            O_exact[2, 1, 1, 2] = A + dτ * symm_mul_mpo(A, D) +
                dτ^2 / 2 * symm_mul_mpo(A, D, D) +
                dτ * (symm_mul_mpo(C, B) + dτ * symm_mul_mpo(C, B, D))
            O_exact[2, 1, 1, 3] = 2 * (symm_mul_mpo(A, C) + dτ * symm_mul_mpo(A, C, D)) +
                dτ * symm_mul_mpo(C, C, B)
            O_exact[3, 1, 1, 1] = dτ^2 / 2 *
                (fuse_mul_mpo(B, B) + dτ * symm_mul_mpo(B, B, D))
            O_exact[3, 1, 1, 2] = dτ * (symm_mul_mpo(A, B) + dτ * symm_mul_mpo(A, B, D)) +
                dτ^2 / 2 * symm_mul_mpo(B, B, C)
            O_exact[3, 1, 1, 3] = fuse_mul_mpo(A, A) + dτ * symm_mul_mpo(A, A, D) +
                2 * dτ * symm_mul_mpo(A, C, B)

            @test all(isapprox.(parent(O), parent(O_exact)))

            # Finite
            H_fin = open_boundary_conditions(H, L)
            mpo_fin = make_time_mpo(H_fin, dt, alg)
            mpo_fin2 = open_boundary_conditions(mpo, L)
            for i in 1:L
                @test all(isapprox.(parent(mpo_fin[i]), parent(mpo_fin2[i])))
            end
        end
    end

    L = 4
    Hs = [transverse_field_ising(; L = L), heisenberg_XXX(; L = L)]

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
