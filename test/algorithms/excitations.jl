println("
-----------------------------
|   Excitations tests       |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

verbosity_conv = 1

@testset "excitations" verbose = true begin
    @testset "infinite (ham)" begin
        H = repeat(force_planar(heisenberg_XXX()), 2)
        ψ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^48, ℙ^48])
        ψ, envs, _ = find_groundstate(
            ψ, H; maxiter = 400, verbosity = verbosity_conv, tol = 1.0e-10
        )
        energies, ϕs = @inferred excitations(
            H, QuasiparticleAnsatz(), Float64(pi), ψ, envs
        )
        @test energies[1] ≈ 0.41047925 atol = 1.0e-4
        @test variance(ϕs[1], H) < 1.0e-8
    end
    @testset "infinite sector convention" begin
        g = 4
        H = repeat(transverse_field_ising(Z2Irrep; g = g, L = Inf), 2)
        V = Z2Space(0 => 24, 1 => 24)
        ψ = InfiniteMPS(physicalspace(H), [V, V])
        ψ, envs = find_groundstate(ψ, H, VUMPS(; tol = 1.0e-10, maxiter = 400))

        # testing istrivial and istopological
        momentum = 0
        exc0, qp0 = excitations(H, QuasiparticleAnsatz(), momentum, ψ; sector = Z2Irrep(0))
        exc1, qp1 = excitations(H, QuasiparticleAnsatz(), momentum, ψ; sector = Z2Irrep(1))
        @test isapprox(first(exc1), abs(2 * (g - 1)); atol = 1.0e-6) # charged excitation lower in energy
        @test first(exc0) > 2 * first(exc1)
        @test variance(qp1[1], H) < 1.0e-8
    end
    @testset "infinite (mpo)" begin
        H = repeat(sixvertex(), 2)
        ψ = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
        ψ, envs, _ = leading_boundary(
            ψ, H, VUMPS(; maxiter = 400, verbosity = verbosity_conv, tol = 1.0e-10)
        )
        energies, ϕs = @inferred excitations(
            H, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ψ, envs; verbosity = 0
        )
        @test abs(energies[1]) > abs(energies[2]) # has a minimum at pi/2
    end

    @testset "finite" begin
        verbosity = verbosity_conv
        H_inf = force_planar(transverse_field_ising())
        ψ_inf = InfiniteMPS([ℙ^2], [ℙ^10])
        ψ_inf, envs, _ = find_groundstate(ψ_inf, H_inf; maxiter = 400, verbosity, tol = 1.0e-9)
        energies, ϕs = @inferred excitations(H_inf, QuasiparticleAnsatz(), 0.0, ψ_inf, envs)
        inf_en = energies[1]

        fin_en = map([20, 10]) do len
            H = force_planar(transverse_field_ising(; L = len))
            ψ = FiniteMPS(rand, ComplexF64, len, ℙ^2, ℙ^10)
            ψ, envs, = find_groundstate(ψ, H; verbosity)

            # find energy with quasiparticle ansatz
            energies_QP, ϕs = @inferred excitations(H, QuasiparticleAnsatz(), ψ, envs)
            @test variance(ϕs[1], H) < 1.0e-6

            # find energy with normal dmrg
            for gsalg in (
                    DMRG(; verbosity, tol = 1.0e-6),
                    DMRG2(; verbosity, tol = 1.0e-6, trscheme = trunctol(; atol = 1.0e-4)),
                )
                energies_dm, _ = @inferred excitations(H, FiniteExcited(; gsalg), ψ)
                @test energies_dm[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
            end

            # find energy with Chepiga ansatz
            energies_ch, _ = @inferred excitations(H, ChepigaAnsatz(), ψ, envs)
            @test energies_ch[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
            energies_ch2, _ = @inferred excitations(H, ChepigaAnsatz2(), ψ, envs)
            @test energies_ch2[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
            return energies_QP[1]
        end

        @test issorted(abs.(fin_en .- inf_en))
    end
end
