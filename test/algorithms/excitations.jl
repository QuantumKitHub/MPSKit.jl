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
    @testset "finite" begin # contains an infinite-finite comparison
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
            energies_QP, ϕs = @testinferred excitations(H, QuasiparticleAnsatz(), ψ, envs)
            @test variance(ϕs[1], H) < 1.0e-6

            # find energy with normal dmrg
            for gsalg in (
                    DMRG(; verbosity, tol = 1.0e-6),
                    DMRG2(; verbosity, tol = 1.0e-6, trunc = trunctol(; atol = 1.0e-4)),
                )
                energies_dm, _ = @testinferred excitations(H, FiniteExcited(; gsalg), ψ; num = 3)
                @test energies_dm[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
                @test issorted(real.(energies_dm))
            end

            # find energy with Chepiga ansatz
            energies_ch, _ = @testinferred excitations(H, ChepigaAnsatz(), ψ, envs)
            @test energies_ch[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
            energies_ch2, _ = @testinferred excitations(H, ChepigaAnsatz2(), ψ, envs)
            @test energies_ch2[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
            return energies_QP[1]
        end

        @test issorted(abs.(fin_en .- inf_en))
    end
end
