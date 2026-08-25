println("
-------------------------------------
|   Excitations tests (infinite)    |
-------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

verbosity_conv = 1

@testset "infinite (ham)" begin
    H = repeat(force_planar(heisenberg_XXX()), 2)
    ψ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^48, ℙ^48])
    ψ, envs, _ = find_groundstate(
        ψ, H; maxiter = 400, verbosity = verbosity_conv, tol = 1.0e-10
    )
    energies, ϕs = @testinferred excitations(
        H, QuasiparticleAnsatz(), Float64(pi), ψ, envs
    )
    @test energies[1] ≈ 0.41047925 atol = 1.0e-4
    @test variance(ϕs[1], H) < 1.0e-8
end

@testset "infinite sector convention" begin
    g = 4
    H = repeat(transverse_field_ising(ComplexF64, Z2Irrep; g), 2)
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
    energies, ϕs = @testinferred excitations(
        H, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ψ, envs; verbosity = 0
    )
    @test abs(energies[1]) > abs(energies[2]) # has a minimum at pi/2
end

@testset "infinite (mpo) - quasiparticle input" begin
    # regression test: handing an `InfiniteQP` to an `InfiniteMPO`, rather than a
    # momentum, used to build the effective Hamiltonian out of the (still undefined)
    # `H_eff` instead of `H`, so every call through this entry point threw an
    # `UndefVarError`. The momentum entry point is unaffected, as it converts to a
    # `MultilineMPO` first. The classical Ising transfer matrix is used here (rather than
    # the six-vertex model above) because its quasiparticle eigenproblem is well enough
    # conditioned to compare the two entry points at tight tolerance.
    H = classical_ising(; β = 0.4)
    ψ = InfiniteMPS([ℂ^2], [ℂ^12])
    ψ, envs, = leading_boundary(ψ, H, VUMPS(; maxiter = 400, verbosity = 0, tol = 1.0e-12))

    alg = QuasiparticleAnsatz(; tol = 1.0e-10)
    for momentum in (0.0, Float64(pi / 2))
        ϕ₀ = MPSKit.LeftGaugedQP(rand, ψ, ψ; momentum)
        Es, ϕs = excitations(H, alg, ϕ₀, envs, envs; num = 1)

        # same spectrum as the momentum entry point, which routes through MultilineMPO
        Es_momentum, = excitations(H, alg, momentum, ψ, envs; num = 1)
        @test Es[1] ≈ Es_momentum[1] atol = 1.0e-8

        # the operator really is `H`: the result is an eigenpair of the corresponding
        # effective excitation Hamiltonian
        E = MPSKit.effective_excitation_renormalization_energy(H, ϕs[1], envs, envs)
        H_eff = MPSKit.EffectiveExcitationHamiltonian(H, envs, envs, E)
        @test norm(H_eff(ϕs[1], alg.alg_environments) - Es[1] * ϕs[1]) < 1.0e-8
    end
end
