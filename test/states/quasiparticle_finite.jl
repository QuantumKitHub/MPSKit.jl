println("
-----------------------------------
|   Quasiparticle tests (finite)  |
-----------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle
using TensorKit
using TensorKit: ℙ

quasiparticle_finite_cases = [
    (force_planar(transverse_field_ising(; L = 10)), ℙ^10, ℙ^2),
    (
        heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1, L = 10), Rep[SU₂](1 => 1, 0 => 3),
        Rep[SU₂](1 => 1),
    ),
]
fast_tests && (quasiparticle_finite_cases = quasiparticle_finite_cases[1:1])

@testset "Quasiparticle state" verbose = true begin
    L = 10
    @testset "Finite" verbose = true for (H, D, d) in quasiparticle_finite_cases
        ψ = FiniteMPS(rand, ComplexF64, L, d, D)
        normalize!(ψ)

        #rand_quasiparticle is a private non-exported function
        ϕ₁ = LeftGaugedQP(rand, ψ)
        ϕ₂ = LeftGaugedQP(rand, ψ)
        @test TensorKit.storagetype(ϕ₁) == TensorKit.storagetype(ψ)
        @test TensorKit.storagetype(typeof(ϕ₁)) == TensorKit.storagetype(ψ)
        @test TensorKit.storagetype(ϕ₂) == TensorKit.storagetype(ψ)
        @test TensorKit.storagetype(typeof(ϕ₂)) == TensorKit.storagetype(ψ)

        @test GeometryStyle(ϕ₁) == FiniteChainStyle()
        @test GeometryStyle(typeof(ϕ₂)) == FiniteChainStyle()

        @test @constinferred physicalspace(ϕ₁) == physicalspace(ψ)
        @test @constinferred left_virtualspace(ϕ₁) == left_virtualspace(ψ)
        @test @constinferred right_virtualspace(ϕ₁) == right_virtualspace(ψ)
        @test TensorKit.sectortype(ϕ₁) == TensorKit.sectortype(ψ)

        @test norm(axpy!(1, ϕ₁, copy(ϕ₂))) ≤ norm(ϕ₁) + norm(ϕ₂)
        @test norm(ϕ₁) * 3 ≈ norm(ϕ₁ * 3)

        normalize!(ϕ₁)

        ϕ₁_f = convert(FiniteMPS, ϕ₁)
        ϕ₂_f = convert(FiniteMPS, ϕ₂)

        @test dot(ϕ₁_f, ϕ₂_f) ≈ dot(ϕ₁, ϕ₂) atol = 1.0e-5
        @test norm(ϕ₁_f) ≈ norm(ϕ₁) atol = 1.0e-5

        ev_f = expectation_value(ϕ₁_f, H) - expectation_value(ψ, H)
        ev_q = dot(ϕ₁, MPSKit.effective_excitation_hamiltonian(H, ϕ₁))
        @test ev_f ≈ ev_q atol = 1.0e-5
    end
end
