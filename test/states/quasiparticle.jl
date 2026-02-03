println("
-----------------------------
|   Quasiparticle tests      |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle
using TensorKit
using TensorKit: ℙ

@testset "Quasiparticle state" verbose = true begin
    L = 10
    @testset "Finite" verbose = true for (H, D, d) in
        [
            (force_planar(transverse_field_ising(; L)), ℙ^10, ℙ^2),
            (heisenberg_XXX(SU2Irrep; spin = 1, L), Rep[SU₂](1 => 1, 0 => 3), Rep[SU₂](1 => 1)),
        ]
        ψ = FiniteMPS(rand, ComplexF64, L, d, D)
        normalize!(ψ)

        #rand_quasiparticle is a private non-exported function
        ϕ₁ = LeftGaugedQP(rand, ψ)
        ϕ₂ = LeftGaugedQP(rand, ψ)

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

    @testset "Infinite" for (th, D, d) in
        [
            (force_planar(transverse_field_ising()), ℙ^10, ℙ^2),
            (
                heisenberg_XXX(SU2Irrep; spin = 1), Rep[SU₂](1 => 3, 0 => 2),
                Rep[SU₂](1 => 1),
            ),
        ]
        period = rand(1:4)
        ψ = InfiniteMPS(fill(d, period), fill(D, period))

        @test eltype(ψ) == eltype(typeof(ψ))

        #rand_quasiparticle is a private non-exported function
        ϕ₁ = LeftGaugedQP(rand, ψ)
        ϕ₂ = LeftGaugedQP(rand, ψ)

        @test GeometryStyle(ϕ₁) == InfiniteChainStyle()
        @test GeometryStyle(typeof(ϕ₂)) == InfiniteChainStyle()

        @test @constinferred physicalspace(ϕ₁) == physicalspace(ψ)
        @test @constinferred left_virtualspace(ϕ₁) == left_virtualspace(ψ)
        @test @constinferred right_virtualspace(ϕ₁) == right_virtualspace(ψ)
        for i in 1:period
            @test physicalspace(ψ, i) == physicalspace(ϕ₁, i)
            @test left_virtualspace(ψ, i) == left_virtualspace(ϕ₁, i)
            @test right_virtualspace(ψ, i) == right_virtualspace(ϕ₁, i)
        end

        @test norm(axpy!(1, ϕ₁, copy(ϕ₂))) ≤ norm(ϕ₁) + norm(ϕ₂)
        @test norm(ϕ₁) * 3 ≈ norm(ϕ₁ * 3)

        @test dot(
            ϕ₁,
            convert(LeftGaugedQP, convert(RightGaugedQP, ϕ₁))
        ) ≈
            dot(ϕ₁, ϕ₁) atol = 1.0e-10
    end
end
