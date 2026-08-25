println("
-------------------------------------
|   Quasiparticle tests (infinite)  |
-------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, InfiniteChainStyle
using TensorKit
using TensorKit: ℙ

quasiparticle_infinite_cases = [
    (force_planar(transverse_field_ising()), ℙ^10, ℙ^2),
    (
        heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1), Rep[SU₂](1 => 3, 0 => 2),
        Rep[SU₂](1 => 1),
    ),
]
fast_tests && (quasiparticle_infinite_cases = quasiparticle_infinite_cases[1:1])

@testset "Quasiparticle state" verbose = true begin
    @testset "Infinite" for (th, D, d) in quasiparticle_infinite_cases
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
