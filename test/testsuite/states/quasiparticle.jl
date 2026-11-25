using ..TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle
using TensorKit
using TensorKit: ℙ
using Adapt

# `@constinferred` evaluates a helper at global scope, which is not allowed inside a function
# body, so the testsuite functions use `@testinferred` (none of the calls have constant arguments)

export states_quasiparticle

function quasiparticle_finite_cases(AT; fast = false)
    cases = [
        (adapt(AT, force_planar(transverse_field_ising(; L = 10))), ℙ^10, ℙ^2),
        (
            adapt(AT, heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1, L = 10)),
            Rep[SU₂](1 => 1, 0 => 3), Rep[SU₂](1 => 1),
        ),
    ]
    fast && (cases = cases[1:1])
    return cases
end

function quasiparticle_infinite_cases(AT; fast = false)
    cases = [
        (adapt(AT, force_planar(transverse_field_ising())), ℙ^10, ℙ^2),
        (
            adapt(AT, heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1)),
            Rep[SU₂](1 => 3, 0 => 2), Rep[SU₂](1 => 1),
        ),
    ]
    fast && (cases = cases[1:1])
    return cases
end

# LeftGaugedQP/RightGaugedQP have no Adapt rules; they are built from an adapted state instead
function states_quasiparticle(AT; fast = false)
    return @testset "Quasiparticle state ($AT)" verbose = true begin
        L = 10
        @testset "Finite" verbose = true for (H, D, d) in quasiparticle_finite_cases(AT; fast)
            ψ = adapt(AT, FiniteMPS(rand, ComplexF64, L, d, D))
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

            @test @testinferred physicalspace(ϕ₁) == physicalspace(ψ)
            @test @testinferred left_virtualspace(ϕ₁) == left_virtualspace(ψ)
            @test @testinferred right_virtualspace(ϕ₁) == right_virtualspace(ψ)
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

        @testset "Infinite" for (th, D, d) in quasiparticle_infinite_cases(AT; fast)
            period = rand(1:4)
            ψ = adapt(AT, InfiniteMPS(fill(d, period), fill(D, period)))

            @test eltype(ψ) == eltype(typeof(ψ))

            #rand_quasiparticle is a private non-exported function
            ϕ₁ = LeftGaugedQP(rand, ψ)
            ϕ₂ = LeftGaugedQP(rand, ψ)

            @test GeometryStyle(ϕ₁) == InfiniteChainStyle()
            @test GeometryStyle(typeof(ϕ₂)) == InfiniteChainStyle()

            @test @testinferred physicalspace(ϕ₁) == physicalspace(ψ)
            @test @testinferred left_virtualspace(ϕ₁) == left_virtualspace(ψ)
            @test @testinferred right_virtualspace(ϕ₁) == right_virtualspace(ψ)
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
end
