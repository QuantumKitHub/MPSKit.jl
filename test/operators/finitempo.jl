println("
--------------------------
|   FiniteMPO tests      |
--------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, OperatorStyle, MPOStyle
using MPSKit: remove_orphans!
using TensorKit
using TensorKit: ℙ
using BlockTensorKit
using BlockTensorKit: SparseBlockTensorMap, nonzero_keys

@testset "FiniteMPO" begin
    # start from random operators
    L = 4
    T = ComplexF64

    for V in (ℂ^2, U1Space(0 => 1, 1 => 1))
        O₁ = rand(T, V^L, V^L)
        O₂ = rand(T, space(O₁))
        O₃ = rand(real(T), space(O₁))

        # create MPO and convert it back to see if it is the same
        mpo₁ = FiniteMPO(O₁) # type-unstable for now!
        mpo₂ = FiniteMPO(O₂)
        mpo₃ = FiniteMPO(O₃)

        @test isfinite(mpo₁)
        @test isfinite(typeof(mpo₁))
        @test GeometryStyle(typeof(mpo₁)) == FiniteChainStyle()
        @test GeometryStyle(mpo₁) == FiniteChainStyle()
        @test OperatorStyle(typeof(mpo₁)) == MPOStyle()

        @test @constinferred physicalspace(mpo₁) == fill(V, L)
        Vleft = @constinferred left_virtualspace(mpo₁)
        Vright = @constinferred right_virtualspace(mpo₂)
        for i in 1:L
            @test Vleft[i] == left_virtualspace(mpo₁, i)
            @test Vright[i] == right_virtualspace(mpo₁, i)
        end

        @test convert(TensorMap, mpo₁) ≈ O₁
        @test convert(TensorMap, -mpo₂) ≈ -O₂
        @test convert(TensorMap, @constinferred complex(mpo₃)) ≈ complex(O₃)

        # test scalar multiplication
        α = rand(T)
        @test convert(TensorMap, α * mpo₁) ≈ α * O₁
        @test convert(TensorMap, mpo₁ * α) ≈ O₁ * α
        @test α * mpo₃ ≈ α * complex(mpo₃) atol = 1.0e-6

        # test addition and multiplication
        @test convert(TensorMap, mpo₁ + mpo₂) ≈ O₁ + O₂
        @test convert(TensorMap, mpo₁ + mpo₃) ≈ O₁ + O₃
        @test convert(TensorMap, mpo₁ * mpo₂) ≈ O₁ * O₂
        @test convert(TensorMap, mpo₁ * mpo₃) ≈ O₁ * O₃

        # test application to a state
        ψ₁ = rand(T, domain(O₁))
        ψ₂ = rand(real(T), domain(O₂))
        mps₁ = FiniteMPS(ψ₁)
        mps₂ = FiniteMPS(ψ₂)

        @test @constinferred GeometryStyle(mps₁, mpo₁, mps₁) == GeometryStyle(mps₁)

        @test convert(TensorMap, mpo₁ * mps₁) ≈ O₁ * ψ₁
        @test mpo₁ * ψ₁ ≈ O₁ * ψ₁
        @test convert(TensorMap, mpo₃ * mps₁) ≈ O₃ * ψ₁
        @test mpo₃ * ψ₁ ≈ O₃ * ψ₁
        @test convert(TensorMap, mpo₁ * mps₂) ≈ O₁ * ψ₂
        @test mpo₁ * ψ₂ ≈ O₁ * ψ₂

        @test dot(mps₁, mpo₁, mps₁) ≈ dot(ψ₁, O₁, ψ₁)
        @test dot(mps₁, mpo₁, mps₁) ≈ dot(mps₁, mpo₁ * mps₁)
        # test conversion to and from mps
        mpomps₁ = convert(FiniteMPS, mpo₁)
        mpompsmpo₁ = convert(FiniteMPO, mpomps₁)

        @test convert(FiniteMPO, mpomps₁) ≈ mpo₁ rtol = 1.0e-6

        @test dot(mpomps₁, mpomps₁) ≈ dot(mpo₁, mpo₁)
    end
end

@testset "remove_orphans!" begin
    T = ComplexF64
    P = ℂ^2
    TT = tensormaptype(ComplexSpace, 2, 2, Vector{T})
    randmpotensor() = rand(T, ℂ^1 ⊗ P ← P ⊗ ℂ^1)

    @testset "finite: dead-end chain" begin
        # channel 2 is never entered on site 1, so the (2, 2) block on site 2 and the
        # (2, 1) block on site 3 are orphaned
        V₁ = SumSpace(ℂ^1)
        V₂ = SumSpace(ℂ^1, ℂ^1)
        Ws = [
            SparseBlockTensorMap{TT}(undef, V₁ ⊗ P ← P ⊗ V₂),
            SparseBlockTensorMap{TT}(undef, V₂ ⊗ P ← P ⊗ V₂),
            SparseBlockTensorMap{TT}(undef, V₂ ⊗ P ← P ⊗ V₁),
        ]
        Ws[1][1, 1, 1, 1] = randmpotensor()
        Ws[2][1, 1, 1, 1] = randmpotensor()
        Ws[2][2, 1, 1, 2] = randmpotensor()
        Ws[3][1, 1, 1, 1] = randmpotensor()
        Ws[3][2, 1, 1, 1] = randmpotensor()

        mpo = MPO(Ws)
        @test isfinite(mpo)
        O = convert(TensorMap, mpo)
        remove_orphans!(mpo)

        @test left_virtualspace(mpo) == right_virtualspace(mpo) == fill(SumSpace(ℂ^1), 3)
        @test convert(TensorMap, mpo) ≈ O
    end
end
