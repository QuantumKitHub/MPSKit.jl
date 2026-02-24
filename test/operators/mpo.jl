println("
--------------------
|   MPO tests       |
--------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, MPOStyle
using TensorKit
using TensorKit: ℙ
using Adapt

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

@testset "InfiniteMPO" begin
    P = ℂ^2
    T = Float64

    H1 = randn(T, P ← P)
    H1 += H1'
    H = InfiniteMPO([H1])

    @test !isfinite(H)
    @test !isfinite(typeof(H))
    @test GeometryStyle(typeof(H)) == InfiniteChainStyle()
    @test GeometryStyle(H) == InfiniteChainStyle()
    @test OperatorStyle(typeof(H)) == MPOStyle()
    @test OperatorStyle(H) == MPOStyle()
end

@testset "Adapt" for V in (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
    L = 3
    o = rand(Float32, V^L ← V^L)
    mpo1 = FiniteMPO(o)
    for T in (Float64, ComplexF64)
        mpo2 = @testinferred adapt(Vector{T}, mpo1)
        @test mpo2 isa FiniteMPO
        @test scalartype(mpo2) == T
        @test storagetype(mpo2) == Vector{T}
        @test convert(TensorMap, mpo2) ≈ o
    end

    mpo3 = InfiniteMPO(mpo1[2:2])
    for T in (Float64, ComplexF64)
        mpo4 = @testinferred adapt(Vector{T}, mpo3)
        @test mpo4 isa InfiniteMPO
        @test scalartype(mpo4) == T
        @test storagetype(mpo4) == Vector{T}
        @test dot(mpo3, mpo4) ≈ norm(mpo3)^2 atol = 1.0e-4
    end
end
