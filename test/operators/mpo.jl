println("
--------------------
|   MPO tests       |
--------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, MPOStyle
using MPSKit: remove_orphans!
using TensorKit
using TensorKit: ℙ
using BlockTensorKit
using BlockTensorKit: SparseBlockTensorMap, nonzero_keys
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
    V = ℂ^2
    T = Float64

    H1 = randn(T, V ⊗ P ← P ⊗ V)
    H = InfiniteMPO([H1])

    @test !isfinite(H)
    @test !isfinite(typeof(H))
    @test GeometryStyle(typeof(H)) == InfiniteChainStyle()
    @test GeometryStyle(H) == InfiniteChainStyle()
    @test OperatorStyle(typeof(H)) == MPOStyle()
    @test OperatorStyle(H) == MPOStyle()

    @test physicalspace(H, 1) == P
    @test left_virtualspace(H, 1) == V
    @test left_virtualspace(H, 4) == V
    @test right_virtualspace(H, 1) == V

    multiH = MultilineMPO([H, H])
    @test physicalspace(multiH, 1, 1) == P
    @test left_virtualspace(multiH, 1, 1) == left_virtualspace(multiH, 2, 1) == V
    @test right_virtualspace(multiH, CartesianIndex(1, 2)) == V
    @test leftunit(multiH) == leftunit(H) == unit(sectortype(P))
    @test rightunit(multiH) == rightunit(H) == unit(sectortype(P))
end

@testset "remove_orphans!" begin
    T = ComplexF64
    P = ℂ^2
    TT = tensormaptype(ComplexSpace, 2, 2, Vector{T})
    randmpotensor() = rand(T, ℂ^1 ⊗ P ← P ⊗ ℂ^1)

    @testset "infinite: dead-end chain" begin
        # 1 -> 1, 1 -> 2 -> 3 -> 4, where channel 4 is a dead end: removing it orphans 3,
        # removing 3 orphans 2, ... so the fixed point takes several passes to reach
        V = SumSpace(fill(ℂ^1, 4)...)
        W = SparseBlockTensorMap{TT}(undef, V ⊗ P ← P ⊗ V)
        for (i, j) in ((1, 1), (1, 2), (2, 3), (3, 4))
            W[i, 1, 1, j] = randmpotensor()
        end
        W₁₁ = copy(W[1, 1, 1, 1])

        mpo = InfiniteMPO([W])
        @test !isfinite(mpo)
        @test remove_orphans!(mpo) === mpo

        # only the (1, 1) self-loop survives
        @test left_virtualspace(mpo, 1) == right_virtualspace(mpo, 1) == SumSpace(ℂ^1)
        @test collect(nonzero_keys(mpo[1])) == [CartesianIndex(1, 1, 1, 1)]
        @test mpo[1][1, 1, 1, 1] ≈ W₁₁
    end

    @testset "infinite: nothing to remove" begin
        # every channel is alive, so this must terminate immediately and change nothing
        H = transverse_field_ising(T; g = 0.7)
        mpo = MPO(map(SparseBlockTensorMap, parent(H)))
        @test !isfinite(mpo)

        spaces₀ = map(space, parent(mpo))
        keys₀ = map(W -> sort!(collect(nonzero_keys(W))), parent(mpo))
        remove_orphans!(mpo)
        @test map(space, parent(mpo)) == spaces₀
        @test map(W -> sort!(collect(nonzero_keys(W))), parent(mpo)) == keys₀
    end

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
