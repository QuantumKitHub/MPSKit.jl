println("
----------------------------
|   InfiniteMPO tests      |
----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, InfiniteChainStyle, OperatorStyle, MPOStyle
using MPSKit: remove_orphans!
using TensorKit
using BlockTensorKit
using BlockTensorKit: SparseBlockTensorMap, nonzero_keys

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
end

@testset "DenseMPO" for ham in (transverse_field_ising(), heisenberg_XXX(; spin = 1))
    pspace = physicalspace(ham, 1)
    ou = rightunitspace(pspace)

    ψ = InfiniteMPS([pspace], [ou ⊕ pspace])

    W = MPSKit.DenseMPO(make_time_mpo(ham, 1im * 0.5, WII()))

    @test GeometryStyle(ψ, W) == GeometryStyle(ψ)
    @test W * (W * ψ) ≈ (W * W) * ψ atol = 1.0e-2 # TODO: there is a normalization issue here
end
