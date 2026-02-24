println("
------------------------
|   MultilineMPS tests  |
------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, InfiniteChainStyle, TransferMatrix
using TensorKit
using TensorKit: ℙ

@testset "MultilineMPS ($(sectortype(D)), $elt)" for (D, d, elt) in
    [(ℙ^10, ℙ^2, ComplexF64), (Rep[U₁](1 => 3), Rep[U₁](0 => 1), ComplexF32)]
    tol = Float64(eps(real(elt)) * 100)
    ψ = MultilineMPS(
        [
            rand(elt, D * d, D) rand(elt, D * d, D)
            rand(elt, D * d, D) rand(elt, D * d, D)
        ]; tol
    )

    @test GeometryStyle(typeof(ψ)) == InfiniteChainStyle()
    @test GeometryStyle(ψ) == InfiniteChainStyle()

    @test !isfinite(typeof(ψ))

    @test physicalspace(ψ) == fill(d, 2, 2)
    @test all(x -> x ≾ D, left_virtualspace(ψ))
    @test all(x -> x ≾ D, right_virtualspace(ψ))

    for i in 1:size(ψ, 1), j in 1:size(ψ, 2)
        @plansor difference[-1 -2; -3] := ψ.AL[i, j][-1 -2; 1] * ψ.C[i, j][1; -3] -
            ψ.C[i, j - 1][-1; 1] * ψ.AR[i, j][1 -2; -3]
        @test norm(difference, Inf) < tol * 10

        @test l_LL(ψ, i, j) * TransferMatrix(ψ.AL[i, j], ψ.AL[i, j]) ≈ l_LL(ψ, i, j + 1)
        @test l_LR(ψ, i, j) * TransferMatrix(ψ.AL[i, j], ψ.AR[i, j]) ≈ l_LR(ψ, i, j + 1)
        @test l_RL(ψ, i, j) * TransferMatrix(ψ.AR[i, j], ψ.AL[i, j]) ≈ l_RL(ψ, i, j + 1)
        @test l_RR(ψ, i, j) * TransferMatrix(ψ.AR[i, j], ψ.AR[i, j]) ≈ l_RR(ψ, i, j + 1)

        @test TransferMatrix(ψ.AL[i, j], ψ.AL[i, j]) * r_LL(ψ, i, j) ≈ r_LL(ψ, i, j + 1)
        @test TransferMatrix(ψ.AL[i, j], ψ.AR[i, j]) * r_LR(ψ, i, j) ≈ r_LR(ψ, i, j + 1)
        @test TransferMatrix(ψ.AR[i, j], ψ.AL[i, j]) * r_RL(ψ, i, j) ≈ r_RL(ψ, i, j + 1)
        @test TransferMatrix(ψ.AR[i, j], ψ.AR[i, j]) * r_RR(ψ, i, j) ≈ r_RR(ψ, i, j + 1)
    end
end
