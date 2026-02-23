println("
------------------------
|   InfiniteMPS tests   |
------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, InfiniteChainStyle, TransferMatrix
using TensorKit
using TensorKit: ℙ

@testset "InfiniteMPS ($(sectortype(D)), $elt)" for (D, d, elt) in
    [(ℙ^10, ℙ^2, ComplexF64), (Rep[U₁](1 => 3), Rep[U₁](0 => 1), ComplexF64)]
    tol = Float64(eps(real(elt)) * 100)

    ψ = InfiniteMPS([rand(elt, D * d, D), rand(elt, D * d, D)]; tol)

    @test !isfinite(typeof(ψ))
    @test !isfinite(ψ)
    @test isfinite(ψ) == isfinite(typeof(ψ))
    @test GeometryStyle(typeof(ψ)) == InfiniteChainStyle()
    @test GeometryStyle(ψ) == InfiniteChainStyle()

    @test eltype(ψ) == eltype(typeof(ψ))

    @test physicalspace(ψ) == fill(d, 2)
    @test all(x -> x ≾ D, left_virtualspace(ψ))
    @test all(x -> x ≾ D, right_virtualspace(ψ))

    for i in 1:length(ψ)
        @plansor difference[-1 -2; -3] := ψ.AL[i][-1 -2; 1] * ψ.C[i][1; -3] -
            ψ.C[i - 1][-1; 1] * ψ.AR[i][1 -2; -3]
        @test norm(difference, Inf) < tol * 10

        @test l_LL(ψ, i) * TransferMatrix(ψ.AL[i], ψ.AL[i]) ≈ l_LL(ψ, i + 1)
        @test l_LR(ψ, i) * TransferMatrix(ψ.AL[i], ψ.AR[i]) ≈ l_LR(ψ, i + 1)
        @test l_RL(ψ, i) * TransferMatrix(ψ.AR[i], ψ.AL[i]) ≈ l_RL(ψ, i + 1)
        @test l_RR(ψ, i) * TransferMatrix(ψ.AR[i], ψ.AR[i]) ≈ l_RR(ψ, i + 1)

        @test TransferMatrix(ψ.AL[i], ψ.AL[i]) * r_LL(ψ, i) ≈ r_LL(ψ, i + 1)
        @test TransferMatrix(ψ.AL[i], ψ.AR[i]) * r_LR(ψ, i) ≈ r_LR(ψ, i + 1)
        @test TransferMatrix(ψ.AR[i], ψ.AL[i]) * r_RL(ψ, i) ≈ r_RL(ψ, i + 1)
        @test TransferMatrix(ψ.AR[i], ψ.AR[i]) * r_RR(ψ, i) ≈ r_RR(ψ, i + 1)
    end
end

@testset "InfiniteMPS copying" begin
    mps1 = InfiniteMPS(rand, ComplexF64, ℂ^2, ℂ^5)
    mps2 = copy(mps1)

    @test mps1 !== mps2

    # elements are equal
    @test mps1.AL[1] == mps2.AL[1]
    @test mps1.AR[1] == mps2.AR[1]
    @test mps1.AC[1] == mps2.AC[1]
    @test mps1.C[1] == mps2.C[1]

    # arrays are distinct
    @test mps1.AL !== mps2.AL
    @test mps1.AR !== mps2.AR
    @test mps1.AC !== mps2.AC
    @test mps1.C !== mps2.C

    # tensors are distinct
    @test mps1.AL[1] !== mps2.AL[1]
    @test mps1.AR[1] !== mps2.AR[1]
    @test mps1.AC[1] !== mps2.AC[1]
    @test mps1.C[1] !== mps2.C[1]
end
