using MPSKit
using MPSKit: _transpose_front, _transpose_tail
using MPSKit: GeometryStyle, InfiniteChainStyle, TransferMatrix
using TensorKit
using TensorKit: ℙ
using Adapt, CUDA, cuTENSOR

@testset "CuMPS ($(sectortype(D)), $elt)" for (D, d, elt) in
    [(ℙ^10, ℙ^2, ComplexF64), (Rep[U₁](1 => 3), Rep[U₁](0 => 1), ComplexF64)]
    tol = Float64(eps(real(elt)) * 100)

    ψ = adapt(CuArray, InfiniteMPS([rand(elt, D * d, D), rand(elt, D * d, D)]; tol))
    @test TensorKit.storagetype(ψ) == CuVector{ComplexF64, CUDA.DeviceMemory}
    @test eltype(ψ) == eltype(typeof(ψ))

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

    L = rand(3:20)
    ψ = adapt(CuArray, FiniteMPS(rand, elt, L, d, D))
    @test TensorKit.storagetype(ψ) == CuVector{ComplexF64, CUDA.DeviceMemory}
    @test eltype(ψ) == eltype(typeof(ψ))
    ovl = dot(ψ, ψ)

    @test ovl ≈ norm(ψ.AC[1])^2

    for i in 1:length(ψ)
        @test ψ.AC[i] ≈ ψ.AL[i] * ψ.C[i]
        @test ψ.AC[i] ≈ _transpose_front(ψ.C[i - 1] * _transpose_tail(ψ.AR[i]))
    end

    @test ComplexF64 == scalartype(ψ)
    ψ = ψ * 3
    @test ovl * 9 ≈ norm(ψ)^2
    ψ = 3 * ψ
    @test ovl * 9 * 9 ≈ norm(ψ)^2

    @test norm(2 * ψ + ψ - 3 * ψ) ≈ 0.0 atol = sqrt(eps(real(ComplexF64)))
end
