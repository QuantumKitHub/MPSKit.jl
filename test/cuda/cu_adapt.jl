using Test
using MPSKit
using TensorKit
using CUDA, cuTENSOR, Adapt
using MPSKit: GeometryStyle, OperatorStyle, MPOStyle, FiniteChainStyle

@testset "Adapting to CuArray" begin
    # start from random operators
    L = 4
    T = ComplexF64

    for V in (ℂ^2, U1Space(0 => 1, 1 => 1))
        O₁ = rand(T, V^L, V^L)
        O₂ = rand(T, space(O₁))
        O₃ = rand(real(T), space(O₁))

        # create MPO and convert it back to see if it is the same
        mpo₁ = adapt(CuArray, FiniteMPO(O₁))
        mpo₂ = adapt(CuArray, FiniteMPO(O₂))
        mpo₃ = adapt(CuArray, FiniteMPO(O₃))

        @test isfinite(mpo₁)
        @test isfinite(typeof(mpo₁))
        @test GeometryStyle(typeof(mpo₁)) == FiniteChainStyle()
        @test GeometryStyle(mpo₁) == FiniteChainStyle()
        @test OperatorStyle(typeof(mpo₁)) == MPOStyle()
        @test TensorKit.storagetype(mpo₁) == CuVector{T, CUDA.DeviceMemory}
        @test TensorKit.storagetype(mpo₂) == CuVector{T, CUDA.DeviceMemory}
        @test TensorKit.storagetype(mpo₃) == CuVector{T, CUDA.DeviceMemory}
    end
end
