using Test
using MPSKit
using TensorKit
using AMDGPU, Adapt

@testset "Adapting to CuArray" begin
    # start from random operators
    L = 4
    T = ComplexF64

    for V in (ℂ^2, U1Space(0 => 1, 1 => 1))
        O₁ = rand(T, V^L, V^L)
        O₂ = rand(T, space(O₁))
        O₃ = rand(real(T), space(O₁))

        # create MPO and convert it back to see if it is the same
        mpo₁ = adapt(ROCArray, FiniteMPO(O₁))
        mpo₂ = adapt(ROCArray, FiniteMPO(O₂))
        mpo₃ = adapt(ROCArray, FiniteMPO(O₃))

        @test isfinite(mpo₁)
        @test isfinite(typeof(mpo₁))
        @test MPSKit.GeometryStyle(typeof(mpo₁)) == MPSKit.FiniteChainStyle()
        @test MPSKit.GeometryStyle(mpo₁) == MPSKit.FiniteChainStyle()
        @test MPSKit.OperatorStyle(typeof(mpo₁)) == MPSKit.MPOStyle()
        @test TensorKit.storagetype(mpo₁) == ROCVector{T, AMD.Mem.HIPBuffer}
        @test TensorKit.storagetype(mpo₂) == ROCVector{T, AMD.Mem.HIPBuffer}
        @test TensorKit.storagetype(mpo₃) == ROCVector{real(T), AMD.Mem.HIPBuffer}
    end
end
