println("
------------------------
|   MPO tests (mixed)   |
------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using Adapt

mpo_adapt_Vs = fast_tests ? (ℂ^2,) : (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
Ts = fast_tests ? (Float64,) : (Float64, ComplexF64)
@testset "Adapt" for V in mpo_adapt_Vs
    L = 3
    o = rand(Float32, V^L ← V^L)
    mpo1 = FiniteMPO(o)
    for T in Ts
        mpo2 = @testinferred adapt(Vector{T}, mpo1)
        @test mpo2 isa FiniteMPO
        @test scalartype(mpo2) == T
        @test storagetype(mpo2) == Vector{T}
        @test convert(TensorMap, mpo2) ≈ o
    end

    mpo3 = InfiniteMPO(mpo1[2:2])
    for T in Ts
        mpo4 = @testinferred adapt(Vector{T}, mpo3)
        @test mpo4 isa InfiniteMPO
        @test scalartype(mpo4) == T
        @test storagetype(mpo4) == Vector{T}
        @test dot(mpo3, mpo4) ≈ norm(mpo3)^2 atol = 1.0e-4
    end
end
