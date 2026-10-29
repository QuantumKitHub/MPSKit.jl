using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using Adapt

export mpo_mixed_adapt

function mpo_mixed_adapt(AT; fast = false)
    mpo_adapt_Vs = fast ? (ℂ^2,) : (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
    Ts = fast ? (Float64,) : (Float64, ComplexF64)
    return @testset "Adapt ($AT)" for V in mpo_adapt_Vs
        L = 3
        o = adapt(AT, rand(Float32, V^L ← V^L))
        mpo1 = FiniteMPO(o)
        for T in Ts
            TA = TensorKit.similarstoragetype(AT, T)
            mpo2 = @testinferred adapt(TA, mpo1)
            @test mpo2 isa FiniteMPO
            @test scalartype(mpo2) == T
            @test storagetype(mpo2) == TA
            @test convert(TensorMap, mpo2) ≈ o
        end

        mpo3 = InfiniteMPO(mpo1[2:2])
        for T in Ts
            TA = TensorKit.similarstoragetype(AT, T)
            mpo4 = @testinferred adapt(TA, mpo3)
            @test mpo4 isa InfiniteMPO
            @test scalartype(mpo4) == T
            @test storagetype(mpo4) == TA
            @test dot(mpo3, mpo4) ≈ norm(mpo3)^2 atol = 1.0e-4
        end
    end
end
