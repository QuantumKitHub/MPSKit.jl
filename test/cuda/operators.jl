using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, MPOStyle
using TensorKit
using MatrixAlgebraKit
using TensorKit: ℙ, tensormaptype, TensorMapWithStorage
using Adapt, CUDA, cuTENSOR

MPSKit.Defaults.alg_svd() = CUSOLVER_QRIteration()

@testset "CuFiniteMPO" for V in (ℂ^2, U1Space(0 => 1, 1 => 1))
    # start from random operators
    L = 4
    T = ComplexF64

    O₁ = rand(T, V^L, V^L)
    O₂ = rand(T, space(O₁))
    O₃ = rand(real(T), space(O₁))
    
    dO₁ = adapt(CuArray, O₁)
    dO₂ = adapt(CuArray, O₂)
    dO₃ = adapt(CuArray, O₃)

    mpo₁ = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPO(O₁))
    mpo₂ = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPO(O₂))
    mpo₃ = adapt(CuVector{T, CUDA.DeviceMemory}, FiniteMPO(O₃))

    @test isfinite(mpo₁)
    @test isfinite(typeof(mpo₁))
    @test GeometryStyle(typeof(mpo₁)) == FiniteChainStyle()
    @test GeometryStyle(mpo₁) == FiniteChainStyle()
    @test OperatorStyle(typeof(mpo₁)) == MPOStyle()
    @test TensorKit.storagetype(mpo₁) == CuVector{T, CUDA.DeviceMemory}
    @test TensorKit.storagetype(mpo₂) == CuVector{T, CUDA.DeviceMemory}
    @test TensorKit.storagetype(mpo₃) == CuVector{T, CUDA.DeviceMemory}

    @test @constinferred physicalspace(mpo₁) == fill(V, L)
    Vleft = @constinferred left_virtualspace(mpo₁)
    Vright = @constinferred right_virtualspace(mpo₂)
    for i in 1:L
        @test Vleft[i] == left_virtualspace(mpo₁, i)
        @test Vright[i] == right_virtualspace(mpo₁, i)
    end

    TM = TensorMapWithStorage{T, CuVector{T, CUDA.DeviceMemory}}
    #@test convert(TM, mpo₁) ≈ O₁
    #@test convert(TM, -mpo₂) ≈ -O₂
    #@test convert(TM, @constinferred complex(mpo₃)) ≈ complex(O₃)


    # test scalar multiplication
    α = rand(T)
    #@test convert(TM, α * mpo₁) ≈ α * O₁
    #@test convert(TM, mpo₁ * α) ≈ O₁ * α
    @test α * mpo₃ ≈ α * complex(mpo₃) atol = 1.0e-6

    # test addition and multiplication
    #@test convert(TM, mpo₁ + mpo₂) ≈ O₁ + O₂
    #@test convert(TM, mpo₁ + mpo₃) ≈ O₁ + O₃
    #@test convert(TM, mpo₁ * mpo₂) ≈ O₁ * O₂
    #@test convert(TM, mpo₁ * mpo₃) ≈ O₁ * O₃

    # test application to a state
    ψ₁ = adapt(CuArray, rand(T, domain(O₁)))
    #ψ₂ = adapt(CuArray, rand(real(T), domain(O₂))) # not allowed due to cuTENSOR
    mps₁ = adapt(CuArray, FiniteMPS(ψ₁))
    #mps₂ = adapt(CuArray, FiniteMPS(ψ₂))

    @test @constinferred GeometryStyle(mps₁, mpo₁, mps₁) == GeometryStyle(mps₁)

    #@test convert(TM, mpo₁ * mps₁) ≈ dO₁ * ψ₁
    @test mpo₁ * ψ₁ ≈ dO₁ * ψ₁
    #@test convert(TM, mpo₃ * mps₁) ≈ dO₃ * ψ₁
    @test mpo₃ * ψ₁ ≈ dO₃ * ψ₁
    #@test convert(TM, mpo₁ * mps₂) ≈ dO₁ * ψ₂
    #@test mpo₁ * ψ₂ ≈ dO₁ * ψ₂

    @test dot(mps₁, mpo₁, mps₁) ≈ dot(ψ₁, dO₁, ψ₁)
    @test dot(mps₁, mpo₁, mps₁) ≈ dot(mps₁, mpo₁ * mps₁)
    # test conversion to and from mps
    mpomps₁ = convert(FiniteMPS, mpo₁)
    mpompsmpo₁ = convert(FiniteMPO, mpomps₁)

    @test convert(FiniteMPO, mpomps₁) ≈ mpo₁ rtol = 1.0e-6

    @test dot(mpomps₁, mpomps₁) ≈ dot(mpo₁, mpo₁)
end
