println("
-----------------------------------------------
|   MPOHamiltonian tests (mixed/constructors) |
-----------------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, HamiltonianStyle
using TensorKit
using TensorKit: ℙ
using Adapt

@testset "MPOHamiltonian constructors" begin
    P = ℂ^2
    T = Float64

    H1 = randn(T, P ← P)
    H1 += H1'
    D = FiniteMPO(H1)[1]

    H2 = randn(T, P^2 ← P^2)
    H2 += H2'
    C, B = FiniteMPO(H2)[1:2]

    Elt = Union{Missing, typeof(D), scalartype(D)}
    Wmid = Elt[1.0 C D; 0.0 0.0 B; 0.0 0.0 1.0]
    Wleft = Wmid[1:1, :]
    Wright = Wmid[:, end:end]

    # Finite
    Ws = [Wleft, Wmid, Wmid, Wright]
    H = FiniteMPOHamiltonian(
        fill(P, 4), [(i,) => H1 for i in 1:4]..., [(i, i + 1) => H2 for i in 1:3]...
    )
    H′ = FiniteMPOHamiltonian(Ws)
    @test H ≈ H′

    H′ = FiniteMPOHamiltonian(map(Base.Fix1(collect, Any), Ws)) # without type info
    @test H ≈ H′

    @test isfinite(H)
    @test isfinite(typeof(H))
    @test GeometryStyle(typeof(H)) == FiniteChainStyle()
    @test GeometryStyle(H) == FiniteChainStyle()
    @test OperatorStyle(typeof(H)) == HamiltonianStyle()
    @test OperatorStyle(H) == HamiltonianStyle()
    @test OperatorStyle(H, H′) == OperatorStyle(H)
    @test TensorKit.storagetype(H) == Vector{T}
    @test TensorKit.storagetype(typeof(H)) == Vector{T}

    # Infinite
    Ws = [Wmid]
    H = InfiniteMPOHamiltonian(
        fill(P, 1), [(i,) => H1 for i in 1:1]..., [(i, i + 1) => H2 for i in 1:1]...
    )
    H′ = InfiniteMPOHamiltonian(Ws)
    @test all(parent(H) .≈ parent(H′))

    H′ = InfiniteMPOHamiltonian(map(Base.Fix1(collect, Any), Ws)) # without type info
    @test all(parent(H) .≈ parent(H′))

    @test !isfinite(H)
    @test !isfinite(typeof(H))
    @test GeometryStyle(typeof(H)) == InfiniteChainStyle()
    @test GeometryStyle(H) == InfiniteChainStyle()
    @test OperatorStyle(typeof(H)) == HamiltonianStyle()
    @test OperatorStyle(H) == HamiltonianStyle()
    @test TensorKit.storagetype(H′) == Vector{T}
    @test TensorKit.storagetype(typeof(H′)) == Vector{T}
end

@testset "Adapt" for V in (ℂ^2, U1Space(-1 => 1, 0 => 1, 1 => 1))
    h = rand(Float32, V^2 ← V^2)
    h += h'

    L = 4
    H1 = FiniteMPOHamiltonian(
        fill(V, L),
        ((i, i + 1) => h for i in 1:(L - 1))...,
        ((i, i + 2) => h for i in 1:(L - 2))...,
        ((i, i + 3) => h for i in 1:(L - 3))...,
    )
    mps1 = FiniteMPS(physicalspace(H1), oneunit(V))

    for T in (Float64, ComplexF64)
        H2 = if VERSION <= v"1.12"
            adapt(Vector{T}, H1)
        else
            @testinferred adapt(Vector{T}, H1)
        end
        @test H2 isa FiniteMPOHamiltonian
        @test scalartype(H2) == T
        @test storagetype(H2) == Vector{T}
        @test expectation_value(mps1, H1) ≈ expectation_value(mps1, H2)
    end

    H3 = InfiniteMPOHamiltonian(fill(V, L), (1, 2) => h, (1, 3) => h, (1, 4) => h)
    mps2 = InfiniteMPS(physicalspace(H3), [oneunit(V)])
    for T in (Float64, ComplexF64)
        H4 = if VERSION <= v"1.12"
            # this is type unstable for LTS for some reason
            adapt(Vector{T}, H3)
        else
            @testinferred adapt(Vector{T}, H3)
        end
        @test H4 isa InfiniteMPOHamiltonian
        @test scalartype(H4) == T
        @test storagetype(H4) == Vector{T}
        @test storagetype(typeof(H4)) == Vector{T}
        @test expectation_value(mps2, H3) ≈ expectation_value(mps2, H4)
    end
end
