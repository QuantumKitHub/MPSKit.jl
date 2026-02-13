println("
---------------------
|   Styles tests     |
---------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using MPSKit: GeometryStyle, FiniteChainStyle, InfiniteChainStyle, OperatorStyle, MPOStyle, HamiltonianStyle

@testset "Styles" begin
    @test_throws MethodError OperatorStyle(42)
    @test_throws MethodError OperatorStyle(Float64)
    @test_throws MethodError GeometryStyle("abc")
    @test_throws MethodError GeometryStyle(UInt8)

    @test OperatorStyle(MPO) == MPOStyle()
    @test OperatorStyle(InfiniteMPO) == MPOStyle()
    @test OperatorStyle(HamiltonianStyle()) == HamiltonianStyle()
    @test @constinferred OperatorStyle(MPO, InfiniteMPO, MPO) == MPOStyle()
    @test_throws ErrorException OperatorStyle(MPO, HamiltonianStyle())

    @test GeometryStyle(FiniteMPOHamiltonian) == FiniteChainStyle()
    @test GeometryStyle(InfiniteMPS) == InfiniteChainStyle()
    @test GeometryStyle(FiniteMPS) == FiniteChainStyle()
    @test GeometryStyle(FiniteMPO) == FiniteChainStyle()
    @test GeometryStyle(FiniteMPOHamiltonian) == FiniteChainStyle()
    @test GeometryStyle(InfiniteMPO) == InfiniteChainStyle()
    @test GeometryStyle(InfiniteMPOHamiltonian) == InfiniteChainStyle()

    @test GeometryStyle(GeometryStyle(FiniteMPS)) == GeometryStyle(FiniteMPS)
    @test GeometryStyle(FiniteMPS, FiniteMPO) == FiniteChainStyle()
    @test_throws ErrorException GeometryStyle(FiniteMPS, InfiniteMPO)
    @test @constinferred GeometryStyle(InfiniteMPS, InfiniteMPO, InfiniteMPS) == InfiniteChainStyle()
    @test_throws ErrorException GeometryStyle(FiniteMPS, FiniteMPO, InfiniteMPS)
end
