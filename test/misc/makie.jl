println("
-----------------------------------
|     Plot tests with Makie.jl    |
-----------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using CairoMakie

@testset "plot tests" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^5])
    @test transferplot(ψ) isa CairoMakie.Plot
    @test entanglementplot(ψ) isa CairoMakie.Plot
end
