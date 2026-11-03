println("
-----------------------------------
|     Plot tests with Plots.jl    |
-----------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Plots

@testset "plot tests" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^5])
    @test (@test_deprecated transferplot(ψ)) isa Plots.Plot
    @test (@test_deprecated entanglementplot(ψ)) isa Plots.Plot
end
