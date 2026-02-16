println("
---------------------
|   Plot tests       |
---------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Plots
# using CairoMakie

@testset "plot tests" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^5])
    @test transferplot(ψ) isa Plots.Plot
    @test entanglementplot(ψ) isa Plots.Plot
end
