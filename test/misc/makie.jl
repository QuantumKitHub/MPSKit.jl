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
    @test transferplot(ψ) isa Makie.FigureAxisPlot
    @test transferplot(ψ, ψ) isa Makie.FigureAxisPlot
    @test transferplot(ψ; transferkwargs = (; howmany = 3)) isa Makie.FigureAxisPlot

    @test entanglementplot(ψ) isa Makie.FigureAxisPlot

    # mutating plots
    fig = Figure()
    ax = Axis(fig[1, 1])
    @test entanglementplot!(ax, ψ) isa Makie.Plot
    @test transferplot!(Axis(fig[1, 2]), ψ) isa Makie.Plot
    @test transferplot!(Axis(fig[1, 3]), ψ, ψ) isa Makie.Plot

    # no target -> default to current axis
    fig1 = Figure()
    Axis(fig1[1, 1])
    @test entanglementplot!(ψ) isa Makie.Plot
    @test transferplot!(ψ) isa Makie.Plot
    @test transferplot!(ψ, ψ) isa Makie.Plot

    # detect plotkwargs in targeted axis
    fig3 = Figure()
    ax3 = Axis(fig3[1, 1])
    entanglementplot!(ax3, ψ; plotkwargs = (; title = "custom"))
    @test ax3.title[] == "custom"

    # plotting into a non-current axis must not style the current one
    fig2 = Figure()
    target = Axis(fig2[1, 1])
    current = Axis(fig2[1, 2]) # created last, so this is the current axis
    entanglementplot!(target, ψ)
    @test target.title[] != ""
    @test current.title[] == ""
end

@testset "graded plots" begin
    ψ = InfiniteMPS([Z2Space(0 => 1, 1 => 1)], [Z2Space(0 => 4, 1 => 4)])

    @test entanglementplot(ψ) isa Makie.FigureAxisPlot
    @test entanglementplot(ψ; site = 1) isa Makie.FigureAxisPlot

    @test transferplot(ψ) isa Makie.FigureAxisPlot
    @test transferplot(ψ, ψ) isa Makie.FigureAxisPlot

    # restrict sectors
    triv = unit(sectortype(ψ))
    @test transferplot(ψ; sectors = [triv]) isa Makie.FigureAxisPlot
end
