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
using Plots: Plots

@testset "plot tests" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^5])
    _, ax, _ = entanglement_plot(ψ)
    @test ax.title[] == "Entanglement Spectrum"
    _, ax, _ = transfer_plot(ψ, ψ; transferkwargs = (; howmany = 3), axis = (; title = "custom"))
    @test ax.title[] == "custom"

    # mutating plots only style the axis on request
    fig = Figure()
    ax = Axis(fig[1, 1]; title = "mine")
    @test transfer_plot!(ax, ψ) isa Makie.Plot
    @test ax.title[] == "mine"
    ax = Axis(fig[1, 2])
    entanglement_plot!(ax, ψ; use_axis_hints = true)
    @test ax.title[] == "Entanglement Spectrum"
end

@testset "graded plots" begin
    ψ = InfiniteMPS([Z2Space(0 => 1, 1 => 1)], [Z2Space(0 => 4, 1 => 4)])
    @test entanglement_plot(ψ; site = 1, expand_symmetry = true) isa Makie.FigureAxisPlot

    _, ax, p = transfer_plot(ψ)
    @test length(p.plots) == 2
    @test axislegend(ax) isa Legend
    _, _, p = transfer_plot(ψ; sectors = [unit(sectortype(ψ))])
    @test length(p.plots) == 1
    θs = MPSKit.transfer_plot_data(ψ; thetaorigin = -π).θs
    @test all(θ -> -π <= θ < π, reduce(vcat, θs))
end

@testset "coexistence with Plots.jl" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^5])
    @test entanglement_plot(ψ) isa Makie.FigureAxisPlot
    @test (@test_deprecated MPSKit.entanglementplot(ψ)) isa Plots.Plot
end
