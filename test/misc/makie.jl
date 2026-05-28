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

@testset "reactivity (ComputeGraph)" begin
    ψ   = InfiniteMPS([ℙ^2], [ℙ^5])
    fig = Figure()
    ax  = Axis(fig[1, 1])
    plt = entanglementplot!(ax, ψ)

    n_runs = Ref(0)
    on(plt.attributes[:raw]) do _
        n_runs[] += 1
    end

    plt.mps[] = InfiniteMPS([ℙ^2], [ℙ^6])
    @test n_runs[] == 1                # Tier-1 reran exactly once.

    plt.markersize[] = 20
    @test n_runs[] == 1                # Cosmetic change did NOT re-run Tier 1.

    plt.sector_margin[] = 1 // 5
    @test n_runs[] == 1                # Tier-2-only change did NOT re-run Tier 1.
end
