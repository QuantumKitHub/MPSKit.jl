module MPSKitPlotsExt

using RecipesBase
using MPSKit, TensorKit
import MPSKit: entanglementplot, transferplot

@userplot EntanglementPlot

@recipe function f(
        h::EntanglementPlot; site = 0, expand_symmetry = false, sortby = maximum,
        sector_margin = 1 // 10, sector_formatter = string
    )
    Base.depwarn(
        "`entanglementplot` (Plots.jl) is deprecated, use `entanglement_plot` (Makie.jl) instead.",
        :entanglementplot
    )
    mps = h.args[1]
    (; sectors, xs, ys, dropped) = MPSKit.entanglement_plot_data(
        mps, site; expand_symmetry, sortby, sector_margin
    )
    dropped && @warn "Entanglement spectrum contains vanishing Schmidt values. These are omitted from the plot."

    for (x, y, sector) in zip(xs, ys, sectors)
        @series begin
            seriestype := :scatter
            label := sector_formatter(sector)
            return x, y
        end
    end

    title --> "Entanglement Spectrum"
    legend --> false
    grid --> :xy
    widen --> true

    xguide --> "χ = $(dim(MPSKit._firstspace(mps.C[site])))"
    xticks --> (1:length(sectors), sector_formatter.(sectors))
    xtickfonthalign --> :center
    xtick_direction --> :out
    xrotation --> 45
    xlims --> (1, length(sectors) + 1)

    ylims --> (-Inf, 1 + 1.0e-1)
    yscale --> :log10
    label := nothing

    return []
end

@userplot TransferPlot

@recipe function f(
        h::TransferPlot; sectors = nothing, transferkwargs = (;), thetaorigin = 0,
        sector_formatter = string
    )
    Base.depwarn(
        "`transferplot` (Plots.jl) is deprecated, use `transfer_plot` (Makie.jl) instead.",
        :transferplot
    )
    data = MPSKit.transfer_plot_data(h.args...; sectors, transferkwargs, thetaorigin)

    for (θ, r, sector) in zip(data.θs, data.rs, data.sectors)
        @series begin
            yguide --> "r"
            ylims --> (-Inf, 1.05)

            xguide --> "θ"
            xlims --> (thetaorigin, thetaorigin + 2pi)
            xticks --> range(0, 2pi; length = 7)
            xformatter --> x -> "$(rationalize(x / π, tol = 0.05))π"
            xwiden --> true
            seriestype := :scatter
            markershape --> :auto
            label := sector_formatter(sector)
            return θ, r
        end
    end

    title --> "Transfer Spectrum"
    legend --> false
    grid --> :xy
    framestyle --> :zerolines

    return nothing
end

end
