module MPSKitMakieExt

using Makie
using MPSKit: MPSKit, AbstractMPS, _firstspace
using TensorKit: dim
import MPSKit: entanglement_plot, entanglement_plot!, transfer_plot, transfer_plot!

# Makie lowercases the type name to get the function name, hence the underscores
@recipe Entanglement_Plot (mps::AbstractMPS,) begin
    site = 0
    expand_symmetry = false
    sortby = maximum
    sector_margin = 1 // 10
    sector_formatter = string
    colors = Makie.wong_colors()
    markersize = @inherit markersize
end

function entanglement_data(mps, site, expand_symmetry, sortby, sector_margin)
    data = MPSKit.entanglement_plot_data(mps, site; expand_symmetry, sortby, sector_margin)
    return (; data.sectors, points = map((x, y) -> Point2d.(x, y), data.xs, data.ys), data.dropped)
end

function Makie.plot!(p::Entanglement_Plot)
    data = lift(entanglement_data, p, p.mps, p.site, p.expand_symmetry, p.sortby, p.sector_margin)
    data[].dropped && @warn "Entanglement spectrum contains vanishing Schmidt values. These are omitted from the plot."
    add_sector_scatters!(p, data)
    return p
end

function Makie.preferred_axis_attributes(::Type{Axis}, p::Entanglement_Plot)
    mps, site = p.mps[], p.site[]
    sectors, points = entanglement_data(mps, site, p.expand_symmetry[], p.sortby[], p.sector_margin[])
    n = length(sectors)
    smallest = minimum(last, Iterators.flatten(points); init = 1.0)
    # limits must precede `yscale`: an empty axis has limits (0, 10), invalid on a log scale
    return (;
        limits = ((1, n + 1), (exp10(floor(log10(smallest))), 1.1)),
        title = "Entanglement Spectrum",
        xlabel = "χ = $(dim(_firstspace(mps.C[site])))",
        ylabel = "Schmidt values",
        xticks = ((1:n) .+ 0.5, p.sector_formatter[].(sectors)),
        xticklabelrotation = π / 4,
        xgridvisible = false,
        yscale = log10,
    )
end

Makie.get_plots(p::Entanglement_Plot) = p.plots

@recipe Transfer_Plot (above::AbstractMPS, below::AbstractMPS) begin
    sectors = nothing
    transferkwargs = (;)
    thetaorigin = 0.0
    sector_formatter = string
    colors = Makie.wong_colors()
    markersize = @inherit markersize
end

Makie.convert_arguments(::Type{<:Transfer_Plot}, above::AbstractMPS) = (above, above)

function transfer_data(above, below, sectors, transferkwargs, thetaorigin)
    # Makie stores NamedTuple attributes as nested `Attributes` of observables
    transferkwargs = (; (k => to_value(v) for (k, v) in pairs(transferkwargs))...)
    data = MPSKit.transfer_plot_data(above, below; sectors, transferkwargs, thetaorigin)
    return (; data.sectors, points = map((θ, r) -> Point2d.(θ, r), data.θs, data.rs))
end

function Makie.plot!(p::Transfer_Plot)
    data = lift(transfer_data, p, p.above, p.below, p.sectors, p.transferkwargs, p.thetaorigin)
    add_sector_scatters!(p, data)
    return p
end

function Makie.preferred_axis_attributes(::Type{Axis}, p::Transfer_Plot)
    θ₀ = float(p.thetaorigin[])
    ks = ceil(Int, 4θ₀ / π):floor(Int, 4(θ₀ + 2π) / π)
    return (;
        title = "Transfer Spectrum",
        xlabel = "θ",
        ylabel = "r",
        xticks = (ks .* (π / 4), pilabel.(ks .// 4)),
        limits = ((θ₀ - 0.1, θ₀ + 2π + 0.1), (nothing, 1.05)),
    )
end

Makie.get_plots(p::Transfer_Plot) = p.plots

# one labelled scatter per sector so `axislegend` lists them; the sector count is fixed at creation
function add_sector_scatters!(p, data)
    sectors = data[].sectors
    isempty(sectors) && return scatter!(p, Point2d[]; visible = false)
    colors = p.colors[]
    for (i, sector) in enumerate(sectors)
        scatter!(
            p, lift(d -> d.points[i], p, data);
            color = colors[mod1(i, length(colors))],
            markersize = p.markersize,
            label = p.sector_formatter[](sector),
        )
    end
    return p
end

function pilabel(x::Rational)
    n, d = numerator(x), denominator(x)
    iszero(n) && return "0"
    s = (abs(n) == 1 ? (n < 0 ? "-" : "") : string(n)) * "π"
    return d == 1 ? s : "$s/$d"
end

end
