module MPSKitMakieExt

using Makie, LaTeXStrings
using MPSKit, TensorKit

#TODO?: add Colors.jl to access this, allows Plots extension to also use these colors
const JLCOLORS = Makie.Colors.JULIA_LOGO_COLORS

sector_color(i::Integer) = JLCOLORS[mod1(i, length(JLCOLORS))]

convert_kwargs(kwargs::NamedTuple) = kwargs
function convert_kwargs(kwargs) # weird convert thing
    return NamedTuple(Symbol(k) => (v isa Observable ? v[] : v) for (k, v) in pairs(kwargs))
end

# the recipes publish the axis attributes they want as an `:axis_info` node instead of `current_axis()`
function apply_axis_info!(ax, plot, plotkwargs = (;))
    ax isa Makie.AbstractAxis || return ax
    function apply!(info)
        for (k, v) in pairs(info)
            setproperty!(ax, k, v)
        end
        # user-provided attributes take precedence
        for (k, v) in pairs(plotkwargs)
            setproperty!(ax, k, v)
        end
        return nothing
    end
    node = plot.attributes[:axis_info]
    apply!(node[])
    on(apply!, node) # keep the axis in sync when the inputs change
    return ax
end

@recipe EntanglementPlot (mps,) begin
    site = 0
    expand_symmetry = false
    sortby = maximum
    sector_margin = 1 // 10
    sector_formatter = string
    markersize = 12
    marker = :circle
end

function Makie.plot!(ep::EntanglementPlot)
    # this closure only reruns when one of the inputs changes
    map!(ep.attributes, [:mps, :site, :expand_symmetry, :sortby], :spectrum_data) do mps, site, expand_symmetry, sortby
        spectra = entanglement_spectrum(mps, site)

        sectors = sectortype(mps)[]
        spectrum = Vector{Vector{Float64}}()
        for (c, b) in pairs(spectra)
            if expand_symmetry
                b′ = repeat(collect(b), dim(c))
                sort!(b′; rev = true)
            else
                b′ = collect(b)
            end
            push!(spectrum, b′)
            push!(sectors, c)
        end

        if any(v -> any(<=(0), v), spectrum)
            @warn "Entanglement spectrum contains vanishing Schmidt values. These are omitted from the plot."
            foreach(v -> filter!(>(0), v), spectrum)
        end

        # Sort sectors according to provided method
        if length(spectrum) > 1
            order = sortperm(spectrum; by = sortby, rev = true)
            spectrum = spectrum[order]
            sectors = sectors[order]
        end
        return (; sectors, spectrum)
    end

    # styling
    # only reruns when spectrum_data or sector_margin changes
    map!(ep.attributes, [:spectrum_data, :sector_margin], :positions) do data, margin
        points = Point2d[]
        for (i, partial_spectrum) in enumerate(data.spectrum)
            n_spectrum = length(partial_spectrum)
            xs = if n_spectrum == 1
                range(i + 0.5, i + 0.5; length = 1)
            else
                range(i + float(margin), i + 1 - float(margin); length = n_spectrum)
            end
            for (x, λ) in zip(xs, partial_spectrum)
                push!(points, Point2d(x, log10(λ)))
            end
        end
        return points
    end

    # only reruns when spectrum_data changes
    map!(ep.attributes, [:spectrum_data], :colors) do data
        colors = typeof(sector_color(1))[]
        for (i, partial_spectrum) in enumerate(data.spectrum)
            append!(colors, fill(sector_color(i), length(partial_spectrum)))
        end
        return colors
    end

    map!(ep.attributes, [:mps, :site, :spectrum_data, :sector_formatter], :axis_info) do mps, site, data, sector_formatter
        nsectors = length(data.sectors)
        bottom = if isempty(data.spectrum)
            -1
        else
            smallest = minimum(Iterators.flatten(data.spectrum); init = 1.0) # spectrum is already > 0
            floor(Int, log10(smallest))
        end
        return (;
            xticks = (1:nsectors, sector_formatter.(data.sectors)),
            xticklabelsize = 16,
            xticklabelrotation = 45.0,
            xticklabelalign = (:right, :top),
            ylabel = L"\log(\lambda)",
            ylabelsize = 24,
            yticks = (bottom:2:0, latexstring.(collect(bottom:2:0))),
            yticklabelsize = 16,
            limits = ((1, nsectors + 1), (bottom, 0 + 1.0e-1)),
        )
    end

    scatter!(ep, ep.positions; color = ep.colors, markersize = ep.markersize, marker = ep.marker)
    return ep
end

function MPSKit.entanglementplot(args...; plotkwargs = (;), kwargs...)
    p = entanglementplot(args...; kwargs...)
    apply_axis_info!(p.axis, p.plot, plotkwargs)
    return p
end

function MPSKit.entanglementplot!(state::MPSKit.AbstractMPS; plotkwargs = (;), kwargs...)
    p = entanglementplot!(state; kwargs...)
    apply_axis_info!(Makie.current_axis(), p, plotkwargs)
    return p
end
function MPSKit.entanglementplot!(target, state::MPSKit.AbstractMPS; plotkwargs = (;), kwargs...)
    p = entanglementplot!(target, state; kwargs...)
    apply_axis_info!(target, p, plotkwargs)
    return p
end

#------------------------------------------------------------

@recipe TransferPlot (above, below) begin
    sectors = nothing
    transferkwargs = NamedTuple()
    thetaorigin = 0.0
    sector_formatter = string
    legend_position = :ct
    markersize = 12
    marker = :circle
end

function Makie.plot!(tp::TransferPlot)
    #TODO: consider radial plot
    # this only reruns when one of the inputs changes
    map!(tp.attributes, [:above, :below, :sectors, :transferkwargs], :spectrum_data) do above, below, sectors, transferkwargs
        kwargs = convert_kwargs(transferkwargs)
        if sectors !== nothing && get(kwargs, :howmany, 20) isa Int
            # restrict the computation to the requested sectors
            howmany = Dict(c => get(kwargs, :howmany, 20) for c in sectors)
            kwargs = (; kwargs..., howmany)
        end
        spectra = transfer_spectrum(above, below; kwargs...)

        data = Pair{sectortype(above), Vector{complex(scalartype(above))}}[]
        for (sector, spectrum) in pairs(spectra)
            sectors === nothing || sector in sectors || continue
            push!(data, sector => collect(spectrum))
        end
        return data
    end

    map!(tp.attributes, [:spectrum_data, :thetaorigin], :positions) do data, thetaorigin
        points = Point2d[]
        for (_, spectrum) in data, λ in spectrum
            θ = mod2pi(angle(λ) + thetaorigin) - thetaorigin
            push!(points, Point2d(θ, abs(λ)))
        end
        return points
    end

    map!(tp.attributes, [:spectrum_data], :colors) do data
        colors = typeof(sector_color(1))[]
        for (i, (_, spectrum)) in enumerate(data)
            append!(colors, fill(sector_color(i), length(spectrum)))
        end
        return colors
    end

    map!(tp.attributes, [:thetaorigin], :axis_info) do thetaorigin
        return (;
            xlabel = L"\theta",
            xlabelsize = 24,
            xticks = pitick(0, 2pi, 4; mode = :latex),
            xticklabelsize = 16,
            ylabel = L"r",
            ylabelsize = 24,
            yticks = (range(0, 1.0; length = 6), latexstring.(range(0, 1.0; length = 6))),
            yticklabelsize = 16,
            xgridvisible = true,
            ygridvisible = true,
            leftspinevisible = true,
            rightspinevisible = false,
            bottomspinevisible = true,
            topspinevisible = false,
            limits = ((thetaorigin - 0.1, thetaorigin + 2π + 0.1), (nothing, 1.05)),
        )
    end

    map!(tp.attributes, [:spectrum_data, :sector_formatter], :legend_entries) do data, sector_formatter
        return [(sector_formatter(sector), sector_color(i)) for (i, (sector, _)) in enumerate(data)]
    end

    scatter!(tp, tp.positions; color = tp.colors, markersize = tp.markersize, marker = tp.marker)
    return tp
end

function add_sector_legend!(ax, plot, legend_position)
    ax isa Makie.AbstractAxis || return nothing
    entries = plot.attributes[:legend_entries][]
    isempty(entries) && return nothing
    elements = [MarkerElement(; color, marker = :circle, markersize = 12) for (_, color) in entries]
    # cannot use current_figure() when supporting in-place method
    axislegend(ax, elements, [label for (label, _) in entries]; position = legend_position)
    return nothing
end

function MPSKit.transferplot(above, below = above; plotkwargs = (;), kwargs...)
    p = transferplot(above, below; kwargs...)
    apply_axis_info!(p.axis, p.plot, plotkwargs)
    add_sector_legend!(p.axis, p.plot, p.plot.legend_position[])
    return p
end

function MPSKit.transferplot!(
        above::MPSKit.AbstractMPS, below::MPSKit.AbstractMPS = above;
        plotkwargs = (;), kwargs...
    )
    p = transferplot!(above, below; kwargs...)
    ax = Makie.current_axis()
    apply_axis_info!(ax, p, plotkwargs)
    add_sector_legend!(ax, p, p.legend_position[])
    return p
end
function MPSKit.transferplot!(
        target, above::MPSKit.AbstractMPS, below::MPSKit.AbstractMPS = above;
        plotkwargs = (;), kwargs...
    )
    p = transferplot!(target, above, below; kwargs...)
    apply_axis_info!(target, p, plotkwargs)
    add_sector_legend!(target, p, p.legend_position[])
    return p
end

# utility for plotting

function pitick(start, stop, denom; mode = :latex)
    a = Int(cld(start, π / denom))
    b = Int(fld(stop, π / denom))
    tick = range(a * π / denom, b * π / denom; step = π / denom)
    ticklabel = piticklabel.((a:b) .// denom, Val(mode))
    return tick, ticklabel
end

function piticklabel(x::Rational, ::Val{:text})
    iszero(x) && return "0"
    S = x < 0 ? "-" : ""
    n, d = abs(numerator(x)), denominator(x)
    N = n == 1 ? "" : repr(n)
    d == 1 && return S * N * "π"
    return S * N * "π/" * repr(d)
end

function piticklabel(x::Rational, ::Val{:latex})
    iszero(x) && return L"0"
    S = x < 0 ? "-" : ""
    n, d = abs(numerator(x)), denominator(x)
    N = n == 1 ? "" : repr(n)
    d == 1 && return L"%$S%$N\pi"
    return L"%$S\frac{%$N\pi}{%$d}"
end

end
