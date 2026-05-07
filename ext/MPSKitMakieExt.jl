module MPSKitMakieExt

using Makie, LaTeXStrings
using MPSKit, TensorKit

#TODO?: add Colors.jl to access this, allows Plots extension to also use these colors
const JLCOLORS = Makie.Colors.JULIA_LOGO_COLORS

@recipe(EntanglementPlot, mps) do scene
    Attributes(
        site = 0,
        expand_symmetry = false,
        sortby = maximum,
        sector_margin = 1 // 10,
        sector_formatter = string,
    )
end

function Makie.plot!(ep::EntanglementPlot)
    #TODO: still want this style where sectors are separated?
    mps = ep.mps[]
    site = ep.site[]
    margin = ep.sector_margin[]

    (isa(mps, FiniteMPS) && (site == 0 || site > length(mps))) &&
        throw(ArgumentError("Invalid site $site for the given mps."))

    spectra = entanglement_spectrum(mps, site)

    sectors = sectortype(mps)[]
    spectrum = Vector{Vector{Float64}}()

    for (c, b) in pairs(spectra)
        if ep.expand_symmetry[]
            b′ = repeat(b, dim(c))
            sort!(b′; rev = true)
            push!(spectrum, b′)
        else
            push!(spectrum, b)
        end
        push!(sectors, c)
    end

    # Sort sectors according to provided method
    if length(spectrum) > 1
        order = sortperm(spectrum; by = ep.sortby[], rev = true)
        spectrum = spectrum[order]
        sectors = sectors[order]
    end

    ax = Makie.current_axis()

    # Axis styling
    ax.title = L"\text{Entanglement Spectrum}"
    ax.titlesize = 24

    ax.xlabel = latexstring("\$\\chi\$ = $(round(Int, dim(left_virtualspace(mps, site))))") #TODO: still want this?
    ax.xlabelsize = 24
    ax.xticks = (1:length(sectors), ep.sector_formatter[].(sectors))
    ax.xticklabelsize = 16
    ax.xticklabelrotation = 45
    ax.xticklabelalign = (:right, :top)
    xlims!(ax, 1, length(sectors) + 1)

    ax.ylabel = L"\log(\lambda)"
    ax.ylabelsize = 24
    bottom = floor(Int, log10(minimum(spectra)))
    ax.yticks = (bottom:2:0, latexstring.(collect(bottom:2:0)))
    ax.yticklabelsize = 16
    ylims!(ax, bottom, 0 + 1.0e-1)

    # Plot data
    for (i, (partial_spectrum, sector)) in enumerate(zip(spectrum, sectors))
        n_spectrum = length(partial_spectrum)
        if n_spectrum == 1
            x = [i + 0.5]
        else
            x = collect(range(i + float(margin), i + 1 - float(margin); length = n_spectrum))
        end
        scatter!(ep, x, log10.(partial_spectrum), color = JLCOLORS[mod1(i, length(JLCOLORS))])
    end

    return ep
end

MPSKit.entanglementplot(args...; kwargs...) = entanglementplot(args...; kwargs...)

#------------------------------------------------------------

@recipe(TransferPlot, mps) do scene
    Attributes(
        below = nothing,
        sectors = nothing,
        transferkwargs = NamedTuple(),
        thetaorigin = 0.0,
        sector_formatter = string,
    )
end

function Makie.plot!(tp::TransferPlot)
    #TODO: consider radial plot
    mps = tp.mps[]
    below = tp.below[] === nothing ? mps : tp.below[]
    sectors = tp.sectors[] === nothing ? [leftunit(mps)] : tp.sectors[]
    transferkwargs = NamedTuple( # weird convert thing
        k => (v isa Observable ? v[] : v) for (k, v) in pairs(tp.transferkwargs[])
    )
    thetaorigin = tp.thetaorigin[]
    sector_formatter = tp.sector_formatter[]

    ax = Makie.current_axis()
    ax.title = L"\text{Transfer Spectrum}"
    ax.titlesize = 24
    ax.xlabel = L"\theta"
    ax.xlabelsize = 24
    ax.xticklabelsize = 16
    ax.ylabel = L"r"
    ax.ylabelsize = 24
    ax.yticklabelsize = 16

    ax.xticks = pitick(0, 2pi, 4; mode = :latex)
    ax.yticks = (range(0, 1.0; length = 6), latexstring.(range(0, 1.0; length = 6)))
    ax.xgridvisible = true
    ax.ygridvisible = true

    ax.leftspinevisible = true
    ax.rightspinevisible = false
    ax.bottomspinevisible = true
    ax.topspinevisible = false
    @show transferkwargs

    for (i, sector) in enumerate(sectors)
        spectrum = transfer_spectrum(mps; below = below, sector = sector, transferkwargs...)
        θ = mod2pi.(angle.(spectrum) .+ thetaorigin) .- thetaorigin
        r = abs.(spectrum)
        scatter!(tp, θ, r; label = sector_formatter(sector), color = JLCOLORS[mod1(i, length(JLCOLORS))])
    end

    xlims!(ax, thetaorigin - 0.1, thetaorigin + 2π + 0.1)
    ylims!(ax, nothing, 1.05)
    Legend(Makie.current_figure()[1, 1], tp.plots, [sector_formatter(s) for s in sectors]; tellwidth = false, halign = :center, valign = :top)
    return tp
end

MPSKit.transferplot(args...; kwargs...) = transferplot(args...; kwargs...)

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
