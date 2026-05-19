module MPSKitPlotsExt

using RecipesBase, LaTeXStrings
using MPSKit, TensorKit

@userplot EntanglementPlot

@recipe function f(
        h::EntanglementPlot; site = 0, expand_symmetry = false, sortby = maximum,
        sector_margin = 1 // 10, sector_formatter = string
    )
    mps = h.args[1]
    (isa(mps, FiniteMPS) && (site == 0 || site > length(mps))) &&
        throw(ArgumentError("Invalid site $site for the given mps."))

    spectra = entanglement_spectrum(mps, site)
    sectors = sectortype(mps)[]
    spectrum = Vector{Vector{Float64}}()
    for (c, b) in pairs(spectra)
        if expand_symmetry # Duplicate entries according to the quantum dimension.
            b′ = repeat(b, dim(c))
            sort!(b′; rev = true)
            push!(spectrum, b′)
        else
            push!(spectrum, b)
        end
        push!(sectors, c)
    end

    if length(spectrum) > 1
        order = sortperm(spectrum; by = sortby, rev = true)
        spectrum = spectrum[order]
        sectors = sectors[order]
    end

    for (i, (partial_spectrum, sector)) in enumerate(zip(spectrum, sectors))
        @series begin
            title --> "Entanglement Spectrum"
            legend --> false
            grid --> :xy
            widen --> true
            bottom_margin -->(10, :mm)

            xguide --> latexstring("\$\\chi\$ = $(round(Int, dim(left_virtualspace(mps, site))))")
            xticks --> (1:length(sectors), sector_formatter.(sectors))
            xtickfonthalign --> :center
            xtick_direction --> :out
            xrotation --> 45
            xlims --> (1, length(sectors) + 1)

            ylims --> (-Inf, 1 + 1.0e-1)
            yscale --> :log10
            seriestype := :scatter
            label := sector_formatter(sector)
            n_spectrum = length(partial_spectrum)

            # Put single dot in the middle, or a linear range with padding.
            if n_spectrum == 1
                x = [i + 1 // 2]
            else
                x = range(i + sector_margin, i + 1 - sector_margin; length = n_spectrum)
            end
            return x, partial_spectrum
        end
    end

    return nothing
end

MPSKit.entanglementplot(args...; kwargs...) = entanglementplot(args...; kwargs...)

#-----------------------------------------------------------------------------

@userplot TransferPlot

@recipe function f(
        h::TransferPlot; sectors = nothing, transferkwargs = (;), thetaorigin = 0,
        sector_formatter = string
    )
    if sectors === nothing
        sectors = [leftunit(h.args[1])]
    end

    ticks, ticklabels = pitick(0, 2pi, 4; mode = :latex)
    for sector in sectors
        below = length(h.args) == 1 ? h.args[1] : h.args[2]
        spectrum = transfer_spectrum(
            h.args[1]; below = below, sector = sector,
            transferkwargs...
        )

        @series begin
            yguide --> L"r"
            ylims --> (-Inf, 1.05)

            xguide --> L"\theta"
            xlims --> (thetaorigin, thetaorigin + 2pi)
            xticks --> ticks
            xformatter --> x -> ticklabels[findfirst(==(x), ticks)]
            xwiden --> true
            seriestype := :scatter
            markershape --> :auto
            label := sector_formatter(sector)
            return mod2pi.(angle.(spectrum) .+ thetaorigin) .- thetaorigin, abs.(spectrum)
        end
    end

    title --> "Transfer Spectrum"
    legend --> false
    grid --> :xy
    framestyle --> :zerolines

    return nothing
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
