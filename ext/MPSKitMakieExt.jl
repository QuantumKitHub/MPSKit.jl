module MPSKitMakieExt

using Makie
using MPSKit
import MPSKit: entanglementplot, transferplot

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

    mps = ep.mps[]
    site = ep.site[]

    (site <= length(mps) && !(isa(mps, FiniteMPS) && site == 0)) ||
        throw(ArgumentError("Invalid site $site for the given mps."))

    spectra = entanglement_spectrum(mps, site)

    sectors = []
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

    # Sort sectors
    if length(spectrum) > 1
        order = sortperm(spectrum; by = ep.sortby[], rev = true)
        spectrum = spectrum[order]
        sectors = sectors[order]
    end

    ax = ep.axis

    # Axis styling
    ax.title = "Entanglement Spectrum"
    ax.xlabel = "χ = $(round(Int, dim(left_virtualspace(mps, site))))"
    ax.yscale = log10
    ax.xticklabelrotation = π / 4
    ax.xticklabelalign = (:center, :top)

    # Plot data
    for (i, (partial_spectrum, sector)) in enumerate(zip(spectrum, sectors))

        n_spectrum = length(partial_spectrum)

        if n_spectrum == 1
            x = [i + 0.5]
        else
            x = collect(
                range(
                    i + float(ep.sector_margin[]),
                    i + 1 - float(ep.sector_margin[]);
                    length = n_spectrum
                )
            )
        end

        scatter!(ep, x, partial_spectrum)
    end

    ax.xticks = (
        1:length(sectors),
        ep.sector_formatter[].(sectors),
    )

    xlims!(ax, 1, length(sectors) + 1)
    ylims!(ax, nothing, 1 + 1.0e-1)

    return ep
end

#------------------------------------------------------------

@recipe(TransferPlot, mps) do scene
    Attributes(
        sectors = nothing,
        transferkwargs = NamedTuple(),
        thetaorigin = 0.0,
        sector_formatter = string,
    )
end

function Makie.plot!(tp::TransferPlot)

    mps = tp.mps[]

    sectors = tp.sectors[]
    transferkwargs = tp.transferkwargs[]
    thetaorigin = tp.thetaorigin[]
    sector_formatter = tp.sector_formatter[]

    if sectors === nothing
        sectors = [leftunit(mps)]
    end

    # axis configuration matches old Plots recipe
    ax = tp.axis
    ax.title = "Transfer Spectrum"
    ax.xlabel = "θ"
    ax.ylabel = "r"
    ax.xlimits = (thetaorigin, thetaorigin + 2π)
    ax.ylimits = (nothing, 1.05)
    ax.xticks = (
        range(0, 2π; length = 7),
        [
            "$(rationalize(x / π, tol = 0.05))π"
                for x in range(0, 2π; length = 7)
        ],
    )
    ax.xgridvisible = true
    ax.ygridvisible = true

    # same as framestyle --> :zerolines
    ax.leftspinevisible = false
    ax.bottomspinevisible = false
    hlines!(ax, 0, linewidth = 1)
    vlines!(ax, 0, linewidth = 1)

    below = length(tp.args) == 1 ? mps : tp.args[2][]

    for sector in sectors
        spectrum = transfer_spectrum(
            mps; below = below, sector = sector,
            transferkwargs...
        )

        θ = mod2pi.(angle.(spectrum) .+ thetaorigin) .- thetaorigin
        r = abs.(spectrum)

        scatter!(tp, θ, r; label = sector_formatter(sector))
    end

    return tp
end

end
