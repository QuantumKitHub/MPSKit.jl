module MPSKitPlotsExt

using RecipesBase
using MPSKit, TensorKit

@userplot EntanglementPlot

@recipe function f(
        h::EntanglementPlot; site = 0, expand_symmetry = false, sortby = maximum,
        sector_margin = 1 // 10, sector_formatter = string
    )
    mps = h.args[1]

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
            bottom_margin --> (10, :mm)

            xguide --> "χ = $(dim(MPSKit._firstspace(mps.C[site])))"
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

MPSKit.entanglementplot(args...; plotkwargs = (;), kwargs...) = entanglementplot(args...; kwargs..., plotkwargs...)
MPSKit.entanglementplot!(args...; plotkwargs = (;), kwargs...) = entanglementplot!(args...; kwargs..., plotkwargs...)

#-----------------------------------------------------------------------------

@userplot TransferPlot

@recipe function f(
        h::TransferPlot; sectors = nothing, transferkwargs = (;), thetaorigin = 0,
        sector_formatter = string
    )
    below = length(h.args) == 1 ? h.args[1] : h.args[2]
    kwargs = (; transferkwargs...)
    if sectors !== nothing && get(kwargs, :howmany, 20) isa Int
        howmany = Dict(c => get(kwargs, :howmany, 20) for c in sectors)
        kwargs = (; kwargs..., howmany)
    end
    spectra = transfer_spectrum(h.args[1], below; kwargs...)

    for (sector, spectrum) in pairs(spectra)
        sectors === nothing || sector in sectors || continue

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
            return mod2pi.(angle.(spectrum) .+ thetaorigin) .- thetaorigin, abs.(spectrum)
        end
    end

    title --> "Transfer Spectrum"
    legend --> false
    grid --> :xy
    framestyle --> :zerolines

    return nothing
end

MPSKit.transferplot(args...; plotkwargs = (;), kwargs...) = transferplot(args...; kwargs..., plotkwargs...)
MPSKit.transferplot!(args...; plotkwargs = (;), kwargs...) = transferplot!(args...; kwargs..., plotkwargs...)

end
