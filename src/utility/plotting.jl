"""
    entanglementplot(state; site=0[, kwargs])

Plot the entanglement spectrum of a given InfiniteMPS.

# Arguments
- `site::Int=0`: mps index for multisite unit cells.
- `expand_symmetry::Logical=false`: add quantum dimension degeneracies.
- `sortby=maximum`: the method of sorting the sectors.
- `sector_margin=1//10`: the amount of whitespace between sectors.
- `sector_formatter=string`: how to convert sectors to strings.
- `kwargs...: other kwargs are passed on to the plotting backend.
"""
function entanglementplot end
@userplot EntanglementPlot

@recipe function f(h::EntanglementPlot; site=0, expand_symmetry=false, sortby=maximum, sector_margin=1//10, sector_formatter=string)

    mps = h.args[1]
    site <= length(mps) ||
        throw(ArgumentError("Not a valid site for the given mps."))

    (_, s, _) = tsvd(mps.CR[site])
    sectors = blocksectors(s)
    spectrum = []
    for sector in sectors
        partial_spectrum = diag(block(s, sector))

        # Duplicate entries according to the quantum dimension.
        if expand_symmetry
            partial_spectrum = repeat(partial_spectrum, dim(sector))
            sort!(partial_spectrum, rev = true)
        end

        push!(spectrum, diag(block(s, sector)))
    end

    if length(spectrum) > 1
        order = sortperm(spectrum, by=sortby, rev=true)
        spectrum = spectrum[order]
        sectors = sectors[order]
    end


    for (i, (partial_spectrum, sector)) in enumerate(zip(spectrum, sectors))
        @series begin
            seriestype := :scatter
            label := sector_formatter(sector)
            n_spectrum = length(partial_spectrum)

            # Put single dot in the middle, or a linear range with padding.
            if n_spectrum == 1
                x = [i+1//2]
            else
                x = range(i + sector_margin, i + 1 - sector_margin,
                    length=n_spectrum)
            end
            return x, partial_spectrum
        end
    end


    title --> "Entanglement Spectrum"
    legend --> false
    grid --> :xy
    widen --> true

    xguide --> "χ = $(dim(domain(s)))"
    xticks --> (1:length(sectors), sector_formatter.(sectors))
    xtickfonthalign --> :center
    xtick_direction --> :out
    xrotation --> 45
    xlims --> (1, length(sectors)+1)

    ylims --> (-Inf, 1+1e-1)
    yscale --> :log10

    return ([])
end



"""
    transferplot(above, below[, sectors[, transferkwargs[, kwargs]]])

Plot the partial transfer matrix spectrum of two InfiniteMPS's.

# Arguments
- `above::InfiniteMPS`: above mps for ``transfer_spectrum``.
- `below::InfiniteMPS`: below mps for ``transfer_spectrum``.
- `sectors=[]`: vector of sectors for which to compute the spectrum.
- `transferkwargs`: kwargs for call to ``transfer_spectrum``.
- `kwargs`: other kwargs are passed on to the plotting backend.
- `thetaorigin=0`: origin of the angle range.
- `sector_formatter=string`: how to convert sectors to strings.
"""
function transferplot end
@userplot TransferPlot
@recipe function f(h::TransferPlot; sectors=nothing, transferkwargs=(;), thetaorigin=0, sector_formatter=string)

    if sectors === nothing
        sectors = [one(sectortype(h.args[1]))]
    end

    for sector in sectors
        below = length(h.args) == 1 ? h.args[1] : h.args[2];
        spectrum = transfer_spectrum(h.args[1]; below=below, sector=sector, transferkwargs...)

        @series begin
            yguide --> "r"
            ylims --> (-Inf,1.05)

            xguide --> "θ"
            xlims --> (thetaorigin,thetaorigin+2pi)
            xticks --> range(0, 2pi,length=7)
            xformatter --> x-> "$(rationalize(x/π, tol=0.05))π"
            xwiden --> true
            seriestype := :scatter
            markershape --> :auto
            label := sector_formatter(sector)
            return mod2pi.(angle.(spectrum).+thetaorigin).-thetaorigin, abs.(spectrum)
        end
    end

    title --> "Transfer Spectrum"
    legend --> false
    grid --> :xy
    framestyle --> :zerolines

    return nothing
end
