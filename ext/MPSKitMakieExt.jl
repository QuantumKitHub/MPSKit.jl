module MPSKitMakieExt

using Makie
using MPSKit
using TensorKit: dim
import MPSKit: entanglementplot, transferplot

# ──────────────────────────────────────────────────────────────────────────────
# Entanglement spectrum
# ──────────────────────────────────────────────────────────────────────────────

@recipe EntanglementPlot (mps,) begin
    site = 0
    expand_symmetry = false
    sortby = maximum
    sector_margin = 1 // 10
    sector_formatter = string
    markersize = 10
    marker = :circle
    colormap = :tab10
end

function Makie.plot!(p::EntanglementPlot)
    # Tier 1 — expensive. Re-runs only when mps / site / expand_symmetry / sortby change.
    map!(
        p.attributes,
        [:mps, :site, :expand_symmetry, :sortby],
        :raw,
    ) do mps, site, expand, sb
        (site <= length(mps) && !(isa(mps, FiniteMPS) && site == 0)) ||
            throw(ArgumentError("Invalid site $site for the given mps."))

        spectra = entanglement_spectrum(mps, site)
        sectors = collect(keys(spectra))
        spectrum = [
            expand ? sort!(repeat(collect(spectra[c]), dim(c)); rev = true) :
                collect(spectra[c])
                for c in sectors
        ]

        if length(spectrum) > 1
            order = sortperm(spectrum; by = sb, rev = true)
            spectrum = spectrum[order]
            sectors = sectors[order]
        end
        return (; sectors, spectrum)
    end

    # Tier 2 — cheap reshapes. Each is a separate node so changes propagate
    # independently and cosmetic attrs flow direct to scatter.
    map!(p.attributes, [:raw], :ys) do r
        return reduce(vcat, r.spectrum; init = Float64[])
    end

    map!(p.attributes, [:raw, :sector_margin], :xs) do r, m
        x = Float64[]
        for (i, vals) in enumerate(r.spectrum)
            n = length(vals)
            if n == 1
                push!(x, i + 0.5)
            else
                append!(x, range(i + float(m), i + 1 - float(m); length = n))
            end
        end
        return x
    end

    map!(p.attributes, [:raw], :ci) do r
        return reduce(
            vcat,
            (fill(i, length(v)) for (i, v) in enumerate(r.spectrum));
            init = Int[],
        )
    end

    scatter!(
        p, p.xs, p.ys;
        color = p.ci,
        colormap = p.colormap,
        markersize = p.markersize,
        marker = p.marker,
    )
    return p
end

# ──────────────────────────────────────────────────────────────────────────────
# Transfer spectrum
# ──────────────────────────────────────────────────────────────────────────────

@recipe TransferPlot (above, below) begin
    sectors = nothing
    transferkwargs = NamedTuple()
    thetaorigin = 0.0
    sector_formatter = string
    markersize = 10
    marker = :circle
    colormap = :tab10
end

# 1-arg surface parity with the Plots recipe: transferplot(ψ) ≡ transferplot(ψ, ψ).
# The non-bang form lives in src/utility/plotting.jl; bang forms live here because
# they only exist once @recipe has generated them.
transferplot!(above; kwargs...) = transferplot!(above, above; kwargs...)
transferplot!(ax, above; kwargs...) = transferplot!(ax, above, above; kwargs...)

function Makie.plot!(p::TransferPlot)
    # Tier 1 — VERY expensive (Krylov per sector).
    map!(
        p.attributes,
        [:above, :below, :sectors, :transferkwargs],
        :eigs,
    ) do a, b, secs, kw
        slist = secs === nothing ? [leftunit(a)] : collect(secs)
        return [(s, transfer_spectrum(a; below = b, sector = s, kw...)) for s in slist]
    end

    # Tier 2 — coordinate splits, each its own cheap node.
    map!(p.attributes, [:eigs, :thetaorigin], :θ) do d, θ0
        return reduce(
            vcat,
            (mod2pi.(angle.(λ) .+ θ0) .- θ0 for (_, λ) in d);
            init = Float64[],
        )
    end
    map!(p.attributes, [:eigs], :r) do d
        return reduce(vcat, (abs.(λ) for (_, λ) in d); init = Float64[])
    end
    map!(p.attributes, [:eigs], :ci) do d
        return reduce(
            vcat,
            (fill(i, length(λ)) for (i, (_, λ)) in enumerate(d));
            init = Int[],
        )
    end

    scatter!(
        p, p.θ, p.r;
        color = p.ci,
        colormap = p.colormap,
        markersize = p.markersize,
        marker = p.marker,
    )
    return p
end

end # module
