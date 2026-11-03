"""
    entanglement_plot(state; site = 0, kwargs...)
    entanglement_plot(gridposition, state; kwargs...)
    entanglement_plot!([axis], state; kwargs...)

Plot the entanglement spectrum (see [`entanglement_spectrum`](@ref)) of a given MPS `state`
using [Makie.jl](https://github.com/MakieOrg/Makie.jl). The Schmidt values of each sector
are drawn as a group of points, on a logarithmic scale.

# Arguments

- `state`: the MPS for which to compute the entanglement spectrum.

# Keyword Arguments

- `site::Int = 0`: MPS index for multisite unit cells. The spectrum is computed for the bond
  between `site` and `site + 1`.
- `expand_symmetry = false`: add quantum dimension degeneracies.
- `sortby = maximum`: the method of sorting the sectors.
- `sector_margin = 1 // 10`: the amount of whitespace between sectors.
- `sector_formatter = string`: how to convert sectors to strings.
- `colors = Makie.wong_colors()`: the colors that are cycled through for the sectors.

All other keywords are handled by Makie as usual, e.g. `axis = (; title = "custom title")`
or `figure = (; size = (800, 600))` for the non-mutating version. The mutating version does
not modify the axis it plots into, unless `use_axis_hints = true` is passed.
Each sector is labelled, such that `axislegend(ax)` adds a legend.

!!! note
    This function requires a Makie backend to be loaded, e.g. `using CairoMakie`.
"""
function entanglement_plot end
function entanglement_plot! end

"""
    transfer_plot(above, below = above; sectors = nothing, transferkwargs = (;), kwargs...)
    transfer_plot(gridposition, above, below = above; kwargs...)
    transfer_plot!([axis], above, below = above; kwargs...)

Plot the partial transfer matrix spectrum (see [`transfer_spectrum`](@ref)) of two
`InfiniteMPS`s using [Makie.jl](https://github.com/MakieOrg/Makie.jl), as the moduli
against the angles of the eigenvalues.

# Arguments

- `above::InfiniteMPS`: above mps for [`transfer_spectrum`](@ref).
- `below::InfiniteMPS = above`: below mps for [`transfer_spectrum`](@ref).

# Keyword Arguments

- `sectors = nothing`: restrict the spectrum to the given sectors; by default all sectors of
  the transfer space are included.
- `transferkwargs = (;)`: kwargs for call to [`transfer_spectrum`](@ref).
- `thetaorigin = 0`: origin of the angle range.
- `sector_formatter = string`: how to convert sectors to strings.
- `colors = Makie.wong_colors()`: the colors that are cycled through for the sectors.

All other keywords are handled by Makie as usual, e.g. `axis = (; title = "custom title")`
or `figure = (; size = (800, 600))` for the non-mutating version. The mutating version does
not modify the axis it plots into, unless `use_axis_hints = true` is passed.
Each sector is labelled, such that `axislegend(ax)` adds a legend.

!!! note
    This function requires a Makie backend to be loaded, e.g. `using CairoMakie`.
"""
function transfer_plot end
function transfer_plot! end

"""
    entanglementplot(state; site = 0[, kwargs...])

Plot the entanglement spectrum (see [`entanglement_spectrum`](@ref)) of a given MPS `state`
using [Plots.jl](https://github.com/JuliaPlots/Plots.jl).

!!! warning "Deprecated"
    The Plots.jl recipes are deprecated, use the Makie.jl based [`entanglement_plot`](@ref) instead.

# Arguments

- `state`: the MPS for which to compute the entanglement spectrum.

# Keyword Arguments

- `site::Int = 0`: MPS index for multisite unit cells. The spectrum is computed for the bond
  between `site` and `site + 1`.
- `expand_symmetry = false`: add quantum dimension degeneracies.
- `sortby = maximum`: the method of sorting the sectors.
- `sector_margin = 1 // 10`: the amount of whitespace between sectors.
- `sector_formatter = string`: how to convert sectors to strings.
- `kwargs...`: other kwargs are passed on to the plotting backend.

!!! note
    You will need to manually import [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to
    be able to use this function.
"""
function entanglementplot end

"""
    transferplot(above, below = above; sectors = nothing, transferkwargs = (;)[, kwargs...])

Plot the partial transfer matrix spectrum of two InfiniteMPS's using
[Plots.jl](https://github.com/JuliaPlots/Plots.jl).

!!! warning "Deprecated"
    The Plots.jl recipes are deprecated, use the Makie.jl based [`transfer_plot`](@ref) instead.

# Arguments

- `above::InfiniteMPS`: above mps for [`transfer_spectrum`](@ref).
- `below::InfiniteMPS = above`: below mps for [`transfer_spectrum`](@ref).

# Keyword Arguments

- `sectors = nothing`: restrict the spectrum to the given sectors; by default all sectors of
  the transfer space are included.
- `transferkwargs`: kwargs for call to [`transfer_spectrum`](@ref).
- `kwargs`: other kwargs are passed on to the plotting backend.
- `thetaorigin = 0`: origin of the angle range.
- `sector_formatter = string`: how to convert sectors to strings.

!!! note
    You will need to manually import [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to
    be able to use this function.
"""
function transferplot end
