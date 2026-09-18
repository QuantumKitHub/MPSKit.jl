"""
    entanglementplot(state; site = 0[, kwargs...])

Plot the entanglement spectrum (see [`entanglement_spectrum`](@ref)) of a given MPS `state`.

# Arguments

- `state`: the MPS for which to compute the entanglement spectrum.

# Keyword Arguments

- `site::Int = 0`: MPS index for multisite unit cells. The spectrum is computed for the bond
  between `site` and `site + 1`.
- `expand_symmetry = false`: add quantum dimension degeneracies.
- `sortby = maximum`: the method of sorting the sectors.
- `sector_margin = 1//10`: the amount of whitespace between sectors.
- `sector_formatter = string`: how to convert sectors to strings.
- `plotkwargs = (; )`: kwargs for the underlying plot, e.g. `plotkwargs = (; title = "custom title", xlabel =   L"\text{custom label}", xticks = (1:2, ["a", "b"]))`.

!!! note
    You will need to manually import any plotting backend of [Makie.jl](https://github.com/MakieOrg/Makie.jl) or
    [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to be able to use this function.

See also [`entanglementplot!`](@ref) for plotting into an existing figure.
"""
function entanglementplot end
function entanglementplot! end

"""
    transferplot(above, below = above; sectors = nothing, transferkwargs = (;), plotkwargs = (;), legend_position = :ct)

Plot the partial transfer matrix spectrum of two InfiniteMPS's.

# Arguments

- `above::InfiniteMPS`: above mps for [`transfer_spectrum`](@ref).
- `below::InfiniteMPS = above`: below mps for [`transfer_spectrum`](@ref).

# Keyword Arguments

- `sectors = nothing`: restrict the spectrum to the given sectors; by default all sectors of
  the transfer space are included.
- `transferkwargs`: kwargs for call to [`transfer_spectrum`](@ref).
- `plotkwargs = (; )`: kwargs for the underlying plot, e.g. `plotkwargs = (; title = "custom title", xlabel = L"latexstring", xticks = (1:2, ["a", "b"]))`.
- `thetaorigin = 0`: origin of the angle range.
- `sector_formatter = string`: how to convert sectors to strings.
- `legend_position = :ct`: Makie only, the `position` passed to `axislegend`.
  For Plots, use the standard `legend` attribute instead (e.g. `legend = :topright`) in `plotkwargs`.

!!! note
    You will need to manually import any plotting backend of [Makie.jl](https://github.com/MakieOrg/Makie.jl) or
    [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to be able to use this function.

See also [`transferplot!`](@ref) for plotting into an existing figure.
"""
function transferplot end
function transferplot! end
