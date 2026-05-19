"""
    entanglementplot(state; site = 0[, kwargs...])

Plot the [entanglement spectrum](@ref entanglement_spectrum) of a given MPS `state`. 

# Arguments
- `state`: the MPS for which to compute the entanglement spectrum.

# Keyword Arguments
- `site::Int = 0`: MPS index for multisite unit cells. The spectrum is computed for the bond
  between `site` and `site + 1`.
- `expand_symmetry::Logical = false`: add quantum dimension degeneracies.
- `sortby = maximum`: the method of sorting the sectors.
- `sector_margin = 1//10`: the amount of whitespace between sectors.
- `sector_formatter = string`: how to convert sectors to strings.

!!! note
    You will need to manually import any plotting backend of [Makie.jl](https://github.com/MakieOrg/Makie.jl) or
    [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to be able to use this function. 
"""
function entanglementplot end

"""
    transferplot(above, below = above; sectors = [], transferkwargs = (;))

Plot the partial transfer matrix spectrum of two InfiniteMPS's.

# Arguments
- `above::InfiniteMPS`: above mps for [`transfer_spectrum`](@ref).
- `below::InfiniteMPS = above`: below mps for [`transfer_spectrum`](@ref).

# Keyword Arguments
- `sectors = []`: vector of sectors for which to compute the spectrum. If nothing is passed, the spectrum is computed for the trivial sector.
- `transferkwargs`: kwargs for call to [`transfer_spectrum`](@ref).
- `thetaorigin = 0`: origin of the angle range.
- `sector_formatter = string`: how to convert sectors to strings.

!!! note
    You will need to manually import any plotting backend of [Makie.jl](https://github.com/MakieOrg/Makie.jl) or
    [Plots.jl](https://github.com/JuliaPlots/Plots.jl) to be able to use this function. 
"""
function transferplot end

#TODO: expand on the docstring
