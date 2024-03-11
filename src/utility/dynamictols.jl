module DynamicTols

import ..MPSKit: Algorithm
using Accessors

export updatetol, DynamicTol

@doc """
    updatetol(alg, iter, ϵ)

Update the tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`.
""" updatetol

updatetol(alg, iter::Integer, ϵ::Real) = alg

# Wrapper for dynamic tolerance adjustment
# ----------------------------------------

"""
    DynamicTol{A}(alg, tol_min, tol_max, tol_factor)

Algorithm wrapper with dynamically adjusted tolerances.

See also [`updatetol`](@ref).
"""
struct DynamicTol{A} <: Algorithm
    alg::A
    tol_min::Float64
    tol_max::Float64
    tol_factor::Float64
    function DynamicTol(alg::A, tol_min::Real, tol_max::Real,
                        tol_factor::Real) where {A}
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("tol_min must be between 0 and tol_max"))
        return new{A}(alg, tol_min, tol_max, tol_factor)
    end
end
function DynamicTol(alg; tol_min=1e-6, tol_max=1e-2, tol_factor=0.1)
    return DynamicTol(alg, tol_min, tol_max, tol_factor)
end

"""
    updatetol(alg::DynamicTol, iter, ϵ)

Update the tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`,
where the new tolerance is given by
    
    new_tol = clamp(ϵ * alg.tol_factor / sqrt(iter), alg.tol_min, alg.tol_max)
"""
function updatetol(alg::DynamicTol, iter::Integer, ϵ::Real)
    new_tol = clamp(ϵ * alg.tol_factor / sqrt(iter), alg.tol_min, alg.tol_max)
    return _updatetol(alg.alg, new_tol)
end

# default implementation with Accessors.jl, but can be hooked into
function _updatetol(alg, tol::Real)
    return @set alg.tol = tol
end

end
