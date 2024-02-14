"""
    updatetol(alg, iter, ϵ)

Update the tolerance of the algorithm `alg` based on the current iteration `iter` and the current error `ϵ`.
"""
updatetol(alg, iter::Integer, ϵ::Real) = alg

"""
    DynamicTolerance{A}(alg, tol_min, tol_max, tol_factor)

Algorithm wrapper with dynamically adjusted tolerances.
"""
struct DynamicTolerance{A} <: Algorithm
    alg::A
    tol_min::Float64
    tol_max::Float64
    tol_factor::Float64
    function DynamicTolerance(alg::A, tol_min::Real, tol_max::Real,
                              tol_factor::Real) where {A}
        0 <= tol_min <= tol_max ||
            throw(ArgumentError("tol_min must be between 0 and tol_max"))
        return new{A}(alg, tol_min, tol_max, tol_factor)
    end
end

function updatetol(alg::DynamicTolerance, iter::Integer, ϵ::Real)
    new_tol = between(alg.tol_min, ϵ * alg.tol_factor / sqrt(iter), alg.tol_max)
    old_alg = alg.alg
    return @set old_alg.tol = new_tol
end
