# wrapper around KrylovKit.jl's eigsolve function

"""
    fixedpoint(A, x₀, which::Symbol; kwargs...) -> val, vec
    fixedpoint(A, x₀, which::Symbol, alg) -> val, vec

Compute the fixedpoint of a linear operator `A` using the specified eigensolver `alg`. The
fixedpoint is assumed to be unique.
"""
function fixedpoint(A, x₀, which::Symbol, alg::Lanczos)
    vals, vecs, info = eigsolve(A, x₀, 1, which, alg)

    info.converged == 0 &&
        @warnv 1 "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"

    return vals[1], vecs[1]
end

function fixedpoint(A, x₀, which::Symbol, alg::Arnoldi)
    TT, vecs, vals, info = schursolve(A, x₀, 1, which, alg)

    info.converged == 0 &&
        @warnv 1 "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    size(TT, 2) > 1 && TT[2, 1] != 0 && @warnv 1 "non-unique fixedpoint detected"

    return vals[1], vecs[1]
end

function fixedpoint(A, x₀, which::Symbol; kwargs...)
    alg = KrylovKit.eigselector(A, scalartype(x₀); kwargs...)
    return fixedpoint(A, x₀, which, alg)
end
