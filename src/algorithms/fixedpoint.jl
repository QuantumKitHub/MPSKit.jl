# wrapper around KrylovKit.jl's eigsolve function

function fixedpoint(A, x₀, which::Symbol; kwargs...)
    alg = KrylovKit.eigselector(A, scalartype(x₀); kwargs...)
    return fixedpoint(A, x₀, which, alg)
end

"""
    fixedpoint(A, x₀, which::Symbol; kwargs...) -> val, vec, info
    fixedpoint(A, x₀, which::Symbol, alg) -> val, vec, info

Compute the fixed point of a given linear operator `A` with initial guess `x₀`.
The dominant eigenvector is assumed to be unique.
"""
function fixedpoint(A, x₀, which::Symbol, alg::Lanczos)
    vals, vecs, info = eigsolve(A, x₀, 1, which, alg)
    return vals[1], vecs[1], info
end
function fixedpoint(A, x₀, which::Symbol, alg::Arnoldi)
    TT, vecs, vals, info = schursolve(A, x₀, 1, which, alg)
    size(TT, 2) > 1 && TT[2, 1] != 0 && @warnv 1 "non-unique fixed point detected"
    return vals[1], vecs[1], info
end
