# wrapper around KrylovKit.jl's eigsolve function

"""
    fixedpoint(A, x₀, which::Symbol, alg) -> val, vec

Compute the fixedpoint of a linear operator `A` using the specified eigensolver `alg`. The
fixedpoint is assumed to be unique.
"""
function fixedpoint(A, x₀, which::Symbol, alg::Lanczos)
    vals, vecs, info = eigsolve(A, x₀, 1, which, alg)

    if info.converged == 0
        @warn "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end

    return vals[1], vecs[1]
end

function fixedpoint(A, x₀, which::Symbol, alg::Arnoldi)
    TT, vecs, vals, info = schursolve(A, x₀, 1, which, alg)

    if info.converged == 0
        @warn "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end
    if size(TT, 2) > 1 && TT[2, 1] != 0
        @warn "non-unique fixedpoint detected"
    end

    return vals[1], vecs[1]
end
