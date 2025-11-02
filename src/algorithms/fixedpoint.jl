# wrapper around KrylovKit.jl's eigsolve function

"""
    fixedpoint(A, x₀, which::Symbol; kwargs...) -> val, vec
    fixedpoint(A, x₀, which::Symbol, alg) -> val, vec

Compute the fixed point of a linear operator `A` using the specified eigensolver `alg`. The
fixedpoint is assumed to be unique.
"""
function fixedpoint(A, x₀, which::Symbol, alg::Lanczos)
    A′, x₀′ = prepare_operator!!(A, x₀)
    vals, vecs, info = eigsolve(A′, x₀′, 1, which, alg)

    info.converged == 0 &&
        @warnv 1 "fixed point not converged after $(info.numiter) iterations: normres = $(info.normres[1])"

    λ = vals[1]
    v = unprepare_operator!!(vecs[1], A′, x₀)

    return λ, v
end

function fixedpoint(A, x₀, which::Symbol, alg::Arnoldi)
    A′, x₀′ = prepare_operator!!(A, x₀)
    TT, vecs, vals, info = schursolve(A′, x₀′, 1, which, alg)

    info.converged == 0 &&
        @warnv 1 "fixed point not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    size(TT, 2) > 1 && TT[2, 1] != 0 && @warnv 1 "non-unique fixed point detected"

    λ = vals[1]
    v = unprepare_operator!!(vecs[1], A′, x₀)

    return λ, v
end

function fixedpoint(A, x₀, which::Symbol; kwargs...)
    alg = KrylovKit.eigselector(A, scalartype(x₀); kwargs...)
    return fixedpoint(A, x₀, which, alg)
end
