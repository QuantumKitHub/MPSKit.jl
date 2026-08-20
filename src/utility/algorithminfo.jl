"""
$(TYPEDEF)

Information about how an algorithm arrived at its result, returned as the last value by
[`find_groundstate`](@ref), [`leading_boundary`](@ref), [`approximate`](@ref), [`timestep`](@ref)
and [`time_evolve`](@ref).

Algorithms in MPSKit produce genuinely different error measures, and a single bare number cannot
carry that distinction. This struct keeps the return signature uniform while naming each quantity,
so that nothing is promised that an algorithm does not actually compute.

## Convergence

  - `converged`: whether the algorithm reached its stopping criterion, or `nothing` for algorithms
    that do not iterate to a fixed point.
  - `normres`: the quantity compared against the algorithm's `tol`, or `nothing` when there is none.
    Which measure this is depends on the algorithm.
  - `numiter`: number of iterations.

## Truncation

Both truncation fields are built from the same per-factorisation quantity, namely the 2-norm of the
singular values a single local factorisation discarded, but aggregate it differently, because no
single aggregation answers every question:

  - `ϵ_max`: the largest of them. It is still a per-factorisation quantity rather than a combination
    of them, so it does not grow with system size or iteration count, which is what makes it
    comparable between runs. It is also the field a `trunc` setting most directly controls, though
    how directly depends on the strategy.
  - `ϵ_total`: all of them combined in quadrature, ``\\sqrt{\\sum_k \\epsilon_k^2}``. This grows 
    with system size and iteration count, so unlike `ϵ_max` it is not comparable between runs.
  - `numtrunc`: how many local factorisations actually discarded anything.

See [Aggregating truncation errors](@ref) for how the two relate to a `trunc` setting, and for the
per-strategy caveats.

These are `0` for an algorithm that never truncates, which does not mean the result is exact. Rather,
it means this particular error channel is absent. See the manual on [Errors and accuracy](@ref)
for what is *not* measured here.
"""
struct AlgorithmInfo{T <: Real}
    converged::Union{Bool, Nothing}
    normres::Union{T, Nothing}
    ϵ_max::T
    ϵ_total::T
    numtrunc::Int
    numiter::Int
end

"""
    AlgorithmInfo(; converged, normres, truncation, numiter)

Keyword constructor, with every field defaulting to "not produced by this algorithm": no
convergence notion, and nothing truncated. `truncation` accepts a
[`TruncationAccumulator`](@ref), or is left out when the algorithm does not truncate.
"""
function AlgorithmInfo(;
        converged = nothing, normres = nothing, truncation = nothing, numiter::Int = 1
    )
    T = _info_scalartype(normres, truncation)
    acc = isnothing(truncation) ? TruncationAccumulator(T) : truncation
    return AlgorithmInfo{T}(
        converged, isnothing(normres) ? nothing : convert(T, normres),
        convert(T, acc.ϵ_max), convert(T, sqrt(acc.ϵ_sq)), acc.numtrunc, numiter
    )
end

_info_scalartype(::Nothing, ::Nothing) = Float64
_info_scalartype(normres, ::Nothing) = float(typeof(normres))
_info_scalartype(::Nothing, acc) = _acc_type(acc)
_info_scalartype(normres, acc) = promote_type(float(typeof(normres)), _acc_type(acc))

"""
    TruncationAccumulator{T}

Collects the per-factorisation truncation errors of a sweep. Algorithms push errors in as they are
produced with [`push_error!`](@ref) and never decide how they aggregate.
The latter is [`AlgorithmInfo`](@ref)'s job.
"""
mutable struct TruncationAccumulator{T <: Real}
    ϵ_max::T
    ϵ_sq::T
    numtrunc::Int
end
function TruncationAccumulator(::Type{T}) where {T <: Real}
    return TruncationAccumulator{T}(zero(T), zero(T), 0)
end
TruncationAccumulator(ψ) = TruncationAccumulator(_truncation_scalartype(ψ))
_truncation_scalartype(ψ) = real(scalartype(ψ))
_acc_type(::TruncationAccumulator{T}) where {T} = T

"""
    push_error!(acc::TruncationAccumulator, ϵ) -> acc

Record the error of a single local factorisation. A factorisation that discarded nothing
(a QR gauge, or a truncation that kept everything) is not counted.
"""
function push_error!(acc::TruncationAccumulator, ϵ)
    iszero(ϵ) && return acc
    acc.ϵ_max = max(acc.ϵ_max, ϵ)
    acc.ϵ_sq += ϵ^2
    acc.numtrunc += 1
    return acc
end

# combining infos follows the same rules as combining the per-bond errors within one of them:
# worst case for `ϵ_max`, sum of squares for `ϵ_total`, and the later convergence verdict wins
# assumes type stability of the scalars and an ordering of receiving this info
function _combine(a::AlgorithmInfo, b::AlgorithmInfo)
    return AlgorithmInfo(
        b.converged, b.normres,
        max(a.ϵ_max, b.ϵ_max), sqrt(a.ϵ_total^2 + b.ϵ_total^2),
        a.numtrunc + b.numtrunc, a.numiter + b.numiter
    )
end

function Base.show(io::IO, ::MIME"text/plain", info::AlgorithmInfo)
    println(io, "AlgorithmInfo:")
    if !isnothing(info.converged)
        println(io, "  converged = ", info.converged, " after ", info.numiter, " iterations")
    else
        println(io, "  ", info.numiter, " iteration", info.numiter == 1 ? "" : "s")
    end
    isnothing(info.normres) || println(io, "  normres   = ", info.normres)
    if info.numtrunc > 0
        println(io, "  ϵ_max     = ", info.ϵ_max, "\t(largest single factorization)")
        println(io, "  ϵ_total   = ", info.ϵ_total, "\t(quadrature over ", info.numtrunc, ")")
    else
        println(io, "  no truncation")
    end
    return nothing
end
function Base.show(io::IO, info::AlgorithmInfo)
    return print(
        io, "AlgorithmInfo(converged = ", info.converged, ", normres = ", info.normres,
        ", ϵ_max = ", info.ϵ_max, ", ϵ_total = ", info.ϵ_total,
        ", numtrunc = ", info.numtrunc, ", numiter = ", info.numiter, ")"
    )
end
