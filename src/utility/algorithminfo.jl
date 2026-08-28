"""
$(TYPEDEF)

Information about how an algorithm arrived at its result, returned as the last value by
[`find_groundstate`](@ref), [`leading_boundary`](@ref), [`approximate`](@ref), [`timestep`](@ref)
and [`time_evolve`](@ref).

Algorithms in MPSKit produce genuinely different measures, and not every algorithm even has access
to the same information. To avoid reporting two different quantities under one name, the information is
carried in a `Dict{Symbol, Any}` that each algorithm fills with only the entries it actually
computes.

Entries are read as properties (`info.galerkin`), by indexing (`info[:galerkin]`), or through the
usual dictionary interface (`keys`, `haskey`, `get`, `pairs`, `length`). Asking for an entry the
algorithm never reported is an error that names what it did report, rather than a silent
`nothing`. Displaying the object (or calling `keys(info)`) shows what a given algorithm actually produced.

## The vocabulary

The keys below are the ones currently in use. Each algorithm's own docstring states which of them
it reports. Nothing prevents an algorithm from adding its own.

### Convergence

  - `converged::Bool`: whether the algorithm met its stopping criterion.
  - `numiter::Int`: number of iterations (sweeps or steps).

The quantity that was compared against the algorithm's `tol` is stored under a name that says
which measure it is:

  - `galerkin`: the Galerkin error, i.e. the maximum over sites of the local update projected onto
    the orthogonal complement of the current tensor. Reported by [`DMRG`](@ref), [`DMRG2`](@ref),
    [`VUMPS`](@ref) and [`VOMPS`](@ref) when solving for a state.
  - `gradientnorm`: the norm of the Riemannian (Grassmann) gradient, as supplied by the optimiser.
    Reported by [`GradientGrassmann`](@ref).
  - `bondresidual`: the change in the center bond tensor over a sweep. This is a fixed-point
    residual: it says the sweeps have stopped moving, which is weaker than saying the state is
    variationally stationary. Reported by [`IDMRG`](@ref) and [`IDMRG2`](@ref).
  - `localchange`: the largest relative change of a local tensor over a sweep. Reported by
    [`DMRG`](@ref) and [`DMRG2`](@ref) inside [`approximate`](@ref).

[`convergence_measure`](@ref) returns whichever of these is present, for code that only wants
"the number that was compared against `tol`" without caring which one it is.

### Truncation

Both truncation entries are built from the same per-factorisation quantity, namely the 2-norm of
the singular values a single local factorisation discarded, but aggregate it differently, because
no single aggregation answers every question:

  - `max_truncation_error`: the largest of them. It is still a per-factorisation quantity rather
    than a combination of them, so it does not grow with system size or iteration count, which is
    what makes it comparable between runs. It is also the entry a `trunc` setting most directly
    controls, though how directly depends on the strategy.
  - `total_truncation_error`: all of them combined in quadrature,
    ``\\sqrt{\\sum_k \\epsilon_k^2}``. This grows with system size and iteration count, so unlike
    `max_truncation_error` it is not comparable between runs.
  - `numtrunc`: how many of the recorded errors were non-zero, i.e. how many actually discarded
    anything.

The two error entries also read under the short aliases `ϵ_max` and `ϵ_total` (`info.ϵ_max`,
`info[:ϵ_total]`, `haskey(info, :ϵ_max)`). They are only ever stored under the descriptive names,
so `keys` and displaying/showing them returns one name per quantity.

Which factorisations get recorded is not the same for every algorithm, and this is worth knowing
before comparing `numtrunc` (or `total_truncation_error`) between them:

  - [`IDMRG`](@ref), [`IDMRG2`](@ref), [`TDVP2`](@ref), [`BUG`](@ref) and [`Zipup`](@ref) record
    every factorisation as it happens, so `numtrunc` is a count of factorisations. A sweep that
    visits a bond twice contributes twice.
  - [`DMRG`](@ref) and [`DMRG2`](@ref) instead keep one slot per update position, overwritten as
    the sweep passes, and record those slots once at the end. `numtrunc` is therefore the number of
    positions whose most recent cut discarded something . This is never more than `length(ψ)` for
    [`DMRG`](@ref) or `length(ψ) - 1` for [`DMRG2`](@ref), however many sweeps ran and however many
    SVDs each performed.

The sweeping choice is deliberate: what the returned state still throws away at a bond is the last
cut made there, not the sum of every cut ever made there. It does mean `numtrunc` counts different
things in the two families, so read it as "how many recorded errors were non-zero" rather than as a
tally of SVD calls.

See [Aggregating truncation errors](@ref) for how the two relate to a `trunc` setting, and for the
per-strategy caveats.

An algorithm that truncates reports all three even on a run where it happened to discard nothing,
so `max_truncation_error == 0` means "truncated, but cut nothing away", whereas the entries being
absent altogether means the algorithm never truncates. Neither says the result is exact. See the
manual on [Errors and accuracy](@ref) for what is *not* measured here.
"""
struct AlgorithmInfo
    data::Dict{Symbol, Any}
end

"""
    AlgorithmInfo(; truncation = nothing, kwargs...)

Build an [`AlgorithmInfo`](@ref) from the entries an algorithm actually produced. Every keyword
becomes an entry.

A keyword whose value is `nothing` is omitted rather than stored. This is how an algorithm
reports a quantity it computes only on some branches: write `galerkin = measured ? g : nothing`
to leave the entry out where there is nothing to report, instead of assembling a different keyword
set per branch. A missing entry is an error to read, so pass `nothing` only where absence is
the meaning you intend.

`truncation` is special: it accepts a [`TruncationAccumulator`](@ref) and expands into the
`max_truncation_error`/`total_truncation_error`/`numtrunc` entries. Leave it out for an algorithm
that does not truncate.
`numiter` defaults to `1` for the single-shot algorithms.
"""
function AlgorithmInfo(; truncation = nothing, kwargs...)
    data = Dict{Symbol, Any}()
    for (key, value) in kwargs
        isnothing(value) || (data[_canonical_key(key)] = value)
    end
    get!(data, :numiter, 1)
    if !isnothing(truncation)
        data[:max_truncation_error] = truncation.ϵ_max
        data[:total_truncation_error] = sqrt(truncation.ϵ_sq)
        data[:numtrunc] = truncation.numtrunc
    end
    return AlgorithmInfo(data)
end

# the entries holding "the number that was compared against `tol`"
const convergence_keys = (:galerkin, :gradientnorm, :bondresidual, :localchange)

"""
    convergence_measure(info::AlgorithmInfo)

The quantity that was compared against the algorithm's `tol`, whichever of
`$(join(convergence_keys, "`/`"))` the algorithm reported, or `nothing` for an algorithm that does
not iterate towards a fixed point and reports none of them.

Use this when you only want the number, and read the specific entry when the kind of measure
matters, since these are not comparable with one another.
"""
function convergence_measure(info::AlgorithmInfo)
    data = getfield(info, :data)
    for key in convergence_keys
        haskey(data, key) && return data[key]
    end
    return nothing
end

# short aliases for the two truncation entries
# entries remain stored under the descriptive name
const _key_aliases = Dict{Symbol, Symbol}(
    :ϵ_max => :max_truncation_error,
    :ϵ_total => :total_truncation_error,
)
_canonical_key(key::Symbol) = get(_key_aliases, key, key)

# dictionary interface
Base.getindex(info::AlgorithmInfo, key::Symbol) = getfield(info, :data)[_canonical_key(key)]
Base.haskey(info::AlgorithmInfo, key::Symbol) = haskey(getfield(info, :data), _canonical_key(key))
function Base.get(info::AlgorithmInfo, key::Symbol, default)
    return get(getfield(info, :data), _canonical_key(key), default)
end
Base.keys(info::AlgorithmInfo) = keys(getfield(info, :data))
Base.values(info::AlgorithmInfo) = values(getfield(info, :data))
Base.pairs(info::AlgorithmInfo) = pairs(getfield(info, :data))
Base.length(info::AlgorithmInfo) = length(getfield(info, :data))

# property sugar: asking for an entry the algorithm never reported is an
# error naming what it did report, rather than a silent `nothing`
Base.propertynames(info::AlgorithmInfo) = Tuple(sort!(collect(keys(getfield(info, :data)))))
function Base.getproperty(info::AlgorithmInfo, key::Symbol)
    key === :data && return getfield(info, :data)
    data = getfield(info, :data)
    canonical = _canonical_key(key)
    haskey(data, canonical) && return data[canonical]
    return _no_entry_error(info, canonical)
end

@noinline function _no_entry_error(info::AlgorithmInfo, key::Symbol)
    reported = join(propertynames(info), ", ")
    msg = "this AlgorithmInfo has no entry `$key`; this algorithm reports $reported."
    throw(ArgumentError(msg))
end

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
# worst case for `ϵ_max`, sum of squares for `ϵ_total`, later convergence verdict wins,
# and counts are summed
# assumes type stability of the scalars and an ordering of receiving this info
_later_wins(_, later) = later
const _combine_rules = Dict{Symbol, Any}(
    :max_truncation_error => max,
    :total_truncation_error => (a, b) -> sqrt(a^2 + b^2),
    :numtrunc => +,
    :numiter => +,
)

function _combine(a::AlgorithmInfo, b::AlgorithmInfo)
    dataₐ = copy(getfield(a, :data))
    for (key, value) in getfield(b, :data)
        dataₐ[key] = haskey(dataₐ, key) ?
            get(_combine_rules, key, _later_wins)(dataₐ[key], value) : value
    end
    return AlgorithmInfo(dataₐ)
end

# custom show
# entries are displayed in a fixed order, with anything outside the known vocabulary listed last
const _show_order = (convergence_keys..., :max_truncation_error, :total_truncation_error)
const _show_handled = (_show_order..., :converged, :numiter, :numtrunc)

function Base.show(io::IO, ::MIME"text/plain", info::AlgorithmInfo)
    data = getfield(info, :data)
    println(io, "AlgorithmInfo:")
    numiter = get(data, :numiter, nothing)
    tab_space = "  "

    if haskey(data, :converged)
        println(
            io, tab_space, rpad("converged", 22), " = ", data[:converged],
            " after ", numiter, " iterations"
        )
    elseif !isnothing(numiter)
        println(io, tab_space, numiter, " iteration", numiter == 1 ? "" : "s")
    end

    for key in _show_order
        haskey(data, key) || continue
        suffix = if key === :max_truncation_error
            "\t(largest single factorisation)"
        elseif key === :total_truncation_error
            "\t(quadrature over $(get(data, :numtrunc, 0)) truncations)"
        else
            ""
        end
        println(io, tab_space, rpad(string(key), 22), " = ", data[key], suffix)
    end
    haskey(data, :numtrunc) && data[:numtrunc] == 0 && println(io, "  no truncation")

    for key in sort!(collect(keys(data)))
        key in _show_handled && continue
        println(io, tab_space, rpad(string(key), 22), " = ", data[key])
    end
    return nothing
end

function Base.show(io::IO, info::AlgorithmInfo)
    data = getfield(info, :data)
    print(io, "AlgorithmInfo(")
    join(io, (string(key, " = ", data[key]) for key in sort!(collect(keys(data)))), ", ")
    return print(io, ")")
end
