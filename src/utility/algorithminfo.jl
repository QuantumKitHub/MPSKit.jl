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

  - `truncation_errors`: the truncation error of every bond, i.e. the 2-norm of the singular values
    discarded by the most recent factorisation of that bond. Entry `i` is the bond between sites
    `i` and `i + 1`, so a finite MPS of length `L` has `L - 1` entries and an infinite unit cell of
    length `L` has `L`, the last one being the bond across the unit cell boundary. A `Multiline`
    algorithm reports a matrix instead, indexed as `[row, bond]`.

A sweep that cuts a bond more than once keeps only the last cut, which is what the returned state
still throws away there. An algorithm that truncates reports this entry even when it discarded
nothing (all entries zero), whereas the entry being absent means the algorithm never truncates.
Neither says the result is exact. See the manual on [Errors and accuracy](@ref) for what is *not*
measured here.
"""
struct AlgorithmInfo
    data::Dict{Symbol, Any}
end

"""
    AlgorithmInfo(; kwargs...)

Build an [`AlgorithmInfo`](@ref) from the entries an algorithm actually produced. Every keyword
becomes an entry.

A keyword whose value is `nothing` is omitted rather than stored. This is how an algorithm
reports a quantity it computes only on some branches: write `galerkin = measured ? g : nothing`
to leave the entry out where there is nothing to report, instead of assembling a different keyword
set per branch. A missing entry is an error to read, so pass `nothing` only where absence is
the meaning you intend.

`numiter` defaults to `1` for the single-shot algorithms.
"""
function AlgorithmInfo(; kwargs...)
    data = Dict{Symbol, Any}()
    for (key, value) in kwargs
        isnothing(value) || (data[key] = value)
    end
    get!(data, :numiter, 1)
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

# the single-line wrappers of the `Multiline` algorithms report a vector instead of a 1-row matrix
function _singleline_info(info::AlgorithmInfo)
    data = copy(getfield(info, :data))
    haskey(data, :truncation_errors) && (data[:truncation_errors] = vec(data[:truncation_errors]))
    return AlgorithmInfo(data)
end

# dictionary interface
Base.getindex(info::AlgorithmInfo, key::Symbol) = getfield(info, :data)[key]
Base.haskey(info::AlgorithmInfo, key::Symbol) = haskey(getfield(info, :data), key)
Base.get(info::AlgorithmInfo, key::Symbol, default) = get(getfield(info, :data), key, default)
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
    haskey(data, key) && return data[key]
    return _no_entry_error(info, key)
end

@noinline function _no_entry_error(info::AlgorithmInfo, key::Symbol)
    reported = join(propertynames(info), ", ")
    msg = "this AlgorithmInfo has no entry `$key`; this algorithm reports $reported."
    throw(ArgumentError(msg))
end

# custom show
# convergence measures are displayed first, followed by everything else in alphabetical order
function Base.show(io::IO, ::MIME"text/plain", info::AlgorithmInfo)
    data = getfield(info, :data)
    println(io, "AlgorithmInfo:")
    numiter = get(data, :numiter, nothing)
    tab_space = "  "

    if haskey(data, :converged)
        println(
            io, tab_space, rpad("converged", 18), " = ", data[:converged],
            " after ", numiter, " iterations"
        )
    elseif !isnothing(numiter)
        println(io, tab_space, numiter, " iteration", numiter == 1 ? "" : "s")
    end

    handled = (:converged, :numiter, convergence_keys...)
    others = sort!(filter(∉(handled), collect(keys(data))))
    compact = IOContext(io, :compact => true, :limit => true)
    for key in (filter(in(keys(data)), convergence_keys)..., others...)
        print(io, tab_space, rpad(string(key), 18), " = ")
        show(compact, data[key])
        println(io)
    end
    return nothing
end

function Base.show(io::IO, info::AlgorithmInfo)
    data = getfield(info, :data)
    print(io, "AlgorithmInfo(")
    join(io, (string(key, " = ", data[key]) for key in sort!(collect(keys(data)))), ", ")
    return print(io, ")")
end
