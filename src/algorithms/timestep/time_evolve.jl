"""
    time_evolve(ψ₀, H, t_span, [alg], [envs]; kwargs...)
    time_evolve!(ψ₀, H, t_span, [alg], [envs]; kwargs...)

Time-evolve the initial state `ψ₀` with Hamiltonian `H` over a given time span by stepping
through each of the time points obtained by iterating t_span.

# Arguments
- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t_span::AbstractVector{<:Number}`: time points over which the time evolution is stepped
- `[alg]`: algorithm to use for the time evolution. Defaults to [`TDVP`](@ref).
- `[envs]`: MPS environment manager
"""
function time_evolve end,
function time_evolve! end

function time_evolve(Ψ₀::AbstractFiniteMPS, H, 


"""
    SimpleScheme{F} <: Algorithm

The simplest numerical scheme to do time evolution by doing consecutive individual timesteps.

# Fields
- `stepalg::S`: algorithm used to do the individual timesteps
- `ts`::AbstractVector: time step points
- `dts`::AbstractVector: dt of each time step
- `time_finalize::F`: user-supplied function which is applied after each iteration which can be used to save objects during the time evolution, with
    signature `finalize(iter, Ψ, H, envs) -> Ψ, envs, saved`

"""
struct SimpleScheme{S,F,N} <: MPSKit.Algorithm
    stepalg::S
    ts::AbstractVector{N}
    dts::AbstractVector{N}
    time_finalize::F

    function SimpleScheme(
        stepalg::S, ts::AbstractVector{N}, dts::AbstractVector{N}, time_finalize::F
    ) where {N<:Number,S,F}
        length(ts) == length(dts) + 1 ||
            throw(ArgumentError("times and timesteps length need to be compatible"))
        all(isapprox.(test.ts[1:(end - 1)] .+ test.dts, test.ts[2:end])) ||
            throw(ArgumentError("times and timesteps need to be compatible"))
        return new{S,F,N}(stepalg, ts, dts, time_finalize)
    end
end
function SimpleScheme(stepalg, ts::AbstractVector, time_finalize)
    return SimpleScheme(stepalg, ts, ts[2:end] .- ts[1:(end - 1)], time_finalize)
end

function SimpleScheme(stepalg, ts::AbstractRange, time_finalize)
    return SimpleScheme(stepalg, ts, repeat([step(ts)], length(ts) - 1), time_finalize)
end

#implement iteration
function Base.iterate(x::SimpleScheme, state=1)
    if length(x.ts) == 0 || state == length(x.ts)
        return nothing
    else
        return ((x.ts[state], x.dts[state]), state + 1)
    end
end

"""
    time_evolve(Ψ, H, tspan, [environments]; kwargs...)
    time_evolve(Ψ, H, scheme, [environments]; kwargs...)

Time evolve the initial state `Ψ` with Hamiltonian `H` over a time span `tspan`. If not specified the
time step `scheme` defaults to `SimpleScheme` which is just a consecutive iteration over `tspan`.
If not specified, an algorithm for the individual time steps will be attempted based on the supplied keywords.

## Arguments
- `Ψ::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `[environments]`: MPS environment manager
- `tspan::AbstractVector`: time points over which the time evolution is stepped
- `scheme`: time step scheme

## Keywords
- `tol::Float64`: tolerance for convergence criterium
- `maxiter::Int`: maximum amount of iterations
- `verbose::Bool`: display progress information
"""
function time_evolve!(
    Ψ,
    H,
    tspan::AbstractVector{<:Number},
    envs::Cache=environments(Ψ, H);
    integrator=Lanczos(; tol=tol),
    tol=Defaults.tol,
    tolgauge=Defaults.tolgauge,
    gaugemaxiter=Defaults.maxiter,
    verbose=Defaults.verbose,
    trscheme=nothing,
    stepalg=TDVP(;
        integrator=integrator,
        tol=tol,
        tolgauge=tolgauge,
        gaugemaxiter=gaugemaxiter,
        verbose=verbose,
    ),
    time_finalize=Defaults._time_finalize,
    saved=[],
)
    if !isnothing(trscheme)
        stepalg = TDVP2(;
            integrator=integrator,
            tol=tol,
            tolgauge=tolgauge,
            gaugemaxiter=gaugemaxiter,
            verbose=verbose,
            trscheme=trscheme,
        )
    end
    scheme = SimpleScheme(stepalg, tspan, time_finalize)
    return time_evolve!(Ψ, H, scheme, envs; saved=saved)
end

function time_evolve(Ψ, H, tspan, envs::Cache=environments(Ψ, H); kwargs...)
    return time_evolve(copy(Ψ), H, tspan, envs; kwargs...)
end

function time_evolve!(Ψ, H, scheme::SimpleTimeScheme, envs=environments(Ψ, H); saved=[])
    _, _, tobesaved = alg.finalize(1, Ψ, H, envs)
    isnothing(tobesaved) || push!(saved, tobesaved)
    for (iter, (t, dt)) in enumerate(scheme)
        Ψ, envs = timestep!(Ψ, H, t, dt, scheme.stepalg, envs)

        Ψ, envs, tobesaved =
            scheme.finalize(iter, Ψ, H, envs)::Tuple{typeof(Ψ),typeof(envs),eltype(saved)}
        isnothing(tobesaved) || push!(saved, tobesaved)
    end
    return Ψ, envs, saved
end
function time_evolve(Ψ, H, scheme, envs::Cache=environments(Ψ, H); kwargs...)
    return time_evolve!(copy(Ψ), H, scheme, envs; kwargs...)
end
