"""
    time_evolve(ψ₀, H, t_span, alg, [envs]; kwargs...) -> (ψ, envs, info)
    time_evolve!(ψ₀, H, t_span, alg, [envs]; kwargs...) -> (ψ₀, envs, info)

Time-evolve the initial state `ψ₀` with Hamiltonian `H` over a given time span by stepping
through each of the time points obtained by iterating t_span.

# Arguments

- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t_span::AbstractVector{<:Number}`: time points over which the time evolution is stepped
- `alg`: algorithm to use for the time evolution, e.g. [`TDVP`](@ref) or [`TDVP2`](@ref).
- `envs`: MPS environment manager

# Keyword Arguments

- `verbosity::Int = 0`: verbosity level for logging
- `imaginary_evolution::Bool = false`: if true, the time evolution is done with an imaginary time step
    instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``). This can be useful to compute the
    ground state of a Hamiltonian, or to compute finite-temperature properties of a system.
- `normalize::Bool = false`: if true, the state is renormalized after every step, which can be useful
    to retain numerical stability when the norm loss is not information that is needed.

# Returns

- `ψ`: the time-stepped state
- `envs`: the updated environment manager
- `info::AlgorithmInfo`: the truncation performed over the whole evolution.
    See [`AlgorithmInfo`](@ref) and [Time evolution accuracy](@ref) in the manual
    for the difference and when to use which reported error measure,
    and [`timestep`](@ref) for what neither measures.

`ϵ_max` is logged per step at `verbosity ≥ 3` and for the whole evolution at `verbosity ≥ 2`.
The size-independent measure is used here for the same reason as the ground state algorithms.
"""
function time_evolve end, function time_evolve! end

for (timestep, time_evolve) in zip((:timestep, :timestep!), (:time_evolve, :time_evolve!))
    @eval function $time_evolve(
            ψ, H, t_span::AbstractVector{<:Number}, alg,
            envs = environments(ψ, H, ψ);
            verbosity::Int = 0, imaginary_evolution::Bool = false, normalize::Bool = false
        )
        log = IterLog(string(nameof(typeof(alg))))
        info = AlgorithmInfo(; truncation = TruncationAccumulator(ψ), numiter = 0)
        LoggingExtras.withlevel(; verbosity) do
            @infov 2 loginit!(log, 0.0, first(t_span))
            for iter in 1:(length(t_span) - 1)
                t = t_span[iter]
                dt = t_span[iter + 1] - t

                ψ, envs, info_step = $timestep(
                    ψ, H, t, dt, alg, envs; imaginary_evolution, normalize
                )
                ψ, envs = alg.finalize(t, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}
                info = _combine(info, info_step)

                # log the size-independent error measure
                @infov 3 logiter!(log, iter, convert(Float64, info_step.ϵ_max), t)
            end
            @infov 2 logfinish!(log, length(t_span), convert(Float64, info.ϵ_max), t_span[end])
        end
        return ψ, envs, info
    end
end

"""
    timestep(ψ₀, H, t, dt, alg, [envs]; kwargs...) -> (ψ, envs, info)
    timestep!(ψ₀, H, t, dt, alg, [envs]; kwargs...) -> (ψ₀, envs, info)

Time-step the state `ψ₀` with Hamiltonian `H` over a given time step `dt` at time `t`,
solving the Schroedinger equation: ``i ∂ψ/∂t = H ψ``.

# Arguments

- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t::Number`: starting time of time-step
- `dt::Number`: time-step magnitude
- `alg`: algorithm to use for the time evolution, e.g. [`TDVP`](@ref) or [`TDVP2`](@ref).
- `envs`: MPS environment manager

# Keyword Arguments

- `imaginary_evolution::Bool = false`: if true, the time evolution is done with an imaginary time step
    instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``). This can be useful to compute the
    ground state of a Hamiltonian, or to compute finite-temperature properties of a system.
- `normalize::Bool = false`: if true, the state is renormalized after every step, which can be useful
    to retain numerical stability when the norm loss is not information that is needed.

# Returns

- `ψ`: the time-stepped state
- `envs`: the updated environment manager
- `info::AlgorithmInfo`: what the step truncated (see below)

# Truncation error

A step performs many local factorisations, each discarding some weight. Rather than collapse those
into one number, `info` reports both aggregations under names that say what they are:
`info.ϵ_max` is the largest single one (size-independent, comparable against `trunc` and across
runs) and `info.ϵ_total` sums them in squares.

Both are non-zero only for algorithms that truncate ([`TDVP2`](@ref), [`BUG`](@ref) with a
`trunc`, and [`TDVP`](@ref) with a bond expansion), and are exactly `0` for one-site
[`TDVP`](@ref), which runs at fixed bond dimension. A zero here does not mean the step was exact,
but that this particular error channel is absent.

See [`AlgorithmInfo`](@ref) for the fields, and [Time evolution accuracy](@ref) in the manual
for the other error sources.

# Examples

Real-time evolution of the `|+···+⟩` product state under a transverse field `H = ∑ Zₖ`.
Each spin precesses independently, so `⟨Xₖ(t)⟩ = cos(2t)`; after a step `dt = 0.1` this is
`cos(0.2) ≈ 0.980067`. The initial state must be complex, since real-time evolution
multiplies by `-i`:

```jldoctest
julia> X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2, ℂ^2);

julia> Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2, ℂ^2);

julia> ψ₀ = FiniteMPS(ones(ComplexF64, (ℂ^2)^4));

julia> H = FiniteMPOHamiltonian(fill(ℂ^2, 4), ((i,) => Z for i in 1:4));

julia> ψ, envs = timestep(ψ₀, H, 0.0, 0.1, TDVP());

julia> round(real(expectation_value(ψ, 2 => X)); digits = 6)
0.980067
```
"""
function timestep end, function timestep! end

@doc """
    make_time_mpo(H::MPOHamiltonian, dt::Number, alg; kwargs...) -> O::MPO

Construct an `MPO` that approximates ``\\exp(-iHdt)``.

# Keyword Arguments

- `imaginary_evolution::Bool = false`: if true, the time evolution is done with an imaginary time step
    instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``). This can be useful to compute the
    ground state of a Hamiltonian, or to compute finite-temperature properties of a system.
""" make_time_mpo
