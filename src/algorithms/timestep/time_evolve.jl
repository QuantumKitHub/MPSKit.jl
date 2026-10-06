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
- `info::AlgorithmInfo`: `numiter`, the number of steps taken, and for an algorithm that
    truncates `truncation_errors`, holding the per-bond truncation errors of every step as reported
    by [`timestep`](@ref), one vector per step. See [`AlgorithmInfo`](@ref) and
    [Time evolution accuracy](@ref) in the manual for what this does not measure.

The largest truncation error of each step is logged at `verbosity ≥ 3`, and that of the whole
evolution at `verbosity ≥ 2`.
"""
function time_evolve end, function time_evolve! end

for (timestep, time_evolve) in zip((:timestep, :timestep!), (:time_evolve, :time_evolve!))
    @eval function $time_evolve(
            ψ, H, t_span::AbstractVector{<:Number}, alg,
            envs = environments(ψ, H, ψ);
            verbosity::Int = 0, imaginary_evolution::Bool = false, normalize::Bool = false
        )
        log = IterLog(string(nameof(typeof(alg))))
        truncation_errors = []
        ϵ_max = 0.0
        with_verbosity(; verbosity) do
            @log_initialization loginit!(log, 0.0, first(t_span))
            for iter in 1:(length(t_span) - 1)
                t = t_span[iter]
                dt = t_span[iter + 1] - t

                ψ, envs, info_step = $timestep(
                    ψ, H, t, dt, alg, envs; imaginary_evolution, normalize
                )
                ψ, envs = alg.finalize(t, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

                # the log shows the largest per-bond error, or zero for a non-truncating algorithm
                ϵ_step = 0.0
                if haskey(info_step, :truncation_errors)
                    push!(truncation_errors, info_step.truncation_errors)
                    ϵ_step = Float64(maximum(info_step.truncation_errors; init = 0.0))
                end
                ϵ_max = max(ϵ_max, ϵ_step)
                @log_iteration logiter!(log, iter, ϵ_step, t)
            end
            @log_convergence logfinish!(log, length(t_span), ϵ_max, t_span[end])
        end
        info = AlgorithmInfo(;
            numiter = length(t_span) - 1,
            truncation_errors = isempty(truncation_errors) ? nothing : copy(truncation_errors)
        )
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

A finite-system step sweeps over every bond twice, and `info.truncation_errors[i]` is the
truncation error of the last cut made at bond `i`, i.e. of the second half-sweep.
The entries are non-zero only for algorithms that truncate ([`TDVP2`](@ref), [`BUG`](@ref) with a
`trunc`, and [`TDVP`](@ref) with a bond expansion), and exactly `0` for a step that happened to
discard nothing. Infinite one-site [`TDVP`](@ref) never truncates and reports no
`truncation_errors` at all. Neither case means the step was exact, but rather that this particular
source of error is either absent or idle.

See [`AlgorithmInfo`](@ref) for the entries, and [Time evolution accuracy](@ref) in the manual
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
