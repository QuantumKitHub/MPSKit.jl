"""
    time_evolve(ψ₀, H, t_span, [alg], [envs]; kwargs...) -> (ψ, envs)
    time_evolve!(ψ₀, H, t_span, [alg], [envs]; kwargs...) -> (ψ₀, envs)

Time-evolve the initial state `ψ₀` with Hamiltonian `H` over a given time span by stepping
through each of the time points obtained by iterating t_span.

## Arguments

- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t_span::AbstractVector{<:Number}`: time points over which the time evolution is stepped
- `[alg]`: algorithm to use for the time evolution. Defaults to [`TDVP`](@ref).
- `[envs]`: MPS environment manager

## Keyword Arguments

- `verbosity::Int=0`: verbosity level for logging
- `imaginary_evolution::Bool=false`: if true, the time evolution is done with an imaginary time step
    instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``). This can be useful for using this
    function to compute the ground state of a Hamiltonian, or to compute finite-temperature
    properties of a system.
"""
function time_evolve end, function time_evolve! end

for (timestep, time_evolve) in zip((:timestep, :timestep!), (:time_evolve, :time_evolve!))
    @eval function $time_evolve(
            ψ, H, t_span::AbstractVector{<:Number}, alg,
            envs = environments(ψ, H);
            verbosity::Int = 0, imaginary_evolution::Bool = false
        )
        log = IterLog("TDVP")
        LoggingExtras.withlevel(; verbosity) do
            @infov 2 loginit!(log, 0, t)
            for iter in 1:(length(t_span) - 1)
                t = t_span[iter]
                dt = t_span[iter + 1] - t

                ψ, envs = $timestep(ψ, H, t, dt, alg, envs; imaginary_evolution)
                ψ, envs = alg.finalize(t, ψ, H, envs)::Tuple{typeof(ψ), typeof(envs)}

                @infov 3 logiter!(log, iter, 0, t)
            end
            @infov 2 logfinish!(log, length(t_span), 0, t_span[end])
        end
        return ψ, envs
    end
end

"""
    timestep(ψ₀, H, t, dt, [alg], [envs]; kwargs...) -> (ψ, envs)
    timestep!(ψ₀, H, t, dt, [alg], [envs]; kwargs...) -> (ψ₀, envs)

Time-step the state `ψ₀` with Hamiltonian `H` over a given time step `dt` at time `t`,
solving the Schroedinger equation: ``i ∂ψ/∂t = H ψ``.

## Arguments

- `ψ₀::AbstractMPS`: initial state
- `H::AbstractMPO`: operator that generates the time evolution (can be time-dependent).
- `t::Number`: starting time of time-step
- `dt::Number`: time-step magnitude
- `[alg]`: algorithm to use for the time evolution. Defaults to [`TDVP`](@ref).
- `[envs]`: MPS environment manager

## Keyword Arguments

- `imaginary_evolution::Bool=false`: if true, the time evolution is done with an imaginary time step
    instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``). This can be useful for using this
    function to compute the ground state of a Hamiltonian, or to compute finite-temperature
    properties of a system.
"""
function timestep end, function timestep! end

@doc """
    make_time_mpo(H::MPOHamiltonian, dt::Number, alg; kwargs...) -> O::MPO

Construct an `MPO` that approximates ``\\exp(-iHdt)``.

## Keyword Arguments

- `imaginary_evolution::Bool=false`: if true, the time evolution operator is constructed
    with an imaginary time step instead, (i.e. ``\\exp(-Hdt)`` instead of ``\\exp(-iHdt)``).
    This can be useful for using this function to compute the ground state of a Hamiltonian,
    or to compute finite-temperature properties of a system.
""" make_time_mpo
