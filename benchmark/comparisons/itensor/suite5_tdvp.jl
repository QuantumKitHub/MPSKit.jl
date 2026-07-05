# Suite 5 (docs/IMPROVEMENT_PLAN.md §4.2, item 5), ITensorMPS.jl side: TDVP throughput.
# Mirrors `benchmark/suites/suite5_tdvp.jl` — global quench of the spin-1 Heisenberg chain
# (no symmetry) from the Néel product state, two-phase protocol:
#   phase 1 (grow):    two-site TDVP (`nsite = 2`) for `nsteps_grow` steps, growing the
#                      bond dimension from 1 to the χ cap (`maxdim = χ`, `cutoff = 0.0` —
#                      rank-only truncation, matching MPSKit's `truncrank(χ)`).
#   phase 2 (measure): single-site TDVP (`nsite = 1`) at fixed bond dimensions for
#                      `nsteps_measure` steps — the throughput measurement.
#
# API verified against the installed ITensorMPS 0.4.1 source:
#   * `tdvp(operator, t, init; time_step, nsite, maxdim, cutoff, (sweep_observer!), ...)`
#     computes exp(t * operator) * init (src/solvers/tdvp.jl), so REAL-time evolution by
#     exp(-i T H) is `t = -im * T`, `time_step = -im * dt` — the exact counterpart of
#     MPSKit's `timestep` convention dt′ = -im * dt.
#   * one tdvp "sweep" = one time step; `update_observer!(sweep_observer!; state, sweep,
#     current_time, ...)` fires once per step and forwards to `measure!` for an
#     `AbstractObserver` (src/solvers/alternating_update.jl, src/update_observer.jl).
#   * `MPS(ComplexF64, sites, states)` product-state constructor (src/mps.jl:411): the
#     state is built complex explicitly, same as the MPSKit side.
#   * `expect(psi, "Sz"; sites = mid:mid)` (src/mps.jl:997).
#
# Integrator parity: both libraries implement the standard symmetric second-order
# one-timestep TDVP sweep (ITensorMPS `reverse_step = true` + `order = 2` defaults;
# MPSKit's timestep! does the analogous forward/backward half-step sweeps).
# <!-- REVIEW: integrator-order parity asserted from reading both sources; a physicist
# should confirm the two sweeps are the same integrator before publishing physics off
# these trajectories (for the timing comparison the workload is matched either way). -->
module ITensorSuite5TDVP

using ITensors
using ITensorMPS
using Dates

include(joinpath(@__DIR__, "common.jl"))
using .ITensorBenchCommon

"""
    heisenberg_opsum(N; J) -> OpSum

Spin-1 Heisenberg `OpSum`, H = J·Σ_j S_j·S_{j+1} (identical to suites 1-2).
"""
function heisenberg_opsum(N::Int; J::Real = 1.0)
    os = OpSum()
    for j in 1:(N - 1)
        os += J, "Sz", j, "Sz", j + 1
        os += J / 2, "S+", j, "S-", j + 1
        os += J / 2, "S-", j, "S+", j + 1
    end
    return os
end

# Per-step recorder: `update_observer!` forwards its kwargs (state, sweep, current_time,
# ...) to `measure!` for AbstractObserver subtypes. Physical time is reconstructed as
# t_offset + step * dt from the observer's own counter (current_time is complex here),
# matching the MPSKit-side recorder exactly.
mutable struct TdvpTrajectoryObserver <: AbstractObserver
    t0::UInt64
    dt::Float64
    t_offset::Float64
    mid::Int
    step::Int
    records::Vector{Dict{String, Any}}
end
function TdvpTrajectoryObserver(t0, dt, t_offset, mid)
    return TdvpTrajectoryObserver(t0, dt, t_offset, mid, 0, Dict{String, Any}[])
end

function ITensorMPS.measure!(o::TdvpTrajectoryObserver; kwargs...)
    haskey(kwargs, :state) || return nothing   # fires only for the per-step (sweep) hook
    state = kwargs[:state]
    o.step += 1
    push!(
        o.records, Dict{String, Any}(
            "t" => o.t_offset + o.step * o.dt,
            "sz_mid" => real(first(expect(state, "Sz"; sites = o.mid:o.mid))),
            "chi" => maxlinkdim(state),
            "walltime" => (time_ns() - o.t0) / 1.0e9,
        )
    )
    return nothing
end

"""
    tdvp_phase(H, psi, χ, nsite, nsteps, dt, t0, t_offset, mid) -> (psi, records)

Run one phase (`nsite` ∈ {1, 2}) of `nsteps` steps of size `dt` and return the evolved
state plus the per-step records. The ⟨Sz⟩ measurement runs inside the timed region, as on
the MPSKit side (identical extra work on both sides).
"""
function tdvp_phase(H, psi, χ::Int, nsite::Int, nsteps::Int, dt::Real, t0, t_offset, mid)
    nsteps == 0 && return psi, Dict{String, Any}[]
    obs = TdvpTrajectoryObserver(t0, Float64(dt), Float64(t_offset), mid)
    psi = tdvp(
        H, -im * (nsteps * dt), psi;
        time_step = -im * dt, nsite = nsite,
        maxdim = χ, cutoff = 0.0, outputlevel = 0,
        # Krylov-exponentiation tolerance parity: MPSKit's TDVP integrator defaults to
        # tol = 1e-10; ITensorMPS's exponentiate_updater would inherit KrylovKit's
        # tighter default (1e-12), i.e. strictly MORE Krylov work per site update.
        # Matching 1e-10 keeps the local-solver work comparable and favors ITensor
        # relative to its own default. (Both sides use KrylovKit.exponentiate with
        # krylovdim = 30 underneath.)
        updater_kwargs = (; tol = 1.0e-10),
        (sweep_observer!) = obs,
    )
    return psi, obs.records
end

"""
    run(; N, chis, dt, nsteps_measure, J = 1.0, resultsdir = results_dir())

Run the suite-5 protocol with ITensorMPS and write a timestamped JSON result file (shared
schema, `"library" => "ITensorMPS"`) to `resultsdir`. Returns the path.
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, dt::Real, nsteps_measure::Int,
        J::Real = 1.0,
        resultsdir::AbstractString = results_dir(),
    )
    sites = siteinds("S=1", N)
    H = MPO(heisenberg_opsum(N; J = J), sites)
    neel = [isodd(n) ? "Up" : "Dn" for n in 1:N]
    mid = N ÷ 2
    d = 3   # spin-1

    # warmup: JIT-compile both phases once at a tiny size (methodology guardrail §4.3)
    let Nw = 6, sw = siteinds("S=1", Nw)
        Hw = MPO(heisenberg_opsum(Nw; J = J), sw)
        pw = MPS(ComplexF64, sw, [isodd(n) ? "Up" : "Dn" for n in 1:Nw])
        t0w = time_ns()
        pw, _ = tdvp_phase(Hw, pw, 4, 2, 2, dt, t0w, 0.0, 3)
        tdvp_phase(Hw, pw, 4, 1, 2, dt, t0w, 2dt, 3)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        nsteps_grow = ceil(Int, log(d, χ)) + 2   # same growth budget as the MPSKit side
        psi = MPS(ComplexF64, sites, neel)       # deterministic product state: no RNG
        t0 = time_ns()
        psi, rec_grow = tdvp_phase(H, psi, χ, 2, nsteps_grow, dt, t0, 0.0, mid)
        psi, rec_meas = tdvp_phase(H, psi, χ, 1, nsteps_measure, dt, t0, nsteps_grow * dt, mid)
        total = (time_ns() - t0) / 1.0e9

        meas_wall = last(rec_meas)["walltime"] - last(rec_grow)["walltime"]
        throughput = meas_wall / (nsteps_measure * dt)
        chi_actual = maxlinkdim(psi)

        @info "suite 5 (ITensorMPS): χ = $χ done" chi_actual nsteps_grow final_sz = last(rec_meas)["sz_mid"] seconds_per_unit_time = throughput total_time = total

        push!(
            trials, Dict{String, Any}(
                "chi_target" => χ,
                "chi_actual" => chi_actual,
                "nsteps_grow" => nsteps_grow,
                "nsteps_measure" => nsteps_measure,
                "seconds_per_unit_time" => throughput,
                "trajectory_grow" => rec_grow,
                "trajectory_measure" => rec_meas,
            )
        )
    end

    data = collect_metadata()
    data["suite"] = "5-tdvp"
    data["description"] = "TDVP throughput, global quench from Néel state, spin-1 Heisenberg chain, no symmetry (ITensorMPS)"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "Trivial"
    data["algorithm"] = "tdvp nsite=2 grow, then nsite=1 measure (maxdim=chi, cutoff=0)"
    data["eltype"] = "ComplexF64"
    data["N"] = N
    data["J"] = J
    data["spin"] = 1
    data["dt"] = dt
    data["chi_schedule"] = collect(chis)
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "itensor_suite5_tdvp_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
