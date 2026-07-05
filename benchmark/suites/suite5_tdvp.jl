# Suite 5 (docs/IMPROVEMENT_PLAN.md §4.2, item 5): TDVP throughput. Global quench on the
# spin-1 Heisenberg chain, no symmetry: start from the Néel product state and evolve in
# real time at fixed time step, recording per-step (observable, bond dimension, wall time)
# trajectories at a schedule of bond-dimension caps χ.
#
# Two-phase protocol, mirrored exactly on the ITensorMPS side
# (`benchmark/comparisons/itensor/suite5_tdvp.jl`):
#   phase 1 (grow):    two-site TDVP (`TDVP2`, `trscheme = truncrank(χ)`) for `nsteps_grow`
#                      steps — a product state has χ = 1 and single-site TDVP cannot grow
#                      the bond dimension, so two-site updates grow it to the χ cap.
#   phase 2 (measure): single-site TDVP (`TDVP`, fixed bond dimensions) for
#                      `nsteps_measure` steps — the steady-state workload whose wall time
#                      per unit physical time is the throughput headline.
# <!-- REVIEW: physics choices — Néel initial state, dt = 0.05, spin-1 Heisenberg quench,
# and the grow/measure split — are protocol choices for a *throughput* benchmark, not
# statements that this is the best way to simulate this quench. In particular the
# entanglement growth after this quench will saturate any fixed χ eventually; the
# trajectories are only physically faithful at early times, which is fine for a
# fixed-workload timing comparison but should be reviewed before any physics is read off
# these curves. -->
#
# Sign/element-type conventions (verified against src/algorithms/timestep/):
#   * `timestep`/`time_evolve` with real dt > 0 evolve by exp(-i dt H)
#     (src/algorithms/timestep/integrators.jl: dt′ = -im * dt).
#   * Real-time evolution needs a complex state; the in-place `time_evolve!` does NOT
#     promote automatically (only the copying `timestep` wrapper does), so the Néel state
#     is built as ComplexF64 directly. The ITensor side is complex for the same reason —
#     symmetric, unlike the Float64 DMRG suites.
#   * `alg.finalize(t, ψ, H, envs) -> (ψ, envs)` fires once per t_span step inside
#     `time_evolve!` — that is the per-step recording hook.
module Suite5TDVP

using MPSKit
using MPSKitModels
using TensorKit
using LinearAlgebra
using Dates

include(joinpath(@__DIR__, "common.jl"))
using .BenchCommon

"""
    neel_mps(pspace, N) -> FiniteMPS

Néel-like product state |↑↓↑↓...⟩ (maximal/minimal Sz on alternating sites) as a χ = 1
`FiniteMPS` with ComplexF64 scalars. The basis indices for "up"/"down" are read off the
diagonal of the `S_z` operator rather than hard-coded, so this cannot silently disagree
with the MPSKitModels basis convention.
"""
function neel_mps(N::Int; spin::Real = 1)
    Sz = S_z(ComplexF64, Trivial; spin = spin)
    szdiag = real.(diag(reshape(convert(Array, Sz), dim(domain(Sz)), dim(domain(Sz)))))
    up, dn = argmax(szdiag), argmin(szdiag)
    d = length(szdiag)
    pspace = ℂ^d
    tensors = map(1:N) do n
        data = zeros(ComplexF64, 1, d, 1)
        data[1, isodd(n) ? up : dn, 1] = 1
        return TensorMap(data, ℂ^1 ⊗ pspace ← ℂ^1)
    end
    return FiniteMPS(tensors)
end

"""
    tdvp_trajectory!(ψ, H, alg_factory, t0_ns, nsteps, dt, t_offset, mid, Sz) -> Vector

Evolve `ψ` in place for `nsteps` steps of size `dt` starting at physical time `t_offset`,
using the algorithm built by `alg_factory(finalize)`. Records one entry per step:
`(t, ⟨Sz⟩ at site mid, max bond dim, cumulative wall seconds since t0_ns)`. The
measurement cost of ⟨Sz⟩ is inside the timed region on both libraries' sides — identical
extra work, negligible (O(χ²) per step vs O(N χ³) per sweep).
"""
function tdvp_trajectory!(ψ, H, alg_factory, t0_ns, nsteps, dt, t_offset, mid, Sz)
    records = Vector{Dict{String, Any}}()
    nsteps == 0 && return ψ, records
    step = 0
    finalize = function (t, ψ′, H′, envs)
        step += 1
        push!(
            records, Dict{String, Any}(
                "t" => t_offset + step * dt,
                "sz_mid" => real(expectation_value(ψ′, mid => Sz)),
                "chi" => maximum(dim(left_virtualspace(ψ′, n)) for n in 2:length(ψ′)),
                "walltime" => (time_ns() - t0_ns) / 1.0e9,
            )
        )
        return ψ′, envs
    end
    t_span = t_offset .+ (0:nsteps) .* dt
    # MPSKit.time_evolve! is unexported but public-shaped (the exported `time_evolve` is
    # its copying wrapper); the in-place form avoids a per-step state copy + environment
    # rebuild that would pollute the throughput measurement.
    ψ, = MPSKit.time_evolve!(ψ, H, t_span, alg_factory(finalize))
    return ψ, records
end

"""
    run(; N, chis, dt, nsteps_measure, J = 1.0, spin = 1,
        resultsdir = BenchCommon.results_dir())

Run the suite-5 protocol and write a timestamped JSON result file to `resultsdir`.
Returns the path to the written file. The growth phase runs `ceil(log_d(χ)) + 2` steps
(enough two-site updates for the bond dimension to reach the cap from χ = 1).
"""
function run(;
        N::Int, chis::AbstractVector{<:Int}, dt::Real, nsteps_measure::Int,
        J::Real = 1.0, spin::Real = 1,
        resultsdir::AbstractString = BenchCommon.results_dir()
    )
    H = heisenberg_XXX(ComplexF64, Trivial, FiniteChain(N); J = J, spin = spin)
    Sz = S_z(ComplexF64, Trivial; spin = spin)
    mid = N ÷ 2
    d = Int(2 * spin + 1)

    alg2(χ) = fin -> TDVP2(; trscheme = truncrank(χ), finalize = fin)
    alg1 = fin -> TDVP(; finalize = fin)

    # warmup: JIT-compile both phases once at a tiny size (methodology guardrail §4.3)
    let Hw = heisenberg_XXX(ComplexF64, Trivial, FiniteChain(6); J = J, spin = spin)
        ψw = neel_mps(6; spin = spin)
        t0w = time_ns()
        ψw, _ = tdvp_trajectory!(ψw, Hw, alg2(4), t0w, 2, dt, 0.0, 3, Sz)
        tdvp_trajectory!(ψw, Hw, alg1, t0w, 2, dt, 2dt, 3, Sz)
    end

    trials = Vector{Dict{String, Any}}()
    for χ in chis
        nsteps_grow = ceil(Int, log(d, χ)) + 2
        ψ = neel_mps(N; spin = spin)   # deterministic product state: no RNG in this suite
        t0 = time_ns()
        ψ, rec_grow = tdvp_trajectory!(ψ, H, alg2(χ), t0, nsteps_grow, dt, 0.0, mid, Sz)
        ψ, rec_meas = tdvp_trajectory!(ψ, H, alg1, t0, nsteps_measure, dt, nsteps_grow * dt, mid, Sz)
        total = (time_ns() - t0) / 1.0e9

        # throughput: wall seconds per unit physical time, measured over phase 2 only
        meas_wall = last(rec_meas)["walltime"] - last(rec_grow)["walltime"]
        throughput = meas_wall / (nsteps_measure * dt)
        chi_actual = maximum(dim(left_virtualspace(ψ, n)) for n in 2:N)

        @info "suite 5: χ = $χ done" chi_actual nsteps_grow final_sz = last(rec_meas)["sz_mid"] seconds_per_unit_time = throughput total_time = total

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
    data["description"] = "TDVP throughput, global quench from Néel state, spin-1 Heisenberg chain, no symmetry"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "Trivial"
    data["algorithm"] = "TDVP2(trscheme = truncrank(chi)) grow, then TDVP measure"
    data["eltype"] = "ComplexF64"
    data["N"] = N
    data["J"] = J
    data["spin"] = spin
    data["dt"] = dt
    data["chi_schedule"] = collect(chis)
    data["trials"] = trials

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "suite5_tdvp_N$(N)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
