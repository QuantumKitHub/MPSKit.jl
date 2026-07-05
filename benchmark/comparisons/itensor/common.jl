# Shared utilities for the ITensorMPS.jl side of the MPSKit competitive benchmark
# (docs/IMPROVEMENT_PLAN.md §4). This mirrors, as closely as the two libraries allow,
# `benchmark/suites/common.jl` on the MPSKit side: same time-to-accuracy protocol, same
# per-sweep (energy, wall time) trajectory recording, and the same JSON result schema
# (with an added "library" field). Accuracy is reported downstream by
# `benchmark/plot_results.jl` relative to the lowest energy in a result file — no
# ground-truth energy is baked in here.
#
# API is verified against the official ITensorMPS docs (URLs cited next to each call):
#   - dmrg / observer keyword:   https://docs.itensor.org/ITensorMPS/stable/DMRG.html
#   - AbstractObserver interface: https://docs.itensor.org/ITensorMPS/stable/Observer.html
#   - random_mps / maxlinkdim:    https://docs.itensor.org/ITensorMPS/stable/MPSandMPO.html
module ITensorBenchCommon

using ITensors
using ITensorMPS
using LinearAlgebra
using Dates
using Pkg
using JSON

export TrajectoryObserver, collect_metadata, dmrg_trajectory, write_results, results_dir

"""
    results_dir() -> String

Absolute path to `benchmark/results/` (shared with the MPSKit side), creating it if necessary.
This directory is two levels up from `benchmark/comparisons/itensor/`.
"""
function results_dir()
    dir = normpath(joinpath(@__DIR__, "..", "..", "results"))
    mkpath(dir)
    return dir
end

# --- per-sweep trajectory recorder -------------------------------------------------
#
# ITensorMPS's observer hook (https://docs.itensor.org/ITensorMPS/stable/Observer.html):
# `measure!(o; kwargs...)` is invoked at *every* site step; the kwarg `sweep_is_done` is
# `true` exactly once per full sweep (at the end of the return half-sweep). We record the
# energy and cumulative wall time only at that point, giving one (energy, walltime) sample
# per sweep — the direct analogue of MPSKit's `finalize` hook, which fires once per sweep
# (see `benchmark/suites/common.jl`, `dmrg_trajectory`).
mutable struct TrajectoryObserver <: AbstractObserver
    t0::UInt64
    energies::Vector{Float64}
    walltimes::Vector{Float64}
    TrajectoryObserver() = new(time_ns(), Float64[], Float64[])
end

# `measure!` is passed `energy`, `sweep`, `bond`, `sweep_is_done`, `half_sweep`, ...
# per the Observer docs. `energy` is real for a real-symmetric Hamiltonian; `real(...)`
# is defensive and matches the MPSKit side, which also stores `real(...)`.
function ITensorMPS.measure!(o::TrajectoryObserver; kwargs...)
    if get(kwargs, :sweep_is_done, false)
        push!(o.energies, real(kwargs[:energy]))
        push!(o.walltimes, (time_ns() - o.t0) / 1.0e9)
    end
    return nothing
end

"""
    dmrg_trajectory(H, psi0, χ; nsweeps, outputlevel = 0) -> NamedTuple

Run ITensorMPS `dmrg` for exactly `nsweeps` sweeps at *fixed* bond dimension `χ`, recording
the energy and elapsed wall time after every sweep. Returns
`(; psi, energy, energies, walltimes)`.

Fixed-χ protocol (parity with the MPSKit run, which uses a random full-χ `FiniteMPS`,
`alg_expand = nothing`, and a non-truncating gauge so χ never changes):
  * `maxdim = fill(χ, nsweeps)` pins the ceiling to χ on every sweep.
  * `mindim = fill(χ, nsweeps)` pins the floor to χ, so the bond dimension stays at χ
    from the first sweep (the initial `psi0` is already a random full-χ MPS) instead of
    growing adaptively. This is what makes the ITensor trajectory a genuine fixed-χ
    trajectory comparable to MPSKit's. Both floors are capped by the local Hilbert-space
    dimension near the chain ends exactly as MPSKit's are.
  * `cutoff = 0.0`: no discarded-weight truncation, matching MPSKit's no-cutoff fixed-χ
    run. This is also the setting most favorable to ITensor — it never throws away weight
    to save time. (Docs: cutoff is "a float ... specifying the truncation error cutoff",
    https://docs.itensor.org/ITensorMPS/stable/DMRG.html.)
  * `noise = 0.0`: no noise term, matching MPSKit (no subspace expansion).

The clock (`t0`) is reset immediately before the `dmrg` call so the recorded wall times
are cumulative from the start of the solve, exactly as MPSKit sets `t0 = time_ns()` right
before `find_groundstate`.
"""
function dmrg_trajectory(H, psi0, χ::Int; nsweeps::Int, outputlevel::Int = 0)
    observer = TrajectoryObserver()
    maxdim = fill(χ, nsweeps)
    mindim = fill(χ, nsweeps)
    observer.t0 = time_ns()
    # dmrg(H, psi0; nsweeps, maxdim, cutoff, observer, ...) per
    # https://docs.itensor.org/ITensorMPS/stable/DMRG.html
    energy, psi = dmrg(
        H, psi0;
        nsweeps = nsweeps, maxdim = maxdim, mindim = mindim,
        cutoff = 0.0, noise = 0.0, outputlevel = outputlevel, observer = observer,
    )
    return (; psi, energy, energies = observer.energies, walltimes = observer.walltimes)
end

"""
    collect_metadata() -> Dict{String, Any}

Environment metadata recorded alongside every result file, using the SAME field names as
the MPSKit side (`benchmark/suites/common.jl`) so the two are directly comparable, plus a
`"library"` field. Records julia version, resolved ITensor package versions, thread/BLAS
configuration, hostname and timestamp.

Runs MUST use identical `JULIA_NUM_THREADS` and BLAS thread settings on both sides for the
timings to mean anything (methodology guardrail §4.3): `nthreads_julia` and `nthreads_blas`
below are recorded precisely so a reader can confirm the two runs matched.
"""
function collect_metadata()
    tracked = ("ITensorMPS", "ITensors", "NDTensors")
    package_versions = Dict{String, String}()
    for (uuid, info) in Pkg.dependencies()
        if info.name in tracked
            package_versions[info.name] = info.version === nothing ? "dev" : string(info.version)
        end
    end

    blas_config = try
        string(LinearAlgebra.BLAS.get_config())
    catch e
        "unavailable ($(sprint(showerror, e)))"
    end

    return Dict{String, Any}(
        "library" => "ITensorMPS",
        "timestamp" => string(Dates.now()),
        "julia_version" => string(VERSION),
        "package_versions" => package_versions,
        "nthreads_julia" => Threads.nthreads(),
        "nthreads_blas" => LinearAlgebra.BLAS.get_num_threads(),
        "blas_config" => blas_config,
        "hostname" => gethostname(),
    )
end

"""
    write_results(path, data::Dict)

Write `data` as pretty-printed JSON to `path` (same format as the MPSKit side).
"""
function write_results(path::AbstractString, data::Dict)
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
    return path
end

end # module
