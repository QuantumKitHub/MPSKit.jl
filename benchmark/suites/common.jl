# Shared utilities for the MPSKit competitive-benchmark suites (docs/IMPROVEMENT_PLAN.md §4).
#
# These suites measure *time-to-accuracy* for `find_groundstate` with `DMRG`: for a fixed
# bond dimension χ, run a fixed number of sweeps and record the energy and elapsed wall
# time after every sweep via the algorithm's `finalize` callback (see
# `src/algorithms/groundstate/dmrg.jl`, field `finalize`, signature
# `finalize(iter, ψ, H, envs) -> (ψ, envs)`). Accuracy is reported downstream (in
# `plot_results.jl`) relative to the lowest energy reached across the whole batch of runs
# in a result file, per the methodology guardrails in §4.3 (nobody publishes a
# ground-truth energy the suite itself did not produce).
module BenchCommon

using MPSKit
using MPSKitModels
using TensorKit
using Dates
using Pkg
using LinearAlgebra
using JSON

export collect_metadata, dmrg_trajectory, write_results, results_dir

"""
    results_dir() -> String

Absolute path to `benchmark/results/`, creating it if necessary.
"""
function results_dir()
    dir = normpath(joinpath(@__DIR__, "..", "results"))
    mkpath(dir)
    return dir
end

"""
    collect_metadata() -> Dict{String, Any}

Environment metadata recorded alongside every result file, so a published number is
reproducible: julia version, resolved versions of the relevant packages (from this
environment's manifest, i.e. `benchmark/Project.toml`), thread/BLAS configuration,
hostname, and timestamp.
"""
function collect_metadata()
    tracked = (
        "MPSKit", "MPSKitModels", "TensorKit", "KrylovKit", "MatrixAlgebraKit",
        "BlockTensorKit", "TensorOperations",
    )
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
    dmrg_trajectory(ψ₀, H; nsweeps, tol = 0.0, verbosity = 0) -> NamedTuple

Run single-site `DMRG` for exactly `nsweeps` sweeps (by setting `tol = 0.0` so the
Galerkin-error convergence check practically never fires early — see
`find_groundstate!(::AbstractFiniteMPS, H, ::DMRG, envs)`), recording the energy and
elapsed wall time after every sweep via the `finalize` hook.

Returns `(; ψ, envs, ϵ, energies, walltimes)` where `energies[i]`/`walltimes[i]` are the
real part of the energy and the elapsed wall time (seconds) after sweep `i`.
"""
function dmrg_trajectory(ψ₀, H; nsweeps::Int, tol::Real = 0.0, verbosity::Int = 0)
    energies = Float64[]
    walltimes = Float64[]
    t0 = time_ns()
    finalize = function (iter, ψ, H′, envs)
        push!(energies, real(expectation_value(ψ, H′, envs)))
        push!(walltimes, (time_ns() - t0) / 1.0e9)
        return ψ, envs
    end
    alg = DMRG(; tol = tol, maxiter = nsweeps, verbosity = verbosity, finalize = finalize)
    ψ, envs, ϵ = find_groundstate(ψ₀, H, alg)
    return (; ψ, envs, ϵ, energies, walltimes)
end

"""
    write_results(path, data::Dict)

Write `data` as pretty-printed JSON to `path`.
"""
function write_results(path::AbstractString, data::Dict)
    open(path, "w") do io
        JSON.print(io, data, 2)
    end
    return path
end

end # module
