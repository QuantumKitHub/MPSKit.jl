# Suite 7 (docs/IMPROVEMENT_PLAN.md §4.2, item 7): thread scaling. Runs the suite-1
# workload (finite two-site DMRG, spin-1 Heisenberg, no symmetry) at ONE representative
# (N, χ) point and records the wall-time trajectory, so that speedup vs thread count can
# be computed across runs.
#
# One process per grid point: `JULIA_NUM_THREADS` is fixed at launch and cannot be changed
# from inside Julia, so the (julia-threads x blas-threads) grid is driven from the OUTSIDE
# (see `benchmark/slurm/` and the README) by launching this suite once per grid point:
#
#     JULIA_NUM_THREADS=<nj> julia --project=benchmark benchmark/run.jl --suite=7 --blas-threads=<nb>
#
# BLAS threads CAN be set at runtime (`LinearAlgebra.BLAS.set_num_threads`), which is how
# `--blas-threads` is applied before the warmup. The result JSON records both counts in
# its metadata (`nthreads_julia`, `nthreads_blas`) — the plotting side groups on those.
#
# NOTE (methodology): MPSKit and ITensorMPS parallelize differently — MPSKit uses Julia
# threads (multithreaded environments / per-sector block operations via TensorKit and
# OhMyThreads) while ITensor's dense path parallelizes mostly through BLAS threads. The
# grid therefore scans BOTH axes on BOTH libraries; observing that the libraries peak at
# different corners of the grid is a finding, not a parity violation.
module Suite7Threads

using MPSKit
using MPSKitModels
using TensorKit
using Random
using Dates
using LinearAlgebra

include(joinpath(@__DIR__, "common.jl"))
using .BenchCommon

"""
    run(; N, chi, nsweeps, blas_threads, seed = 1234, J = 1.0, spin = 1,
        resultsdir = BenchCommon.results_dir())

Run the suite-1 workload once at bond dimension `chi` with `LinearAlgebra.BLAS` limited to
`blas_threads` threads, and write a timestamped JSON result file to `resultsdir` whose
filename carries the (julia-threads, blas-threads) pair. Returns the path.
"""
function run(;
        N::Int, chi::Int, nsweeps::Int, blas_threads::Int,
        seed::Int = 1234, J::Real = 1.0, spin::Real = 1,
        resultsdir::AbstractString = BenchCommon.results_dir()
    )
    BLAS.set_num_threads(blas_threads)

    H = heisenberg_XXX(Float64, Trivial, FiniteChain(N); J = J, spin = spin)
    pspaces = physicalspace(H)

    # warmup: JIT-compile the pipeline once at a tiny size (methodology guardrail §4.3)
    let Hw = heisenberg_XXX(Float64, Trivial, FiniteChain(6); J = J, spin = spin)
        dmrg2_trajectory(FiniteMPS(Float64, physicalspace(Hw), ℂ^4), Hw; nsweeps = 2, χ = 4)
    end

    Random.seed!(seed)
    ψ₀ = FiniteMPS(Float64, pspaces, ℂ^chi)
    elapsed = @elapsed result = dmrg2_trajectory(ψ₀, H; nsweeps = nsweeps, χ = chi)

    nj = Threads.nthreads()
    nb = BLAS.get_num_threads()
    @info "suite 7: (julia = $nj, blas = $nb) done" chi final_energy = last(result.energies) total_time = elapsed

    data = collect_metadata()
    data["suite"] = "7-threads"
    data["description"] = "thread scaling of the suite-1 workload (finite two-site DMRG, spin-1 Heisenberg, no symmetry)"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "Trivial"
    data["algorithm"] = "DMRG2(trscheme = truncrank(chi))"
    data["eltype"] = "Float64"
    data["N"] = N
    data["J"] = J
    data["spin"] = spin
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["chi_schedule"] = [chi]
    data["best_energy"] = minimum(result.energies)
    data["trials"] = [
        Dict{String, Any}(
            "chi_target" => chi,
            "chi_actual" => maximum(dim(left_virtualspace(result.ψ, n)) for n in 2:N),
            "energies" => result.energies,
            "walltimes" => result.walltimes,
            "gctimes" => result.gctimes,
            "allocd_bytes" => result.allocd,
            "final_galerkin_error" => result.ϵ,
        ),
    ]

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "suite7_threads_N$(N)_chi$(chi)_j$(nj)_b$(nb)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
