# Suite 7 (docs/IMPROVEMENT_PLAN.md §4.2, item 7), ITensorMPS.jl side: thread scaling.
# Mirrors `benchmark/suites/suite7_threads.jl` — the suite-1 workload (finite two-site
# DMRG, spin-1 Heisenberg, no symmetry) at ONE (N, χ) point, launched once per
# (julia-threads x blas-threads) grid point from the outside:
#
#     JULIA_NUM_THREADS=<nj> julia --project=benchmark/comparisons/itensor \
#         benchmark/comparisons/itensor/run.jl --suite=7 --blas-threads=<nb>
#
# NOTE (methodology): the two libraries parallelize differently. On this dense
# (no-symmetry) workload ITensor parallelizes mostly through BLAS threads; MPSKit also
# exploits Julia threads. Scanning the same grid on both sides and reporting where each
# library peaks is the finding itself, not a parity violation. ITensor's block-sparse
# threading (`ITensors.enable_threaded_blocksparse()`) is irrelevant on this dense
# workload and is left at its default.
module ITensorSuite7Threads

using ITensors
using ITensorMPS
using Random
using Dates
using LinearAlgebra

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

"""
    run(; N, chi, nsweeps, blas_threads, seed = 1234, J = 1.0, resultsdir = results_dir())

Run the suite-1 workload once at bond dimension `chi` with BLAS limited to `blas_threads`
threads, and write a timestamped JSON result file whose filename carries the
(julia-threads, blas-threads) pair. Returns the path.
"""
function run(;
        N::Int, chi::Int, nsweeps::Int, blas_threads::Int,
        seed::Int = 1234, J::Real = 1.0,
        resultsdir::AbstractString = results_dir(),
    )
    BLAS.set_num_threads(blas_threads)

    sites = siteinds("S=1", N)
    H = MPO(heisenberg_opsum(N; J = J), sites)

    # warmup: JIT-compile the pipeline once at a tiny size (methodology guardrail §4.3)
    let Nw = 6, sw = siteinds("S=1", Nw)
        Hw = MPO(heisenberg_opsum(Nw; J = J), sw)
        dmrg_trajectory(Hw, random_mps(sw; linkdims = 4), 4; nsweeps = 2)
    end

    Random.seed!(seed)
    ψ₀ = random_mps(sites; linkdims = chi)
    elapsed = @elapsed result = dmrg_trajectory(H, ψ₀, chi; nsweeps = nsweeps)

    nj = Threads.nthreads()
    nb = BLAS.get_num_threads()
    @info "suite 7 (ITensorMPS): (julia = $nj, blas = $nb) done" chi final_energy = last(result.energies) total_time = elapsed

    data = collect_metadata()
    data["suite"] = "7-threads"
    data["description"] = "thread scaling of the suite-1 workload (finite two-site DMRG, spin-1 Heisenberg, no symmetry) (ITensorMPS)"
    data["model"] = "heisenberg_XXX"
    data["symmetry"] = "Trivial"
    data["algorithm"] = "dmrg (two-site), maxdim = mindim = chi, cutoff = 0, noise = 0"
    data["eltype"] = "Float64"
    data["N"] = N
    data["J"] = J
    data["spin"] = 1
    data["nsweeps"] = nsweeps
    data["seed"] = seed
    data["chi_schedule"] = [chi]
    data["best_energy"] = minimum(result.energies)
    data["trials"] = [
        Dict{String, Any}(
            "chi_target" => chi,
            "chi_actual" => maxlinkdim(result.psi),
            "energies" => result.energies,
            "walltimes" => result.walltimes,
            "gctimes" => result.gctimes,
            "allocd_bytes" => result.allocd,
            "final_galerkin_error" => nothing,
        ),
    ]

    timestamp = Dates.format(Dates.now(), "yyyymmdd-HHMMSS")
    path = joinpath(resultsdir, "itensor_suite7_threads_N$(N)_chi$(chi)_j$(nj)_b$(nb)_$(timestamp).json")
    write_results(path, data)
    return path
end

end # module
