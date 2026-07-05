# Entry point for the MPSKit competitive-benchmark harness (docs/IMPROVEMENT_PLAN.md §4).
#
# Usage:
#   julia --project=benchmark benchmark/run.jl [--smoke] [--suite=1,2,5,7] [--blas-threads=n]
#
# --smoke     use a small chi schedule and system size so the whole harness runs in a few
#             minutes; verifies the pipeline works, NOT a real benchmark result.
# --suite=... comma-separated list of suites to run (default: 1 and 2).
#             1 = finite DMRG (trivial), 2 = finite DMRG (U(1)), 5 = TDVP throughput,
#             7 = thread scaling (one grid point per invocation; see benchmark/slurm/).
# --blas-threads=n  BLAS thread count, applied before any suite runs (default: 1, so a
#             plain invocation is single-threaded BLAS by construction). Suite 7 exists
#             to scan this together with JULIA_NUM_THREADS.
#
# See `benchmark/comparisons/` for the competitor scripts and `benchmark/plot_results.jl`
# to render the results this script produces.

const HERE = @__DIR__

using LinearAlgebra
using ThreadPinning

function parse_cli(args)
    smoke = "--smoke" in args
    suite_idx = findfirst(a -> startswith(a, "--suite="), args)
    suites = if isnothing(suite_idx)
        [1, 2]
    else
        value = split(args[suite_idx], "=", limit = 2)[2]
        parse.(Int, split(value, ","))
    end
    blas_idx = findfirst(a -> startswith(a, "--blas-threads="), args)
    blas_threads = isnothing(blas_idx) ? 1 :
        parse(Int, split(args[blas_idx], "=", limit = 2)[2])
    return (; smoke, suites, blas_threads)
end

"""
    pin_cores(blas_threads)

Pin Julia threads 1:1 to distinct cores inside the affinity mask this process was
launched with (`taskset` in `benchmark/slurm/run_all.sbatch`, or Slurm's cpu mask), via
ThreadPinning's `:affinitymask` policy. When the OpenBLAS pool has real workers
(`blas_threads > 1`), pin those to mask cores disjoint from the Julia threads.
No-oversubscription is the launcher's job: every concurrent process gets a disjoint mask.
Pinning failure is a warning, not an error, so smoke runs on unsupported platforms still
work — but cluster timings must never run unpinned, so check the log.
"""
function pin_cores(blas_threads::Int)
    try
        pinthreads(:affinitymask)
        if blas_threads > 1
            mask_ids = findall(==(1), getaffinity(; cutoff = nothing)) .- 1
            free_ids = setdiff(mask_ids, getcpuids())
            if length(free_ids) < blas_threads
                @warn "affinity mask too small to give OpenBLAS workers their own cores; they will share with Julia threads" mask_ids blas_threads
                free_ids = vcat(free_ids, mask_ids)
            end
            openblas_pinthreads(free_ids[1:blas_threads])
        end
    catch err
        @warn "thread pinning failed; timings from this run are not publishable" err
    end
    return nothing
end

opts = parse_cli(ARGS)
resultsdir = joinpath(HERE, "results")
mkpath(resultsdir)
BLAS.set_num_threads(opts.blas_threads)
pin_cores(opts.blas_threads)

println(
    "MPSKit benchmark harness — mode: ", opts.smoke ? "smoke" : "full",
    ", suites: ", opts.suites,
    ", julia threads: ", Threads.nthreads(),
    ", blas threads: ", BLAS.get_num_threads(),
)

if 1 in opts.suites
    include(joinpath(HERE, "suites", "suite1_dmrg_trivial.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256, 512]
    nsweeps = opts.smoke ? 4 : 6
    println("\n--- Suite 1: finite DMRG, no symmetry (N=$N, χ ∈ $chis, $nsweeps sweeps) ---")
    path1 = Suite1DMRGTrivial.run(; N = N, chis = chis, nsweeps = nsweeps, resultsdir = resultsdir)
    println("Suite 1 results written to: ", path1)
end

if 2 in opts.suites
    include(joinpath(HERE, "suites", "suite2_dmrg_u1.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256, 512]
    nsweeps = opts.smoke ? 4 : 10
    println("\n--- Suite 2: finite DMRG, U(1) symmetry (N=$N, χ ∈ $chis, $nsweeps sweeps) ---")
    path2 = Suite2DMRGU1.run(; N = N, chis = chis, nsweeps = nsweeps, resultsdir = resultsdir)
    println("Suite 2 results written to: ", path2)
end

if 5 in opts.suites
    include(joinpath(HERE, "suites", "suite5_tdvp.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256]
    dt = 0.05
    nsteps_measure = opts.smoke ? 4 : 20
    println("\n--- Suite 5: TDVP throughput (N=$N, χ ∈ $chis, dt=$dt, $nsteps_measure measure steps) ---")
    path5 = Suite5TDVP.run(; N = N, chis = chis, dt = dt, nsteps_measure = nsteps_measure, resultsdir = resultsdir)
    println("Suite 5 results written to: ", path5)
end

if 7 in opts.suites
    include(joinpath(HERE, "suites", "suite7_threads.jl"))
    N = opts.smoke ? 20 : 100
    chi = opts.smoke ? 16 : 256
    nsweeps = opts.smoke ? 2 : 3
    println("\n--- Suite 7: thread scaling (N=$N, χ=$chi, $nsweeps sweeps, julia=$(Threads.nthreads()), blas=$(opts.blas_threads)) ---")
    path7 = Suite7Threads.run(; N = N, chi = chi, nsweeps = nsweeps, blas_threads = opts.blas_threads, resultsdir = resultsdir)
    println("Suite 7 results written to: ", path7)
end

println("\nDone. Render plots with: julia --project=benchmark benchmark/plot_results.jl")
