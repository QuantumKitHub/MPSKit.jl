# Entry point for the ITensorMPS.jl side of the MPSKit competitive benchmark
# (docs/IMPROVEMENT_PLAN.md §4). Mirror of `benchmark/run.jl` on the MPSKit side: same CLI
# flags, same N / χ schedules, results written into the shared `benchmark/results/`.
#
# Usage (from the repo root):
#   julia --project=benchmark/comparisons/itensor benchmark/comparisons/itensor/run.jl [--smoke] [--suite=1|2|1,2]
#
# --smoke     N = 20, χ ∈ [8, 16], 4 sweeps — a few-minute pipeline check, NOT a result.
# --suite=... comma-separated list of suites to run (default: both 1 and 2).
#
# IMPORTANT (methodology guardrail §4.3): for the timings to be comparable, this MUST be
# run with the SAME JULIA_NUM_THREADS and the SAME BLAS thread count as the MPSKit side
# (`julia --project=benchmark benchmark/run.jl`). The recorded metadata reports both so a
# reader can verify the two runs matched.

const HERE = @__DIR__

function parse_cli(args)
    smoke = "--smoke" in args
    suite_idx = findfirst(a -> startswith(a, "--suite="), args)
    suites = if isnothing(suite_idx)
        [1, 2]
    else
        value = split(args[suite_idx], "=", limit = 2)[2]
        parse.(Int, split(value, ","))
    end
    return (; smoke, suites)
end

opts = parse_cli(ARGS)
resultsdir = normpath(joinpath(HERE, "..", "..", "results"))
mkpath(resultsdir)

println("ITensorMPS benchmark harness — mode: ", opts.smoke ? "smoke" : "full", ", suites: ", opts.suites)

if 1 in opts.suites
    include(joinpath(HERE, "suite1_dmrg_trivial.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256, 512, 1024]
    nsweeps = opts.smoke ? 4 : 30
    println("\n--- Suite 1: finite DMRG, no symmetry (N=$N, χ ∈ $chis, $nsweeps sweeps) ---")
    path1 = ITensorSuite1DMRGTrivial.run(; N = N, chis = chis, nsweeps = nsweeps, resultsdir = resultsdir)
    println("Suite 1 results written to: ", path1)
end

if 2 in opts.suites
    include(joinpath(HERE, "suite2_dmrg_u1.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256, 512, 1024]
    nsweeps = opts.smoke ? 4 : 30
    println("\n--- Suite 2: finite DMRG, U(1) symmetry (N=$N, χ ∈ $chis, $nsweeps sweeps) ---")
    path2 = ITensorSuite2DMRGU1.run(; N = N, chis = chis, nsweeps = nsweeps, resultsdir = resultsdir)
    println("Suite 2 results written to: ", path2)
end

println("\nDone. These files share the schema of the MPSKit-side results in benchmark/results/.")
