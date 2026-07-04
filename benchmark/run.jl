# Entry point for the MPSKit competitive-benchmark harness (docs/IMPROVEMENT_PLAN.md §4).
#
# Usage:
#   julia --project=benchmark benchmark/run.jl [--smoke] [--suite=1|2|1,2]
#
# --smoke     use a small chi schedule and system size so the whole harness runs in a few
#             minutes; verifies the pipeline works, NOT a real benchmark result.
# --suite=... comma-separated list of suites to run (default: both 1 and 2).
#
# This script only runs the MPSKit side of suites 1-2. See `benchmark/comparisons/` for
# the (currently empty) competitor scripts, and `benchmark/plot_results.jl` to render the
# results this script produces.

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
resultsdir = joinpath(HERE, "results")
mkpath(resultsdir)

println("MPSKit benchmark harness — mode: ", opts.smoke ? "smoke" : "full", ", suites: ", opts.suites)

if 1 in opts.suites
    include(joinpath(HERE, "suites", "suite1_dmrg_trivial.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256, 512, 1024]
    nsweeps = opts.smoke ? 4 : 30
    println("\n--- Suite 1: finite DMRG, no symmetry (N=$N, χ ∈ $chis, $nsweeps sweeps) ---")
    path1 = Suite1DMRGTrivial.run(; N = N, chis = chis, nsweeps = nsweeps, resultsdir = resultsdir)
    println("Suite 1 results written to: ", path1)
end

if 2 in opts.suites
    include(joinpath(HERE, "suites", "suite2_dmrg_u1.jl"))
    N = opts.smoke ? 20 : 100
    chis = opts.smoke ? [8, 16] : [64, 128, 256, 512, 1024]
    nsweeps = opts.smoke ? 4 : 30
    println("\n--- Suite 2: finite DMRG, U(1) symmetry (N=$N, χ ∈ $chis, $nsweeps sweeps) ---")
    path2 = Suite2DMRGU1.run(; N = N, chis = chis, nsweeps = nsweeps, resultsdir = resultsdir)
    println("Suite 2 results written to: ", path2)
end

println("\nDone. Render plots with: julia --project=benchmark benchmark/plot_results.jl")
