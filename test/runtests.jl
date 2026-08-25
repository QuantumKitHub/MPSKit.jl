using ParallelTestRunner
using MPSKit
using Pkg
using Test

Pkg.precompile()

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# remove setup code and add as init
filter!(!(startswith("setup") ∘ first), testsuite)

# only run CUDA if on buildkite
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
is_buildkite && filter!(startswith("gpu") ∘ first, testsuite)

# only run CUDA/cuTENSOR if available
using CUDA, cuTENSOR
(CUDA.functional() && cuTENSOR.functional()) ||
    filter!(!(startswith("gpu/cuda") ∘ first), testsuite)

# only run AMDGPU if available
using AMDGPU

AMDGPU.functional() ||
    filter!(!(startswith("gpu/amd") ∘ first), testsuite)


# parse arguments
args = parse_args(ARGS; custom = ["fast"])
fast = !isnothing(args.custom["fast"])

setup_path = joinpath(@__DIR__, "setup", "testsetup.jl")
init_worker_code = quote
    include($setup_path)
    using .TestSetup
    const fast_tests = $fast
end
const init_code = quote
    using ..TestSetup
    const fast_tests = $fast
end

# route tests into worker pools that stay within one category for their whole lifetime
# e.g. a worker that's already compiled FiniteMPS/DMRG machinery keeps getting fed more finite tests
# ParallelTestRunner's `test_worker` hook can't do this on its own

# infer from names or manually to which category tests belong
const FINITE_ONLY = [
    "algorithms/sector_conventions", "algorithms/dynamical_dmrg", "operators/projection", "states/arithmetic",
]
const INFINITE_ONLY = [
    "algorithms/correlators", "algorithms/periodic_boundary", "operators/dense_mpo", "states/multilinemps",
]

const MIXED = [ # some finite-infinite comparison occurs here
    "algorithms/excitations", "algorithms/fidelity_susceptibility",
]

function test_category(name)
    name in FINITE_ONLY && return :finite
    name in INFINITE_ONLY && return :infinite
    name in MIXED && return :mixed
    occursin("infinite", name) && return :infinite
    occursin("finite", name) && return :finite
    occursin("window", name) && return :window
    occursin("mixed", name) && return :mixed
    startswith(name, "gpu") && return :gpu # folder name part of test name
    startswith(name, "misc") && return :misc
    error("don't know how to categorise test '$name', add it to one of the explicit sets above")
end

const worker_phases = (:finite, :infinite, :window, :mixed, :gpu, :misc)
phase_suites = Dict(phase => Dict{String, Expr}() for phase in worker_phases)
for (name, expr) in testsuite
    push!(phase_suites[test_category(name)], name => expr)
end

# without this a user running `runtests.jl --list` would hit the first non-empty phase,
# print those test names and then exit
# here we intercept `--list` before the phase loop and print the testsuite
if args.list !== nothing
    println("Available tests:")
    for test in sort(collect(keys(testsuite)))
        println(" - $test")
    end
    exit(0)
end

failed = false
for phase in worker_phases
    suite = phase_suites[phase]
    isempty(suite) && continue
    println("\n=== Running $(phase) tests ($(length(suite)) files) ===\n")
    try
        ParallelTestRunner.runtests(MPSKit, args; testsuite = suite, init_worker_code, init_code)
    catch e
        # remaining issues:
        # 1) a single Ctrl+C stops the current phase's remaining tests but lets later phases start
        # 2) catching this exception is sensitive to ParallelTestRunner internals, might change in future versions
        e isa Test.FallbackTestSetException || rethrow()
        failed = true
        # `--quickfail` only stops the phase currently running
        # without this check, we'd just carry on into the next phase
        args.quickfail !== nothing && break
    end
end
failed && error("Test run finished with errors")
