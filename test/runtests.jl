using ParallelTestRunner
using MPSKit
using Pkg

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

ParallelTestRunner.runtests(MPSKit, args; testsuite, init_worker_code, init_code)
