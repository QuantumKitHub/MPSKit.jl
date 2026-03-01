using ParallelTestRunner
using MPSKit

# Start with autodiscovered tests
testsuite = find_tests(@__DIR__)

# remove setup code and add as init
filter!(!(startswith("utilities") ∘ first), testsuite)
init_code = quote
    include($(joinpath(@__DIR__, "utilities", "testsetup.jl")))
    using .TestSetup
end

# only run CUDA if on buildkite
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
is_buildkite && filter!(startswith("cuda") ∘ first, testsuite)

# only run CUDA/cuTENSOR if available
using CUDA, cuTENSOR
(CUDA.functional() && cuTENSOR.has_cutensor()) ||
    filter!(!(startswith("cuda") ∘ first), testsuite)

# run tests
args = parse_args(ARGS)
runtests(MPSKit, args; testsuite, init_code)
