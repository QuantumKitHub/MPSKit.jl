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

args = parse_args(ARGS)
is_buildkite = get(ENV, "BUILDKITE", "false") == "true"
if is_buildkite
    empty!(testsuite)
    gpu_testsuite = find_tests(joinpath(@__DIR__, "cuda"))
    append!(testsuite, gpu_testsuite)
else
    filter!(!(startswith("cuda") ∘ first), testsuite)
end
runtests(MPSKit, args; testsuite, init_code)
