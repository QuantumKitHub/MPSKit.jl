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
runtests(MPSKit, args; testsuite, init_code)
