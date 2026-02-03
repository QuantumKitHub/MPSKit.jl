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

# Parse arguments
# ensure backwards compatibility by filtering out `--group=`
pat = r"(?:--group=)(\w+)"
arg_id = findfirst(contains(pat), ARGS)
!isnothing(arg_id) && (ARGS[arg_id] = lowercase(only(match(pat, ARGS[arg_id]).captures)))

args = parse_args(ARGS)
runtests(MPSKit, args; testsuite, init_code)
