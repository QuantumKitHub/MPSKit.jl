println("
------------------------
|   MultilineMPS tests  |
------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.states_multilinemps(Vector; fast = fast_tests)
