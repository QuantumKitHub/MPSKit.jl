println("
------------------------
|   InfiniteMPS tests   |
------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.states_infinitemps_basics(Vector; fast = fast_tests)
TestSuite.states_infinitemps_copying(Vector; fast = fast_tests)
TestSuite.states_infinitemps_adapt(Vector; fast = fast_tests)
TestSuite.states_infinitemps_entropy(Vector; fast = fast_tests)
TestSuite.states_infinitemps_copying_default(Vector; fast = fast_tests)
