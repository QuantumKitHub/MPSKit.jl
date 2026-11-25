println("
----------------------
|   FiniteMPS tests   |
----------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.states_finitemps_basics(Vector; fast = fast_tests)
TestSuite.states_finitemps_complex(Vector; fast = fast_tests)
TestSuite.states_finitemps_center_indexing(Vector; fast = fast_tests)
TestSuite.states_finitemps_gauging_stackoverflow(Vector; fast = fast_tests)
TestSuite.states_finitemps_copying(Vector; fast = fast_tests)
TestSuite.states_finitemps_entropy(Vector; fast = fast_tests)
