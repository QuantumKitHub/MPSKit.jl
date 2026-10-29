println("
-----------------------------
|   Approximation tests     |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.groundstate_approximate(Vector; fast = fast_tests)
