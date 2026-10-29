println("
-----------------------------
|   Groundstate Algorithms  |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.groundstate_finite(Vector; fast = fast_tests)
TestSuite.groundstate_grassmann_preconditioner(Vector; fast = fast_tests)
TestSuite.groundstate_infinite(Vector; fast = fast_tests)
TestSuite.groundstate_lazysum_finite(Vector; fast = fast_tests)
TestSuite.groundstate_lazysum_infinite(Vector; fast = fast_tests)
TestSuite.groundstate_leading_boundary(Vector; fast = fast_tests)
