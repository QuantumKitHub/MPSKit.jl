println("
------------------------------------
|   Periodic boundary conditions    |
------------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.algorithms_periodic_boundary(Vector; fast = fast_tests)
