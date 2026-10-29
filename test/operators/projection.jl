println("
---------------------------
|   Projection tests       |
---------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.operators_projection(Vector; fast = fast_tests)
