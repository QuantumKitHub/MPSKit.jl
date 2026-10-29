println("
-----------------------------
|   Finite temperature       |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.statmech_finite_temperature(Vector; fast = fast_tests)
