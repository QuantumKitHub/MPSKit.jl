println("
-----------------------------
|   Correlators & Entropy    |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.algorithms_correlators_entropy(Vector; fast = fast_tests)
TestSuite.algorithms_correlators_expectation_value(Vector; fast = fast_tests)
