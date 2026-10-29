println("
--------------------------
|   State arithmetic     |
--------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.states_arithmetic_lazy_gauging(Vector; fast = fast_tests)
TestSuite.states_arithmetic_dense_crosscheck(Vector; fast = fast_tests)
TestSuite.states_arithmetic_argument_checks(Vector; fast = fast_tests)
TestSuite.states_arithmetic_cancellation_across_gauges(Vector; fast = fast_tests)
TestSuite.states_arithmetic_eigenstate_cancellation(Vector; fast = fast_tests)
TestSuite.states_arithmetic_single_site_addition(Vector; fast = fast_tests)
TestSuite.states_arithmetic_scalartype_promotion(Vector; fast = fast_tests)
