println("
------------------------------------
|   Anyonic braiding consistency    |
------------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.algorithms_anyonic_braiding_sandwich(Vector; fast = fast_tests)
TestSuite.algorithms_anyonic_braiding_expectation_values(Vector; fast = fast_tests)
TestSuite.algorithms_anyonic_braiding_derivatives(Vector; fast = fast_tests)
TestSuite.algorithms_anyonic_braiding_finite(Vector; fast = fast_tests)
TestSuite.algorithms_anyonic_braiding_infinite(Vector; fast = fast_tests)
