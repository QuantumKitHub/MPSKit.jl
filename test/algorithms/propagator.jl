println("
-----------------------------
|   Propagator tests         |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.algorithms_propagator_linear_solve_finitemps(Vector; fast = fast_tests)
TestSuite.algorithms_propagator_linear_solve_windowmps(Vector; fast = fast_tests)
TestSuite.algorithms_propagator_dynamical_dmrg_finitemps(Vector; fast = fast_tests)
TestSuite.algorithms_propagator_dynamical_dmrg_windowmps(Vector; fast = fast_tests)
