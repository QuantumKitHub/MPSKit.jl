println("
--------------------------------
|   Fidelity susceptibility     |
--------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.algorithms_fidelity_susceptibility(Vector; fast = fast_tests)
