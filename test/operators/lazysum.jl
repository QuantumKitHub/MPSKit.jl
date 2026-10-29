println("
-----------------------------
|   LazySum / Operators      |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.operators_lazysum_general(Vector; fast = fast_tests)
TestSuite.operators_lazysum_multipliedoperator(Vector; fast = fast_tests)
TestSuite.operators_lazysum_timedependent(Vector; fast = fast_tests)
TestSuite.operators_lazysum_hamiltonian(Vector; fast = fast_tests)
