println("
-----------------------------------------------
|   MPOHamiltonian tests (mixed/constructors) |
-----------------------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.hamiltonian_mixed_constructors(Vector; fast = fast_tests)
TestSuite.hamiltonian_mixed_adapt(Vector; fast = fast_tests)
TestSuite.hamiltonian_mixed_channel_sharing(Vector; fast = fast_tests)
