println("
-----------------------------------
|   FiniteMPOHamiltonian tests    |
-----------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.hamiltonian_finite_mpohamiltonian(Vector; fast = fast_tests)
TestSuite.hamiltonian_finite_canonicalization(Vector; fast = fast_tests)
TestSuite.hamiltonian_finite_repeated_indices(Vector; fast = fast_tests)
