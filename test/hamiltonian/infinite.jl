println("
--------------------------------------
|   InfiniteMPOHamiltonian tests     |
--------------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.hamiltonian_infinite_mpohamiltonian(Vector; fast = fast_tests)
TestSuite.hamiltonian_infinite_canonical_forms(Vector; fast = fast_tests)
