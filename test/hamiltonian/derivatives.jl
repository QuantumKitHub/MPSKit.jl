println("
----------------------------
|   Derivative operators    |
----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.hamiltonian_derivatives_jordan_structure(Vector; fast = fast_tests)
TestSuite.hamiltonian_derivatives_real_complex(Vector; fast = fast_tests)
TestSuite.hamiltonian_derivatives_finite_expval(Vector; fast = fast_tests)
