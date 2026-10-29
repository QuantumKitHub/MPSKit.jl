println("
---------------------------------
|   WindowMPOHamiltonian tests   |
---------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.operators_window(Vector; fast = fast_tests)
