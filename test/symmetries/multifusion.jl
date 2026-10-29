println("
-----------------
|   Multifusion |
-----------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.symmetries_multifusion_infinitemps_construction(Vector; fast = fast_tests)
TestSuite.symmetries_multifusion_finitemps_construction(Vector; fast = fast_tests)
TestSuite.symmetries_multifusion_exact_diagonalization(Vector; fast = fast_tests)
TestSuite.symmetries_multifusion_finite(Vector; fast = fast_tests)
TestSuite.symmetries_multifusion_infinite(Vector; fast = fast_tests)
