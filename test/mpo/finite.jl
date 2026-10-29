println("
--------------------------
|   FiniteMPO tests      |
--------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.mpo_finite_basics(Vector; fast = fast_tests)
TestSuite.mpo_finite_remove_orphans(Vector; fast = fast_tests)
