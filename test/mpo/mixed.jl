println("
------------------------
|   MPO tests (mixed)   |
------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.mpo_mixed_adapt(Vector; fast = fast_tests)
