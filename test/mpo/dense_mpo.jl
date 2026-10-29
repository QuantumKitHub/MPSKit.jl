println("
--------------------
|   DenseMPO tests  |
--------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.mpo_dense_mpo(Vector; fast = fast_tests)
