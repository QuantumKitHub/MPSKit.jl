println("
---------------------------------
|   TaylorCluster time evolution |
---------------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.timeevolution_taylorcluster(Vector; fast = fast_tests)
