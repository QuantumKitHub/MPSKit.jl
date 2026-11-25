@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.timeevolution_clusterexpansion_nonperturbative(Vector; fast = fast_tests)
TestSuite.timeevolution_clusterexpansion_mixed_spaces(Vector; fast = fast_tests)
TestSuite.timeevolution_clusterexpansion_timesteps(Vector; fast = fast_tests)
TestSuite.timeevolution_clusterexpansion_svd_support(Vector; fast = fast_tests)
