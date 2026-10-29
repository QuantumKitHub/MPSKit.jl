println("
-----------------------------
|   Time-stepping tests     |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.timeevolution_timestep(Vector; fast = fast_tests)
TestSuite.timeevolution_timestep_imaginary_bug(Vector; fast = fast_tests)
TestSuite.timeevolution_timestep_bug_truncation(Vector; fast = fast_tests)
TestSuite.timeevolution_timestep_symmetric(Vector; fast = fast_tests)
TestSuite.timeevolution_timestep_cbe_tdvp(Vector; fast = fast_tests)
TestSuite.timeevolution_timestep_truncation_error(Vector; fast = fast_tests)
TestSuite.timeevolution_time_evolve(Vector; fast = fast_tests)
