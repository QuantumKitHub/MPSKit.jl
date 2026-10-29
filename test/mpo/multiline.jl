println("
----------------------------
|   MultilineMPO tests      |
----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.mpo_multiline_mpo_times_mps(Vector; fast = fast_tests)
TestSuite.mpo_multiline_leading_eigenvalue(Vector; fast = fast_tests)
TestSuite.mpo_multiline_multirow_fixed_point(Vector; fast = fast_tests)
TestSuite.mpo_multiline_unsupported_inputs(Vector; fast = fast_tests)
TestSuite.mpo_multiline_line_aliases(Vector; fast = fast_tests)
TestSuite.mpo_multiline_view_bounds(Vector; fast = fast_tests)
