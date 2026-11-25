println("
----------------------
|   WindowMPS tests   |
----------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.states_windowmps(Vector; fast = fast_tests)
