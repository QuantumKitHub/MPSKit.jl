println("
-----------------------------
|   Excitations tests       |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.excitations_excitations(Vector; fast = fast_tests)
