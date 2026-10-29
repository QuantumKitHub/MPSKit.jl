println("
----------------------------
|   InfiniteMPO tests      |
----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.mpo_infinite_basics(Vector; fast = fast_tests)
TestSuite.mpo_infinite_remove_orphans(Vector; fast = fast_tests)
