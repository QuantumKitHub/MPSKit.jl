println("
-----------------------------
|   Sector conventions       |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.symmetries_sector_conventions(Vector; fast = fast_tests)
