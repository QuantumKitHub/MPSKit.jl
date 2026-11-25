println("
-----------------------------
|   Quasiparticle tests      |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.states_quasiparticle(Vector; fast = fast_tests)
