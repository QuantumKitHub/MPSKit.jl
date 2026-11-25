println("
-----------------------------
|   Changebonds tests       |
-----------------------------
")

@isdefined(TestSuite) || include("../testsuite/TestSuite.jl")
using .TestSuite

TestSuite.algorithms_changebonds_mpo(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_infinitemps(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_finitemps(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_cbe_zero_content(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_density_matrix(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_multilinemps(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_svdcut_overflow(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_svdcut_longrange(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_svdcut_infinite_hamiltonian(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_randexpand_nan(Vector; fast = fast_tests)
TestSuite.algorithms_changebonds_nonuniform_unitcell(Vector; fast = fast_tests)
