println("
---------------------------
|   MPO copy behaviour     |
---------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit

@testset "MPO copy behaviour" begin
    # testset that checks the fix for issue #288
    H = transverse_field_ising()
    O = make_time_mpo(H, 0.1, TaylorCluster(2, true, true))
    FO = open_boundary_conditions(O, 4)
    FH = open_boundary_conditions(H, 4)

    # check if the copy of the MPO is the same type as the original
    @test typeof(copy(O)) == typeof(O)
    @test typeof(copy(FO)) == typeof(FO)
    @test typeof(copy(H)) == typeof(H)
    @test typeof(copy(FH)) == typeof(FH)
end
