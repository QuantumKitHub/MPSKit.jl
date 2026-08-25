println("
-------------------------------------
|   Finite temperature (finite)     |
-------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "Finite-size" begin
    imaginary_evolution = true
    L = 6
    H = transverse_field_ising(; L)
    trunc = truncrank(20)
    verbosity = 1
    beta = 0.1

    # exact diagonalization
    H_dense = convert(TensorMap, H)
    Z_dense_1 = tr(exp(-beta * H_dense))^(1 / L)
    Z_dense_2 = tr(exp(-2beta * H_dense))^(1 / L)

    # taylor cluster
    rho_taylor_1 = make_time_mpo(H, beta, TaylorCluster(; N = 2); imaginary_evolution)
    Z_taylor_1 = tr(rho_taylor_1)^(1 / L)
    @test Z_taylor_1 ≈ Z_dense_1 atol = 1.0e-2
    Z_taylor_2 = real(dot(rho_taylor_1, rho_taylor_1))^(1 / L)
    @test Z_taylor_2 ≈ Z_dense_2 atol = 1.0e-2

    E_x_taylor = @constinferred expectation_value(rho_taylor_1, 1 => S_x())
    E_xx_taylor = @constinferred expectation_value(rho_taylor_1, (1, 2) => S_x_S_x())

    # WII
    rho_wii = make_time_mpo(H, beta, WII(); imaginary_evolution)
    Z_wii = tr(rho_wii)^(1 / L)
    @test Z_wii ≈ Z_dense_1 atol = 1.0e-2
    @test expectation_value(rho_wii, 1 => S_x()) ≈ E_x_taylor atol = 1.0e-2
    @test expectation_value(rho_wii, (1, 2) => S_x_S_x()) ≈ E_xx_taylor atol = 1.0e-2

    # MPO multiplication
    rho_mps = convert(FiniteMPS, rho_taylor_1)
    rho_mps, = approximate(rho_mps, (rho_taylor_1, rho_mps), DMRG2(; trunc, verbosity))
    Z_mpomul = tr(convert(FiniteMPO, rho_mps))^(1 / L)
    @test Z_mpomul ≈ Z_dense_2 atol = 1.0e-2

    # TDVP
    rho_0 = MPSKit.infinite_temperature_density_matrix(H)
    rho_0_mps = convert(FiniteMPS, rho_0)
    rho_mps, = timestep(rho_0_mps, H, 0.0, beta, TDVP2(; trunc); imaginary_evolution)
    Z_tdvp = real(dot(rho_mps, rho_mps))^(1 / L)
    @test Z_tdvp ≈ Z_dense_2 atol = 1.0e-2

    @test expectation_value(rho_0_mps, 1 => S_x()) ≈ 0
    @test expectation_value(rho_0_mps, (1, 2) => S_x_S_x()) ≈ 0
    @test expectation_value(rho_mps, 1 => S_x()) ≈ E_x_taylor atol = 1.0e-2
    @test expectation_value(rho_mps, (1, 2) => S_x_S_x()) ≈ E_xx_taylor atol = 1.0e-2
end
