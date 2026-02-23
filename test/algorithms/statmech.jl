println("
-----------------------------
|   Finite temperature       |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

@testset "Finite temperature methods" begin
    imaginary_evolution = true
    @testset "Finite-size" begin
        L = 6
        H = transverse_field_ising(; L)
        trscheme = truncrank(20)
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
        rho_mps, = approximate(rho_mps, (rho_taylor_1, rho_mps), DMRG2(; trscheme, verbosity))
        Z_mpomul = tr(convert(FiniteMPO, rho_mps))^(1 / L)
        @test Z_mpomul ≈ Z_dense_2 atol = 1.0e-2

        # TDVP
        rho_0 = MPSKit.infinite_temperature_density_matrix(H)
        rho_0_mps = convert(FiniteMPS, rho_0)
        rho_mps, = timestep(rho_0_mps, H, 0.0, beta, TDVP2(; trscheme); imaginary_evolution)
        Z_tdvp = real(dot(rho_mps, rho_mps))^(1 / L)
        @test Z_tdvp ≈ Z_dense_2 atol = 1.0e-2

        @test expectation_value(rho_0_mps, 1 => S_x()) ≈ 0
        @test expectation_value(rho_0_mps, (1, 2) => S_x_S_x()) ≈ 0
        @test expectation_value(rho_mps, 1 => S_x()) ≈ E_x_taylor atol = 1.0e-2
        @test expectation_value(rho_mps, (1, 2) => S_x_S_x()) ≈ E_xx_taylor atol = 1.0e-2
    end

    @testset "Infinite-size" begin
        H = transverse_field_ising()
        trscheme = truncrank(20)
        verbosity = 1
        beta = 0.1

        # taylor cluster
        alg_taylor = TaylorCluster(; N = 2)
        rho_taylor = make_time_mpo(H, beta, alg_taylor; imaginary_evolution)
        rho_taylor_2 = make_time_mpo(H, 2beta, alg_taylor; imaginary_evolution)
        E_taylor = @constinferred expectation_value(rho_taylor, H)
        E_taylor2 = @constinferred expectation_value(rho_taylor_2, H)

        E_z_taylor = @constinferred expectation_value(rho_taylor, 1 => S_z())
        @test E_z_taylor ≈ 0 atol = 1.0e-4 # no spontaneous symmetry breaking at finite T

        # WII
        rho_wii = make_time_mpo(H, beta, WII(); imaginary_evolution)
        @test expectation_value(rho_wii, H) ≈ E_taylor atol = 1.0e-2
        @test expectation_value(rho_wii, 1 => S_z()) ≈ E_z_taylor atol = 1.0e-2

        # MPO multiplication
        rho_mps = convert(InfiniteMPS, rho_taylor)
        rho_mps2, = approximate(rho_mps, (rho_taylor, rho_mps), IDMRG(; verbosity))
        E_mpomul = expectation_value(rho_mps2, H)
        @test E_mpomul ≈ E_taylor2 atol = 1.0e-2

        # TDVP
        rho_0 = MPSKit.infinite_temperature_density_matrix(H)
        rho_0_mps, = changebonds(convert(InfiniteMPS, rho_0), H, OptimalExpand(; trscheme = truncrank(20)))
        rho_mps_tdvp, = timestep(rho_0_mps, H, 0.0, beta, TDVP(); imaginary_evolution)
        E_tdvp = expectation_value(rho_mps_tdvp, H)
        @test E_tdvp ≈ E_taylor atol = 1.0e-2

        num_vals = 2
        vals_taylor = @constinferred(transfer_spectrum(convert(InfiniteMPS, rho_taylor); num_vals))
        vals_mps = @constinferred(transfer_spectrum(rho_mps; num_vals))
        @test vals_taylor[1:num_vals] ≈ vals_mps[1:num_vals]
    end

    @testset "2D infinite partition functions with boundary MPS" verbose = true begin
        beta = 0.5 # ferromagnetic phase
        f_th = -2.0515856253898357
        m_th = 0.911319377877496
        e_th = -1.7455645753125533

        alg = VOMPS(; tol = 1.0e-8, verbosity = 1)
        O_mpo = classical_ising(; β = beta)
        ψ₀ = InfiniteMPS(ℂ^2, ℂ^10)
        ψ, envs = leading_boundary(ψ₀, O_mpo, alg)

        λ = expectation_value(ψ, O_mpo, envs)
        f = -log(λ) / beta
        @test f ≈ f_th atol = 1.0e-10

        O, M, E = classical_ising_tensors(beta)

        m = expectation_value(ψ, (O_mpo, 1 => M)) # normalised to give density
        @test abs(m) ≈ m_th atol = 1.0e-8 # account for spin flip

        e = expectation_value(ψ, (O_mpo, 1 => E))
        @test e ≈ e_th atol = 1.0e-2
    end
end
