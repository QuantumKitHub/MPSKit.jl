println("
-----------------------------
|   Groundstate Algorithms  |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

verbosity_full = 5
verbosity_conv = 1

@testset "FiniteMPS ground state" verbose = true begin
    tol = 1.0e-8
    g = 4.0
    D = 6
    L = 10

    H = force_planar(transverse_field_ising(; g, L))

    @testset "DMRG" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, DMRG(; verbosity = verbosity_full, maxiter = 2)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, DMRG(; verbosity = verbosity_conv, maxiter = 10), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "DMRG2" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)
        trscheme = truncrank(floor(Int, D * 1.5))
        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, DMRG2(; verbosity = verbosity_full, maxiter = 2, trscheme)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, DMRG2(; verbosity = verbosity_conv, maxiter = 10, trscheme), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "GradientGrassmann" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, GradientGrassmann(; verbosity = verbosity_full, maxiter = 2)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, GradientGrassmann(; verbosity = verbosity_conv, maxiter = 50), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀ && v < 1.0e-2
    end
end

@testset "InfiniteMPS ground state" verbose = true begin
    tol = 1.0e-8
    g = 4.0
    D = 6

    H_ref = force_planar(transverse_field_ising(; g))
    ψ = InfiniteMPS(ℙ^2, ℙ^D)
    v₀ = variance(ψ, H_ref)

    @testset "VUMPS" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ, H, VUMPS(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        ψ, envs, δ = find_groundstate(ψ, H, VUMPS(; tol, verbosity = verbosity_conv))
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "IDMRG" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ, H, IDMRG(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        ψ, envs, δ = find_groundstate(ψ, H, IDMRG(; tol, verbosity = verbosity_conv))
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "IDMRG2" begin
        ψ = repeat(InfiniteMPS(ℙ^2, ℙ^D), 2)
        H = repeat(H_ref, 2)

        trscheme = trunctol(; atol = 1.0e-8)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ, H, IDMRG2(; tol, verbosity = verbosity_full, maxiter = 2, trscheme)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, IDMRG2(; tol, verbosity = verbosity_conv, trscheme)
        )
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "GradientGrassmann" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ, H, GradientGrassmann(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, GradientGrassmann(; tol, verbosity = verbosity_conv)
        )
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "Combination" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        alg = VUMPS(; tol = 100 * tol, verbosity = verbosity_conv, maxiter = 10) &
            GradientGrassmann(; tol, verbosity = verbosity_conv, maxiter = 50)
        ψ, envs, δ = find_groundstate(ψ, H, alg)

        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end
end

@testset "LazySum FiniteMPS ground state" verbose = true begin
    tol = 1.0e-8
    D = 15
    atol = 1.0e-2
    L = 10

    # test using XXZ model, Δ > 1 is gapped
    spin = 1
    local_operators = [S_x_S_x(; spin), S_y_S_y(; spin), 1.7 * S_z_S_z(; spin)]
    Pspace = space(local_operators[1], 1)
    lattice = fill(Pspace, L)

    mpo_hamiltonians = map(local_operators) do O
        return FiniteMPOHamiltonian(lattice, (i, i + 1) => O for i in 1:(L - 1))
    end

    H_lazy = LazySum(mpo_hamiltonians)
    H = sum(H_lazy)

    ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℂ^3, ℂ^D)
    ψ₀, = find_groundstate(ψ₀, H; tol, verbosity = 1)

    @testset "DMRG" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, DMRG(; tol, verbosity = verbosity_full, maxiter = 1)
        )

        # compare states
        alg = DMRG(; tol, verbosity = verbosity_conv)
        ψ, envs, δ = find_groundstate(ψ, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end

    @testset "DMRG2" begin
        # test logging passes
        trscheme = truncrank(floor(Int, D * 1.5))
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, DMRG2(; tol, verbosity = verbosity_full, maxiter = 1, trscheme)
        )

        # compare states
        alg = DMRG2(; tol, verbosity = verbosity_conv, trscheme)
        ψ, = find_groundstate(ψ₀, H, alg)
        ψ_lazy, envs, δ = find_groundstate(ψ₀, H_lazy, alg)

        @test abs(dot(ψ₀, ψ_lazy)) ≈ 1 atol = atol
    end

    @testset "GradientGrassmann" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, GradientGrassmann(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        # compare states
        alg = GradientGrassmann(; tol, verbosity = verbosity_conv)
        ψ, = find_groundstate(ψ₀, H, alg)
        ψ_lazy, envs, δ = find_groundstate(ψ₀, H_lazy, alg)

        @test abs(dot(ψ₀, ψ_lazy)) ≈ 1 atol = atol
    end
end

@testset "LazySum InfiniteMPS ground state" verbose = true begin
    tol = 1.0e-8
    D = 16
    atol = 1.0e-2

    spin = 1
    local_operators = [S_x_S_x(; spin), S_y_S_y(; spin), 0.7 * S_z_S_z(; spin)]
    Pspace = space(local_operators[1], 1)
    lattice = PeriodicVector([Pspace])
    mpo_hamiltonians = map(local_operators) do O
        return InfiniteMPOHamiltonian(lattice, (1, 2) => O)
    end

    H_lazy = LazySum(mpo_hamiltonians)
    H = sum(H_lazy)

    ψ₀ = InfiniteMPS(ℂ^3, ℂ^D)
    ψ₀, = find_groundstate(ψ₀, H; tol, verbosity = 1)

    @testset "VUMPS" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, VUMPS(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        # compare states
        alg = VUMPS(; tol, verbosity = verbosity_conv)
        ψ, envs, δ = find_groundstate(ψ, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end

    @testset "IDMRG" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, IDMRG(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        # compare states
        alg = IDMRG(; tol, verbosity = verbosity_conv, maxiter = 300)
        ψ, envs, δ = find_groundstate(ψ, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end

    @testset "IDMRG2" begin
        ψ₀′ = repeat(ψ₀, 2)
        H_lazy′ = repeat(H_lazy, 2)
        H′ = repeat(H, 2)

        trscheme = truncrank(floor(Int, D * 1.5))
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀′, H_lazy′, IDMRG2(; tol, verbosity = verbosity_full, maxiter = 2, trscheme)
        )

        # compare states
        alg = IDMRG2(; tol, verbosity = verbosity_conv, trscheme)
        ψ, envs, δ = find_groundstate(ψ, H_lazy′, alg)

        @test abs(dot(ψ₀′, ψ)) ≈ 1 atol = atol
    end

    @testset "GradientGrassmann" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, GradientGrassmann(; tol, verbosity = verbosity_full, maxiter = 2)
        )

        # compare states
        alg = GradientGrassmann(; tol, verbosity = verbosity_conv)
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end
end

@testset "leading_boundary" verbose = true begin
    tol = 1.0e-4
    verbosity = verbosity_conv
    D = 10
    D1 = 13
    algs = [
        VUMPS(; tol, verbosity), VOMPS(; tol, verbosity),
        GradientGrassmann(; tol, verbosity), IDMRG(; tol, verbosity),
        IDMRG2(; tol, verbosity, trscheme = truncrank(D1)),
    ]
    mpo = force_planar(classical_ising())

    ψ₀ = InfiniteMPS([ℙ^2], [ℙ^D])
    @testset "Infinite $i" for (i, alg) in enumerate(algs)
        if alg isa IDMRG2
            ψ2 = repeat(ψ₀, 2)
            mpo2 = repeat(mpo, 2)
            ψ, envs = leading_boundary(ψ2, mpo2, alg)
            @test dim(space(ψ.AL[1, 1], 1)) == dim(space(ψ₀.AL[1, 1], 1)) + (D1 - D)
            @test expectation_value(ψ, mpo2, envs) ≈ 2.5337^2 atol = 1.0e-3
        else
            ψ, envs = leading_boundary(ψ₀, mpo, alg)
            ψ, envs = changebonds(ψ, mpo, OptimalExpand(; trscheme = truncrank(D1 - D)), envs)
            ψ, envs = leading_boundary(ψ, mpo, alg)
            @test dim(space(ψ.AL[1, 1], 1)) == dim(space(ψ₀.AL[1, 1], 1)) + (D1 - D)
            @test expectation_value(ψ, mpo, envs) ≈ 2.5337 atol = 1.0e-3
        end
    end
end
