println("
---------------------------------------
|   Groundstate Algorithms (infinite) |
---------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Random

verbosity_full = 5
verbosity_conv = 1

@testset "InfiniteMPS ground state" verbose = true begin
    tol = 1.0e-8
    g = 4.0
    D = 6

    H_ref = force_planar(transverse_field_ising(; g))
    ψ = InfiniteMPS(ℙ^2, ℙ^D)
    v₀ = variance(ψ, H_ref)

    # VUMPS spawns over the unit cell, so it is run under both schedulers: which allocator serves
    # the local updates follows from the scheduler, and must not change any number
    # NOTE: the starting state is built from scratch rather than from this block's `ψ`. The testsets
    # around this one rebind `ψ` to their own (already repeated) result, and a `@testset for` body is
    # a single scope, so reading it here would feed a length-3 state back into `repeat`.
    @testset "VUMPS (unit cell $unit_cell_size, $schedname)" for unit_cell_size in [1, 3],
            (schedname, scheduler) in SCHEDULERS

        ψ₀ = repeat(InfiniteMPS(ℙ^2, ℙ^D), unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        ψ′, envs, δ = with_scheduler(scheduler) do
            # test logging
            ψ₁, = find_groundstate(
                ψ₀, H, VUMPS(; tol, verbosity = verbosity_full, maxiter = 2)
            )
            return find_groundstate(ψ₁, H, VUMPS(; tol, verbosity = verbosity_conv))
        end
        v = variance(ψ′, H, envs)

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

        trunc = trunctol(; atol = 1.0e-8)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ, H, IDMRG2(; tol, verbosity = verbosity_full, maxiter = 2, trunc)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, IDMRG2(; tol, verbosity = verbosity_conv, trunc)
        )
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    # the gradient is computed concurrently over the unit cell, so the scheduler decides its
    # allocator here too
    @testset "GradientGrassmann (unit cell $unit_cell_size, $schedname)" for unit_cell_size in
            [1, 3], (schedname, scheduler) in SCHEDULERS

        ψ₀ = repeat(InfiniteMPS(ℙ^2, ℙ^D), unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        ψ′, envs, δ = with_scheduler(scheduler) do
            # test logging
            ψ₁, = find_groundstate(
                ψ₀, H, GradientGrassmann(; tol, verbosity = verbosity_full, maxiter = 2)
            )
            return find_groundstate(
                ψ₁, H, GradientGrassmann(; tol, verbosity = verbosity_conv)
            )
        end
        v = variance(ψ′, H, envs)

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

        trunc = truncrank(floor(Int, D * 1.5))
        # test logging passes
        ψ, envs, δ = find_groundstate(
            ψ₀′, H_lazy′, IDMRG2(; tol, verbosity = verbosity_full, maxiter = 2, trunc)
        )

        # compare states
        alg = IDMRG2(; tol, verbosity = verbosity_conv, trunc)
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
        IDMRG2(; tol, verbosity, trunc = truncrank(D1)),
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
            ψ, envs = changebonds(ψ, mpo, OptimalExpand(; trunc = truncrank(D1 - D)), envs)
            ψ, envs = leading_boundary(ψ, mpo, alg)
            @test dim(space(ψ.AL[1, 1], 1)) == dim(space(ψ₀.AL[1, 1], 1)) + (D1 - D)
            @test expectation_value(ψ, mpo, envs) ≈ 2.5337 atol = 1.0e-3
        end
    end

    @testset "IDMRG2 growing bond dimension" begin
        Random.seed!(1234)
        V = Vect[Z2Irrep](0 => 1, 1 => 1)
        O = randn(ComplexF64, V ⊗ V, V ⊗ V)
        mpo = InfiniteMPO([O, O])
        P = physicalspace(O)
        ψ₀ = InfiniteMPS([P, P], [V, V])
        ψ, envs = leading_boundary(ψ₀, mpo, IDMRG2(; verbosity = 0, maxiter = 1, trunc = truncrank(8)))
        @test ψ isa InfiniteMPS
    end
end
