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
using Random

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

        # the algorithm object carries no scratch space of its own - the sweep's allocator is
        # obtained per solve - so re-using one across solves has to reproduce the answer
        alg = DMRG(; verbosity = verbosity_conv, maxiter = 10)
        ψ1, = find_groundstate(ψ₀, H, alg)
        ψ2, = find_groundstate(ψ₀, H, alg)
        @test expectation_value(ψ1, H) ≈ expectation_value(ψ2, H) atol = 1.0e-10
    end

    @testset "DMRG2" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)
        trunc = truncrank(floor(Int, D * 1.5))
        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, DMRG2(; verbosity = verbosity_full, maxiter = 2, trunc)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, DMRG2(; verbosity = verbosity_conv, maxiter = 10, trunc), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
    end

    @testset "CBEDMRG" begin
        # start from a small bond so the bond expansion is exercised
        ψ₀ = FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^(D ÷ 2))
        v₀ = variance(ψ₀, H)
        expand = OptimalExpand(; trunc = truncrank(D ÷ 2))
        trunc = truncrank(D)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, DMRG(; verbosity = verbosity_full, maxiter = 2, alg_expand = expand, trunc)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, DMRG(; verbosity = verbosity_conv, maxiter = 10, alg_expand = expand, trunc), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
        # the bond should have grown to the truncation target
        @test dim(left_virtualspace(ψ, L ÷ 2)) == D
    end

    @testset "CBEDMRG (SketchedExpand)" begin
        # randomized bond expansion at single-site cost. The sketch is redrawn every sweep, so an
        # aggressive expansion (a large fraction of the bond) keeps the single-site Galerkin error
        # noisy; a gentle per-sweep increment lets it converge like the deterministic expanders.
        Random.seed!(1234)
        ψ₀ = FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^(D ÷ 2))
        v₀ = variance(ψ₀, H)
        expand = SketchedExpand(; trunc = truncrank(2), oversampling = 4)
        trunc = truncrank(D)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, DMRG(; verbosity = verbosity_full, maxiter = 2, alg_expand = expand, trunc)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, DMRG(; verbosity = verbosity_conv, maxiter = 15, alg_expand = expand, trunc), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
        # the bond should have grown to the truncation target
        @test dim(left_virtualspace(ψ, L ÷ 2)) == D
    end

    @testset "DMRG3S" begin
        # start from a small bond so the post-expansion is exercised, mirroring CBEDMRG above
        Random.seed!(1234)
        ψ₀ = FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^(D ÷ 2))
        v₀ = variance(ψ₀, H)
        alg_gauge = DMRG3S(0.1, ExponentialDecay(0.7))  # TODO: match final constructor API
        trunc = truncrank(D)

        # test logging
        ψ, envs, δ = find_groundstate(
            ψ₀, H, DMRG(; verbosity = verbosity_full, maxiter = 2, alg_gauge, trunc)
        )

        ψ, envs, δ = find_groundstate(
            ψ, H, DMRG(; verbosity = verbosity_conv, maxiter = 10, alg_gauge, trunc), envs
        )
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1.0e-3
        @test v < v₀
        @test v < 1.0e-2
        # the bond should have grown to the truncation target
        @test dim(left_virtualspace(ψ, L ÷ 2)) == D
    end

    @testset "DMRG3S escapes local minimum (Hubig et al. 2015, Sec. VII A)" begin
        L_heis = 20
        H_heis = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L = L_heis)

        Random.seed!(1234)
        ψ_bad = bad_initial_state(H_heis, L_heis)

        ψ_stuck, envs_stuck, δ_stuck = find_groundstate(
            ψ_bad, H_heis, DMRG(; verbosity = verbosity_conv, maxiter = 30)
        )
        E_stuck = real(expectation_value(ψ_stuck, H_heis, envs_stuck))

        alg_gauge = DMRG3S(0.1, ExponentialDecay(0.8))
        ψ_escape, envs_escape, δ_escape = find_groundstate(
            ψ_bad, H_heis, DMRG(;
                verbosity = verbosity_conv, maxiter = 30,
                alg_gauge, trunc = truncrank(20),
            )
        )
        E_escape = real(expectation_value(ψ_escape, H_heis, envs_escape))

        # paper reports E(α=0) = -6.35479, E(α≠0) = -8.6824724 for this L_heis = 20, S = 1/2 AFM setup
        @test E_escape < E_stuck - 1.0
        @test isapprox(E_escape, -8.6824724; atol = 1.0e-4)
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

@testset "Long-range Hamiltonian with real scalartype" verbose = true begin
    # A real `H` optimised over a complex MPS sends the derivative operators through their
    # converting constructors, and a long-range bond leaves sites with no ending (`B`) block
    tol = 1.0e-8
    D = 8

    @testset "FiniteMPS" begin
        L = 10
        H = long_range_ising(Float64; L)
        @test scalartype(H) <: Real

        ψ₀ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^D)
        # `complex(H)` takes the non-converting construction path
        E_ref = expectation_value(
            find_groundstate(ψ₀, complex(H), DMRG(; tol, verbosity = verbosity_conv))[1], H
        )
        for alg in (
                DMRG(; tol, verbosity = verbosity_conv),
                DMRG2(; tol, verbosity = verbosity_conv, trunc = truncrank(D)),
            )
            ψ, envs, δ = find_groundstate(ψ₀, H, alg)
            @test expectation_value(ψ, H, envs) ≈ E_ref atol = 1.0e-6
        end
    end

    @testset "InfiniteMPS" begin
        H = long_range_ising_infinite(Float64; L = 3)
        @test scalartype(H) <: Real

        ψ₀ = InfiniteMPS(randn, ComplexF64, fill(ℂ^2, 3), fill(ℂ^D, 3))
        E_ref = expectation_value(
            find_groundstate(ψ₀, complex(H), VUMPS(; tol, verbosity = verbosity_conv))[1], H
        )
        for alg in (VUMPS(; tol, verbosity = verbosity_conv), IDMRG(; tol, verbosity = verbosity_conv))
            ψ, envs, δ = find_groundstate(ψ₀, H, alg)
            @test expectation_value(ψ, H, envs) ≈ E_ref atol = 1.0e-6
        end
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
        trunc = truncrank(floor(Int, D * 1.5))
        ψ, envs, δ = find_groundstate(
            ψ₀, H_lazy, DMRG2(; tol, verbosity = verbosity_full, maxiter = 1, trunc)
        )

        # compare states
        alg = DMRG2(; tol, verbosity = verbosity_conv, trunc)
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
end
