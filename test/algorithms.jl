println("
------------------
|   Algorithms   |
------------------
")
module TestAlgorithms

    using ..TestSetup
    using Test, TestExtras
    using MPSKit
    using MPSKit: fuse_mul_mpo
    using TensorKit
    using TensorKit: ℙ
    using LinearAlgebra: eigvals

    verbosity_full = 5
    verbosity_conv = 1

    @testset "FiniteMPS groundstate" verbose = true begin
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

    @testset "InfiniteMPS groundstate" verbose = true begin
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

    @testset "LazySum FiniteMPS groundstate" verbose = true begin
        tol = 1.0e-8
        D = 15
        atol = 1.0e-2
        L = 10

        # test using XXZ model, Δ > 1 is gapped
        spin = 1
        local_operators = [S_xx(; spin), S_yy(; spin), 1.7 * S_zz(; spin)]
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

    @testset "LazySum InfiniteMPS groundstate" verbose = true begin
        tol = 1.0e-8
        D = 16
        atol = 1.0e-2

        spin = 1
        local_operators = [S_xx(; spin), S_yy(; spin), 0.7 * S_zz(; spin)]
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

    @testset "timestep" verbose = true begin
        dt = 0.1
        algs = [TDVP(), TDVP2(; trscheme = truncrank(10))]
        L = 10

        H = force_planar(heisenberg_XXX(Trivial, Float64; spin = 1 // 2, L))
        ψ = FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4)
        E = expectation_value(ψ, H)
        ψ₀, = find_groundstate(ψ, H)
        E₀ = expectation_value(ψ₀, H)

        @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
            ψ1, envs = timestep(ψ₀, H, 0.0, dt, alg)
            E1 = expectation_value(ψ1, H, envs)
            @test E₀ ≈ E1 atol = 1.0e-2
            @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        end

        Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])

        @testset "Finite LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
            ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, alg)
            E = expectation_value(ψ, Hlazy, envs)
            @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1.0e-2
        end

        Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

        @testset "Finite TimeDependent LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
            ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, alg)
            E = expectation_value(ψ, Ht(1.0), envs)

            ψt, envst = timestep(ψ₀, Ht, 1.0, dt, alg)
            Et = expectation_value(ψt, Ht(1.0), envst)
            @test E ≈ Et atol = 1.0e-8
        end

        Ht2 = MultipliedOperator(H, t -> t < 0 ? error("t < 0!") : 4) +
            MultipliedOperator(H, 1.45)
        @testset "Finite TimeDependent LazySum (fix negative t issue) $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
            ψ, envs = timestep(ψ₀, Ht2, 0.0, dt, alg)
            E = expectation_value(ψ, Ht2(0.0), envs)

            ψt, envst = timestep(ψ₀, Ht2, 0.0, dt, alg)
            Et = expectation_value(ψt, Ht2(0.0), envst)
            @test E ≈ Et atol = 1.0e-8
        end

        H = repeat(force_planar(heisenberg_XXX(; spin = 1)), 2)
        ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
        E₀ = expectation_value(ψ₀, H)

        @testset "Infinite TDVP" begin
            ψ, envs = timestep(ψ₀, H, 0.0, dt, TDVP())
            E = expectation_value(ψ, H, envs)
            @test E₀ ≈ E atol = 1.0e-2
        end

        Hlazy = LazySum([3 * deepcopy(H), 1.55 * deepcopy(H), -0.1 * deepcopy(H)])

        @testset "Infinite LazySum TDVP" begin
            ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, TDVP())
            E = expectation_value(ψ, Hlazy, envs)
            @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1.0e-2
        end

        Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

        @testset "Infinite TimeDependent LazySum" begin
            ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, TDVP())
            E = expectation_value(ψ, Ht(1.0), envs)

            ψt, envst = timestep(ψ₀, Ht, 1.0, dt, TDVP())
            Et = expectation_value(ψt, Ht(1.0), envst)
            @test E ≈ Et atol = 1.0e-8
        end
    end

    @testset "time_evolve" verbose = true begin
        t_span = 0:0.1:0.1
        algs = [TDVP(), TDVP2(; trscheme = truncrank(10))]

        L = 10
        H = force_planar(heisenberg_XXX(; spin = 1 // 2, L))
        ψ₀ = FiniteMPS(L, ℙ^2, ℙ^1)
        E₀ = expectation_value(ψ₀, H)

        @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
            ψ, envs = time_evolve(ψ₀, H, t_span, alg)
            E = expectation_value(ψ, H, envs)
            @test E₀ ≈ E atol = 1.0e-2
        end

        H = repeat(force_planar(heisenberg_XXX(; spin = 1)), 2)
        ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
        E₀ = expectation_value(ψ₀, H)

        @testset "Infinite TDVP" begin
            ψ, envs = time_evolve(ψ₀, H, t_span, TDVP())
            E = expectation_value(ψ, H, envs)
            @test E₀ ≈ E atol = 1.0e-2
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

    @testset "excitations" verbose = true begin
        @testset "infinite (ham)" begin
            H = repeat(force_planar(heisenberg_XXX()), 2)
            ψ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^48, ℙ^48])
            ψ, envs, _ = find_groundstate(
                ψ, H; maxiter = 400, verbosity = verbosity_conv, tol = 1.0e-10
            )
            energies, ϕs = @inferred excitations(
                H, QuasiparticleAnsatz(), Float64(pi), ψ, envs
            )
            @test energies[1] ≈ 0.41047925 atol = 1.0e-4
            @test variance(ϕs[1], H) < 1.0e-8
        end
        @testset "infinite sector convention" begin
            g = 4
            H = repeat(transverse_field_ising(Z2Irrep; g = g, L = Inf), 2)
            V = Z2Space(0 => 24, 1 => 24)
            ψ = InfiniteMPS(physicalspace(H), [V, V])
            ψ, envs = find_groundstate(ψ, H, VUMPS(; tol = 1.0e-10, maxiter = 400))

            # testing istrivial and istopological
            momentum = 0
            exc0, qp0 = excitations(H, QuasiparticleAnsatz(), momentum, ψ; sector = Z2Irrep(0))
            exc1, qp1 = excitations(H, QuasiparticleAnsatz(), momentum, ψ; sector = Z2Irrep(1))
            @test isapprox(first(exc1), abs(2 * (g - 1)); atol = 1.0e-6) # charged excitation lower in energy
            @test first(exc0) > 2 * first(exc1)
            @test variance(qp1[1], H) < 1.0e-8
        end
        @testset "infinite (mpo)" begin
            H = repeat(sixvertex(), 2)
            ψ = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
            ψ, envs, _ = leading_boundary(
                ψ, H, VUMPS(; maxiter = 400, verbosity = verbosity_conv, tol = 1.0e-10)
            )
            energies, ϕs = @inferred excitations(
                H, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ψ, envs; verbosity = 0
            )
            @test abs(energies[1]) > abs(energies[2]) # has a minimum at pi/2
        end

        @testset "finite" begin
            verbosity = verbosity_conv
            H_inf = force_planar(transverse_field_ising())
            ψ_inf = InfiniteMPS([ℙ^2], [ℙ^10])
            ψ_inf, envs, _ = find_groundstate(ψ_inf, H_inf; maxiter = 400, verbosity, tol = 1.0e-9)
            energies, ϕs = @inferred excitations(H_inf, QuasiparticleAnsatz(), 0.0, ψ_inf, envs)
            inf_en = energies[1]

            fin_en = map([20, 10]) do len
                H = force_planar(transverse_field_ising(; L = len))
                ψ = FiniteMPS(rand, ComplexF64, len, ℙ^2, ℙ^10)
                ψ, envs, = find_groundstate(ψ, H; verbosity)

                # find energy with quasiparticle ansatz
                energies_QP, ϕs = @inferred excitations(H, QuasiparticleAnsatz(), ψ, envs)
                @test variance(ϕs[1], H) < 1.0e-6

                # find energy with normal dmrg
                for gsalg in (
                        DMRG(; verbosity, tol = 1.0e-6),
                        DMRG2(; verbosity, tol = 1.0e-6, trscheme = trunctol(; atol = 1.0e-4)),
                    )
                    energies_dm, _ = @inferred excitations(H, FiniteExcited(; gsalg), ψ)
                    @test energies_dm[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
                end

                # find energy with Chepiga ansatz
                energies_ch, _ = @inferred excitations(H, ChepigaAnsatz(), ψ, envs)
                @test energies_ch[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
                energies_ch2, _ = @inferred excitations(H, ChepigaAnsatz2(), ψ, envs)
                @test energies_ch2[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1.0e-4
                return energies_QP[1]
            end

            @test issorted(abs.(fin_en .- inf_en))
        end
    end

    @testset "changebonds $((pspace, Dspace))" verbose = true for (pspace, Dspace) in
        [
            (ℙ^4, ℙ^3),
            (Rep[SU₂](1 => 1), Rep[SU₂](0 => 2, 1 => 2, 2 => 1)),
        ]
        @testset "mpo" begin
            #random nn interaction
            nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
            nn += nn'
            H = InfiniteMPOHamiltonian(PeriodicVector(fill(pspace, 1)), (1, 2) => nn)
            Δt = 0.1
            expH = make_time_mpo(H, Δt, WII())

            O = MPSKit.DenseMPO(expH)
            Op = periodic_boundary_conditions(O, 10)
            Op′ = changebonds(Op, SvdCut(; trscheme = truncrank(5)))

            @test dim(space(Op′[5], 1)) < dim(space(Op[5], 1))
        end

        @testset "infinite mps" begin
            # random nn interaction
            nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
            nn += nn'
            H0 = InfiniteMPOHamiltonian(PeriodicVector(fill(pspace, 1)), (1, 2) => nn)

            # test rand_expand
            for unit_cell_size in 2:3
                H = repeat(H0, unit_cell_size)
                state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

                state_re = changebonds(
                    state, RandExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)))
                )
                @test dot(state, state_re) ≈ 1 atol = 1.0e-8
            end
            # test optimal_expand
            for unit_cell_size in 2:3
                H = repeat(H0, unit_cell_size)
                state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

                state_oe, _ = changebonds(
                    state, H, OptimalExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)))
                )
                @test dot(state, state_oe) ≈ 1 atol = 1.0e-8
            end
            # test VUMPSSvdCut
            for unit_cell_size in [1, 2, 3, 4]
                H = repeat(H0, unit_cell_size)
                state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

                state_vs, _ = changebonds(state, H, VUMPSSvdCut(; trscheme = notrunc()))
                @test dim(left_virtualspace(state, 1)) < dim(left_virtualspace(state_vs, 1))

                state_vs_tr = changebonds(state_vs, SvdCut(; trscheme = truncrank(dim(Dspace))))
                @test dim(right_virtualspace(state_vs_tr, 1)) < dim(right_virtualspace(state_vs, 1))
            end
        end

        @testset "finite mps" begin
            #random nn interaction
            L = 10
            nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
            nn += nn'
            H = FiniteMPOHamiltonian(fill(pspace, L), (i, i + 1) => nn for i in 1:(L - 1))

            state = FiniteMPS(L, pspace, Dspace)

            state_re = changebonds(
                state, RandExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)))
            )
            @test dot(state, state_re) ≈ 1 atol = 1.0e-8

            state_oe, _ = changebonds(
                state, H, OptimalExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)))
            )
            @test dot(state, state_oe) ≈ 1 atol = 1.0e-8

            state_tr = changebonds(state_oe, SvdCut(; trscheme = truncrank(dim(Dspace))))

            @test dim(left_virtualspace(state_tr, 5)) < dim(left_virtualspace(state_oe, 5))
        end

        @testset "MultilineMPS" begin
            o = rand(ComplexF64, pspace * pspace, pspace * pspace)
            mpo = MultilineMPO(o)

            t = rand(ComplexF64, Dspace * pspace, Dspace)
            state = MultilineMPS(fill(t, 1, 1))

            state_re = changebonds(
                state, RandExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)))
            )
            @test dot(state, state_re) ≈ 1 atol = 1.0e-8

            state_oe, _ = changebonds(
                state, mpo, OptimalExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)))
            )
            @test dot(state, state_oe) ≈ 1 atol = 1.0e-8

            state_tr = changebonds(state_oe, SvdCut(; trscheme = truncrank(dim(Dspace))))

            @test dim(left_virtualspace(state_tr, 1, 1)) < dim(left_virtualspace(state_oe, 1, 1))
        end
    end

    @testset "Dynamical DMRG" verbose = true begin
        L = 10
        H = force_planar(-transverse_field_ising(; L, g = -4))
        gs, = find_groundstate(FiniteMPS(L, ℙ^2, ℙ^10), H; verbosity = verbosity_conv)
        E₀ = expectation_value(gs, H)

        vals = (-0.5:0.2:0.5) .+ E₀
        eta = 0.3im

        predicted = [1 / (v + eta - E₀) for v in vals]

        @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
            alg = DynamicalDMRG(; flavour = f, verbosity = 0, tol = 1.0e-8)
            data = map(vals) do v
                result, = propagator(gs, v + eta, H, alg)
                return result
            end
            @test data ≈ predicted atol = 1.0e-8
        end
    end

    @testset "fidelity susceptibility" begin
        X = TensorMap(ComplexF64[0 1; 1 0], ℂ^2 ← ℂ^2)
        Z = TensorMap(ComplexF64[1 0; 0 -1], ℂ^2 ← ℂ^2)

        H_X = InfiniteMPOHamiltonian(X)
        H_ZZ = InfiniteMPOHamiltonian(Z ⊗ Z)

        hamiltonian(λ) = H_ZZ + λ * H_X
        analytical_susceptibility(λ) = abs(1 / (16 * λ^2 * (λ^2 - 1)))

        for λ in [1.05, 2.0, 4.0]
            H = hamiltonian(λ)
            ψ = InfiniteMPS([ℂ^2], [ℂ^16])
            ψ, envs, = find_groundstate(ψ, H, VUMPS(; maxiter = 100, verbosity = 0))

            numerical_susceptibility = fidelity_susceptibility(ψ, H, [H_X], envs; maxiter = 10)
            @test numerical_susceptibility[1, 1] ≈ analytical_susceptibility(λ) atol = 1.0e-2

            # test if the finite fid sus approximates the analytical one with increasing system size
            fin_en = map([20, 15, 10]) do L
                Hfin = open_boundary_conditions(hamiltonian(λ), L)
                H_Xfin = open_boundary_conditions(H_X, L)
                ψ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^16)
                ψ, envs, = find_groundstate(ψ, Hfin, DMRG(; verbosity = 0))
                numerical_susceptibility = fidelity_susceptibility(
                    ψ, Hfin, [H_Xfin], envs; maxiter = 10
                )
                return numerical_susceptibility[1, 1] / L
            end
            @test issorted(abs.(fin_en .- analytical_susceptibility(λ)))
        end
    end

    # stub tests
    @testset "correlation length / entropy" begin
        ψ = InfiniteMPS([ℙ^2], [ℙ^10])
        H = force_planar(transverse_field_ising())
        ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
        len_crit = correlation_length(ψ)[1]
        entrop_crit = entropy(ψ)

        H = force_planar(transverse_field_ising(; g = 4))
        ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))
        len_gapped = correlation_length(ψ)[1]
        entrop_gapped = entropy(ψ)

        @test len_crit > len_gapped
        @test real(entrop_crit) > real(entrop_gapped)
    end

    @testset "expectation value / correlator" begin
        g = 4.0
        ψ = InfiniteMPS(ℂ^2, ℂ^10)
        H = transverse_field_ising(; g)
        ψ, = find_groundstate(ψ, H, VUMPS(; verbosity = 0))

        @test expectation_value(ψ, H) ≈
            expectation_value(ψ, 1 => -g * S_x()) + expectation_value(ψ, (1, 2) => -S_zz())
        Z_mpo = MPSKit.add_util_mpoleg(S_z())
        G = correlator(ψ, Z_mpo, Z_mpo, 1, 2:5)
        G2 = correlator(ψ, S_zz(), 1, 3:2:5)
        @test isapprox(G[2], G2[1], atol = 1.0e-2)
        @test isapprox(last(G), last(G2), atol = 1.0e-2)
        @test isapprox(G[1], expectation_value(ψ, (1, 2) => S_zz()), atol = 1.0e-2)
        @test isapprox(G[2], expectation_value(ψ, (1, 3) => S_zz()), atol = 1.0e-2)
    end

    @testset "approximate" verbose = true begin
        verbosity = verbosity_conv
        @testset "mpo * infinite ≈ infinite" begin
            ψ = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^10, ℙ^10])
            ψ0 = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^12, ℙ^12])

            H = force_planar(repeat(transverse_field_ising(; g = 4), 2))

            dt = 1.0e-3
            sW1 = make_time_mpo(H, dt, TaylorCluster(; N = 3, compression = true, extension = true))
            sW2 = make_time_mpo(H, dt, WII())
            W1 = MPSKit.DenseMPO(sW1)
            W2 = MPSKit.DenseMPO(sW2)

            ψ1, _ = approximate(ψ0, (sW1, ψ), VOMPS(; verbosity))
            MPSKit.Defaults.set_scheduler!(:serial)
            ψ2, _ = approximate(ψ0, (W2, ψ), VOMPS(; verbosity))
            MPSKit.Defaults.set_scheduler!()

            ψ3, _ = approximate(ψ0, (W1, ψ), IDMRG(; verbosity))
            ψ4, _ = approximate(ψ0, (sW2, ψ), IDMRG2(; trscheme = truncrank(12), verbosity))
            ψ5, _ = timestep(ψ, H, 0.0, dt, TDVP())
            ψ6 = changebonds(W1 * ψ, SvdCut(; trscheme = truncrank(12)))

            @test abs(dot(ψ1, ψ5)) ≈ 1.0 atol = dt
            @test abs(dot(ψ3, ψ5)) ≈ 1.0 atol = dt
            @test abs(dot(ψ6, ψ5)) ≈ 1.0 atol = dt
            @test abs(dot(ψ2, ψ4)) ≈ 1.0 atol = dt

            nW1 = changebonds(W1, SvdCut(; trscheme = trunctol(; atol = dt))) # this should be a trivial mpo now
            @test dim(space(nW1[1], 1)) == 1
        end

        finite_algs = [DMRG(; verbosity), DMRG2(; verbosity, trscheme = truncrank(10))]
        @testset "finitemps1 ≈ finitemps2" for alg in finite_algs
            a = FiniteMPS(10, ℂ^2, ℂ^10)
            b = FiniteMPS(10, ℂ^2, ℂ^20)

            before = abs(dot(a, b))

            a = first(approximate(a, b, alg))

            after = abs(dot(a, b))

            @test before < after
        end

        @testset "sparse_mpo * finitemps1 ≈ finitemps2" for alg in finite_algs
            L = 10
            ψ₁ = FiniteMPS(L, ℂ^2, ℂ^30)
            ψ₂ = FiniteMPS(L, ℂ^2, ℂ^25)

            H = transverse_field_ising(; g = 4.0, L)
            τ = 1.0e-3

            expH = make_time_mpo(H, τ, WI)
            ψ₂, = approximate(ψ₂, (expH, ψ₁), alg)
            normalize!(ψ₂)
            ψ₂′, = timestep(ψ₁, H, 0.0, τ, TDVP())
            @test abs(dot(ψ₁, ψ₁)) ≈ abs(dot(ψ₂, ψ₂′)) atol = 0.001
        end

        @testset "dense_mpo * finitemps1 ≈ finitemps2" for alg in finite_algs
            L = 10
            ψ₁ = FiniteMPS(L, ℂ^2, ℂ^20)
            ψ₂ = FiniteMPS(L, ℂ^2, ℂ^10)

            O = classical_ising(; L)
            ψ₂, = approximate(ψ₂, (O, ψ₁), alg)

            @test norm(O * ψ₁ - ψ₂) ≈ 0 atol = 0.001
        end
    end

    @testset "periodic boundary conditions" begin
        Hs = [transverse_field_ising(), heisenberg_XXX(), classical_ising(), sixvertex()]
        for N in 2:6
            for H in Hs
                TH = convert(TensorMap, periodic_boundary_conditions(H, N))
                @test TH ≈
                    permute(TH, ((vcat(N, 1:(N - 1))...,), (vcat(2N, (N + 1):(2N - 1))...,)))
            end
        end

        # fermionic tests
        for N in 3:5
            h = real(c_plusmin() + c_minplus())
            H = InfiniteMPOHamiltonian([space(h, 1)], (1, 2) => h)
            H_periodic = periodic_boundary_conditions(H, N)
            terms = [(i, i + 1) => h for i in 1:(N - 1)]
            push!(terms, (1, N) => permute(h, ((2, 1), (4, 3))))
            H_periodic2 = FiniteMPOHamiltonian(physicalspace(H_periodic), terms)
            @test H_periodic ≈ H_periodic2
        end
    end

    @testset "TaylorCluster time evolution" begin
        L = 4
        dt = 0.05
        dτ = -im * dt

        @testset "O(1) exact expression" begin
            alg = TaylorCluster(; N = 1, compression = true, extension = true)
            for H in (transverse_field_ising(), heisenberg_XXX())
                # Infinite
                mpo = make_time_mpo(H, dt, alg)
                O = mpo[1]
                I, A, B, C, D = H[1][1], H.A[1], H.B[1], H.C[1], H.D[1]

                @test size(O, 1) == size(D, 1)^2 + (size(D, 1) * size(B, 1))
                @test size(O, 4) == size(D, 4)^2 + (size(C, 4) * size(D, 4))

                O_exact = similar(O)
                O_exact[1, 1, 1, 1] = I + dτ * D + dτ^2 / 2 * fuse_mul_mpo(D, D)
                O_exact[1, 1, 1, 2] = C + dτ * symm_mul_mpo(C, D)
                O_exact[2, 1, 1, 1] = dτ * (B + dτ * symm_mul_mpo(B, D))
                O_exact[2, 1, 1, 2] = A + dτ * (symm_mul_mpo(A, D) + symm_mul_mpo(C, B))

                @test all(isapprox.(parent(O), parent(O_exact)))

                # Finite
                H_fin = open_boundary_conditions(H, L)
                mpo_fin = make_time_mpo(H_fin, dt, alg)
                mpo_fin2 = open_boundary_conditions(mpo, L)
                for i in 1:L
                    @test all(isapprox.(parent(mpo_fin[i]), parent(mpo_fin2[i])))
                end
            end
        end

        @testset "O(2) exact expression" begin
            alg = TaylorCluster(; N = 2, compression = true, extension = true)
            for H in (transverse_field_ising(), heisenberg_XXX())
                # Infinite
                mpo = make_time_mpo(H, dt, alg)
                O = mpo[1]
                I, A, B, C, D = H[1][1], H.A[1], H.B[1], H.C[1], H.D[1]

                @test size(O, 1) ==
                    size(D, 1)^3 + (size(D, 1)^2 * size(B, 1)) + (size(D, 1) * size(B, 1)^2)
                @test size(O, 4) ==
                    size(D, 4)^3 + (size(C, 4) * size(D, 4)^2) + (size(D, 4) * size(C, 4)^2)

                O_exact = similar(O)
                O_exact[1, 1, 1, 1] = I + dτ * D + dτ^2 / 2 * fuse_mul_mpo(D, D) +
                    dτ^3 / 6 * fuse_mul_mpo(fuse_mul_mpo(D, D), D)
                O_exact[1, 1, 1, 2] = C + dτ * symm_mul_mpo(C, D) +
                    dτ^2 / 2 * symm_mul_mpo(C, D, D)
                O_exact[1, 1, 1, 3] = fuse_mul_mpo(C, C) + dτ * symm_mul_mpo(C, C, D)
                O_exact[2, 1, 1, 1] = dτ *
                    (
                    B + dτ * symm_mul_mpo(B, D) +
                        dτ^2 / 2 * symm_mul_mpo(B, D, D)
                )
                O_exact[2, 1, 1, 2] = A + dτ * symm_mul_mpo(A, D) +
                    dτ^2 / 2 * symm_mul_mpo(A, D, D) +
                    dτ * (symm_mul_mpo(C, B) + dτ * symm_mul_mpo(C, B, D))
                O_exact[2, 1, 1, 3] = 2 * (symm_mul_mpo(A, C) + dτ * symm_mul_mpo(A, C, D)) +
                    dτ * symm_mul_mpo(C, C, B)
                O_exact[3, 1, 1, 1] = dτ^2 / 2 *
                    (fuse_mul_mpo(B, B) + dτ * symm_mul_mpo(B, B, D))
                O_exact[3, 1, 1, 2] = dτ * (symm_mul_mpo(A, B) + dτ * symm_mul_mpo(A, B, D)) +
                    dτ^2 / 2 * symm_mul_mpo(B, B, C)
                O_exact[3, 1, 1, 3] = fuse_mul_mpo(A, A) + dτ * symm_mul_mpo(A, A, D) +
                    2 * dτ * symm_mul_mpo(A, C, B)

                @test all(isapprox.(parent(O), parent(O_exact)))

                # Finite
                H_fin = open_boundary_conditions(H, L)
                mpo_fin = make_time_mpo(H_fin, dt, alg)
                mpo_fin2 = open_boundary_conditions(mpo, L)
                for i in 1:L
                    @test all(isapprox.(parent(mpo_fin[i]), parent(mpo_fin2[i])))
                end
            end
        end

        L = 4
        Hs = [transverse_field_ising(; L = L), heisenberg_XXX(; L = L)]

        Ns = [1, 2, 3]
        dts = [1.0e-2, 1.0e-3]
        for H in Hs
            ψ = FiniteMPS(L, physicalspace(H, 1), ℂ^16)
            for N in Ns
                εs = zeros(ComplexF64, 2)
                for (i, dt) in enumerate(dts)
                    ψ₀, _ = find_groundstate(ψ, H, DMRG(; verbosity = 0))
                    E₀ = expectation_value(ψ₀, H)

                    O = make_time_mpo(H, dt, TaylorCluster(; N = N))

                    ψ₁, _ = approximate(ψ₀, (O, ψ₀), DMRG(; verbosity = 0))
                    εs[i] = norm(dot(ψ₀, ψ₁) - exp(-im * E₀ * dt))
                end
                @test (log(εs[2]) - log(εs[1])) / (log(dts[2]) - log(dts[1])) ≈ N + 1 atol = 0.1
            end

            for N in Ns
                εs = zeros(ComplexF64, 2)
                for (i, dt) in enumerate(dts)
                    ψ₀, _ = find_groundstate(ψ, H, DMRG(; verbosity = 0))
                    E₀ = expectation_value(ψ₀, H)

                    O = make_time_mpo(H, dt, TaylorCluster(; N = N, compression = true))

                    ψ₁, _ = approximate(ψ₀, (O, ψ₀), DMRG(; verbosity = 0))
                    εs[i] = norm(dot(ψ₀, ψ₁) - exp(-im * E₀ * dt))
                end
                @test (log(εs[2]) - log(εs[1])) / (log(dts[2]) - log(dts[1])) ≈ N + 1 atol = 0.1
            end
        end
    end

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
            E_xx_taylor = @constinferred expectation_value(rho_taylor_1, (1, 2) => S_xx())

            # WII
            rho_wii = make_time_mpo(H, beta, WII(); imaginary_evolution)
            Z_wii = tr(rho_wii)^(1 / L)
            @test Z_wii ≈ Z_dense_1 atol = 1.0e-2
            @test expectation_value(rho_wii, 1 => S_x()) ≈ E_x_taylor atol = 1.0e-2
            @test expectation_value(rho_wii, (1, 2) => S_xx()) ≈ E_xx_taylor atol = 1.0e-2

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
            @test expectation_value(rho_0_mps, (1, 2) => S_xx()) ≈ 0
            @test expectation_value(rho_mps, 1 => S_x()) ≈ E_x_taylor atol = 1.0e-2
            @test expectation_value(rho_mps, (1, 2) => S_xx()) ≈ E_xx_taylor atol = 1.0e-2
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
    end

    @testset "Sector conventions" begin
        L = 4
        H = XY_model(U1Irrep; L)

        H_dense = convert(TensorMap, H)
        vals_dense = TensorKit.SectorDict(c => sort(v; by = real) for (c, v) in eigvals(H_dense))

        tol = 1.0e-18 # tolerance required to separate degenerate eigenvalues
        alg = MPSKit.Defaults.alg_eigsolve(; dynamic_tols = false, tol)

        maxVspaces = MPSKit.max_virtualspaces(physicalspace(H))
        gs, = find_groundstate(
            FiniteMPS(physicalspace(H), maxVspaces[2:(end - 1)]), H; verbosity = 0
        )
        E₀ = expectation_value(gs, H)
        @test E₀ ≈ first(vals_dense[unit(U1Irrep)])

        for (sector, vals) in vals_dense
            # ED tests
            num = length(vals)
            E₀s, ψ₀s, info = exact_diagonalization(H; num, sector, alg)
            @test E₀s[1:num] ≈ vals[1:num]
            # this is a trick to make the mps full-rank again, which is not guaranteed by ED
            ψ₀ = changebonds(first(ψ₀s), SvdCut(; trscheme = notrunc()))
            Vspaces = left_virtualspace.(Ref(ψ₀), 1:L)
            push!(Vspaces, right_virtualspace(ψ₀, L))
            @test all(splat(==), zip(Vspaces, MPSKit.max_virtualspaces(ψ₀)))

            # Quasiparticle tests
            Es, Bs = excitations(H, QuasiparticleAnsatz(; tol), gs; sector, num = 1)
            Es = Es .+ E₀
            # first excited state is second eigenvalue if sector is trivial
            @test Es[1] ≈ vals[isunit(sector) ? 2 : 1] atol = 1.0e-8
        end

        # shifted charges tests
        # targeting states with Sz = 1 => vals_shift_dense[0] == vals_dense[1]
        # so effectively shifting the charges by -1
        H_shift = MPSKit.add_physical_charge(H, U1Irrep.([1, 0, 0, 0]))
        H_shift_dense = convert(TensorMap, H_shift)
        vals_shift_dense = TensorKit.SectorDict(c => sort(v; by = real) for (c, v) in eigvals(H_shift_dense))
        for (sector, vals) in vals_dense
            sector′ = only(sector ⊗ U1Irrep(-1))
            @test vals ≈ vals_shift_dense[sector′]
        end
    end

end
