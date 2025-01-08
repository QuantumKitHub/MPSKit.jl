println("
------------------
|   Algorithms   |
------------------
")
module TestAlgorithms

using ..TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

verbosity_full = 5
verbosity_conv = 1

@testset "FiniteMPS groundstate" verbose = true begin
    tol = 1e-8
    g = 4.0
    D = 6
    L = 10

    H = force_planar(transverse_field_ising(; g, L))

    @testset "DMRG" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)

        # test logging
        ψ, envs, δ = find_groundstate(ψ₀, H,
                                      DMRG(; verbosity=verbosity_full, maxiter=2))

        ψ, envs, δ = find_groundstate(ψ, H,
                                      DMRG(; verbosity=verbosity_conv, maxiter=10),
                                      envs)
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀ && v < 1e-2
    end

    @testset "DMRG2" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)
        trscheme = truncdim(floor(Int, D * 1.5))
        # test logging
        ψ, envs, δ = find_groundstate(ψ₀, H,
                                      DMRG2(; verbosity=verbosity_full, maxiter=2,
                                            trscheme))

        ψ, envs, δ = find_groundstate(ψ, H,
                                      DMRG2(; verbosity=verbosity_conv, maxiter=10,
                                            trscheme), envs)
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀ && v < 1e-2
    end

    @testset "GradientGrassmann" begin
        ψ₀ = FiniteMPS(randn, ComplexF64, 10, ℙ^2, ℙ^D)
        v₀ = variance(ψ₀, H)

        # test logging
        ψ, envs, δ = find_groundstate(ψ₀, H,
                                      GradientGrassmann(; verbosity=verbosity_full,
                                                        maxiter=2))

        ψ, envs, δ = find_groundstate(ψ, H,
                                      GradientGrassmann(; verbosity=verbosity_conv,
                                                        maxiter=50),
                                      envs)
        v = variance(ψ, H)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀ && v < 1e-2
    end
end

@testset "InfiniteMPS groundstate" verbose = true begin
    tol = 1e-8
    g = 4.0
    D = 6

    H_ref = force_planar(transverse_field_ising(; g))
    ψ = InfiniteMPS(ℙ^2, ℙ^D)
    v₀ = variance(ψ, H_ref)

    @testset "VUMPS" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        # test logging
        ψ, envs, δ = find_groundstate(ψ, H,
                                      VUMPS(; tol, verbosity=verbosity_full, maxiter=2))

        ψ, envs, δ = find_groundstate(ψ, H, VUMPS(; tol, verbosity=verbosity_conv))
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀
        @test v < 1e-2
    end

    @testset "IDMRG1" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        # test logging
        ψ, envs, δ = find_groundstate(ψ, H,
                                      IDMRG1(; tol, verbosity=verbosity_full, maxiter=2))

        ψ, envs, δ = find_groundstate(ψ, H, IDMRG1(; tol, verbosity=verbosity_conv))
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀
        @test v < 1e-2
    end

    @testset "IDMRG2" begin
        ψ = repeat(InfiniteMPS(ℙ^2, ℙ^D), 2)
        H = repeat(H_ref, 2)

        trscheme = truncbelow(1e-8)

        # test logging
        ψ, envs, δ = find_groundstate(ψ, H,
                                      IDMRG2(; tol, verbosity=verbosity_full, maxiter=2,
                                             trscheme))

        ψ, envs, δ = find_groundstate(ψ, H,
                                      IDMRG2(; tol, verbosity=verbosity_conv, trscheme))
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀
        @test v < 1e-2
    end

    @testset "GradientGrassmann" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        # test logging
        ψ, envs, δ = find_groundstate(ψ, H,
                                      GradientGrassmann(; tol, verbosity=verbosity_full,
                                                        maxiter=2))

        ψ, envs, δ = find_groundstate(ψ, H,
                                      GradientGrassmann(; tol, verbosity=verbosity_conv))
        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀
        @test v < 1e-2
    end

    @testset "Combination" for unit_cell_size in [1, 3]
        ψ = unit_cell_size == 1 ? InfiniteMPS(ℙ^2, ℙ^D) : repeat(ψ, unit_cell_size)
        H = repeat(H_ref, unit_cell_size)

        alg = VUMPS(; tol=100 * tol, verbosity=verbosity_conv, maxiter=10) &
              GradientGrassmann(; tol, verbosity=verbosity_conv, maxiter=50)
        ψ, envs, δ = find_groundstate(ψ, H, alg)

        v = variance(ψ, H, envs)

        # test using low variance
        @test sum(δ) ≈ 0 atol = 1e-3
        @test v < v₀
        @test v < 1e-2
    end
end

@testset "LazySum FiniteMPS groundstate" verbose = true begin
    tol = 1e-8
    D = 15
    atol = 1e-2
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
    ψ₀, = find_groundstate(ψ₀, H; tol, verbosity=1)

    @testset "DMRG" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy,
                                      DMRG(; tol, verbosity=verbosity_full, maxiter=1))

        # compare states
        alg = DMRG(; tol, verbosity=verbosity_conv)
        ψ, envs, δ = find_groundstate(ψ, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end

    @testset "DMRG2" begin
        # test logging passes
        trscheme = truncdim(floor(Int, D * 1.5))
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy,
                                      DMRG2(; tol, verbosity=verbosity_full, maxiter=1,
                                            trscheme))

        # compare states
        alg = DMRG2(; tol, verbosity=verbosity_conv, trscheme)
        ψ, = find_groundstate(ψ₀, H, alg)
        ψ_lazy, envs, δ = find_groundstate(ψ₀, H_lazy, alg)

        @test abs(dot(ψ₀, ψ_lazy)) ≈ 1 atol = atol
    end

    @testset "GradientGrassmann" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy,
                                      GradientGrassmann(; tol, verbosity=verbosity_full,
                                                        maxiter=2))

        # compare states
        alg = GradientGrassmann(; tol, verbosity=verbosity_conv)
        ψ, = find_groundstate(ψ₀, H, alg)
        ψ_lazy, envs, δ = find_groundstate(ψ₀, H_lazy, alg)

        @test abs(dot(ψ₀, ψ_lazy)) ≈ 1 atol = atol
    end
end

@testset "LazySum InfiniteMPS groundstate" verbose = true begin
    tol = 1e-8
    D = 16
    atol = 1e-2

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
    ψ₀, = find_groundstate(ψ₀, H; tol, verbosity=1)

    @testset "VUMPS" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy,
                                      VUMPS(; tol, verbosity=verbosity_full, maxiter=2))

        # compare states
        alg = VUMPS(; tol, verbosity=verbosity_conv)
        ψ, envs, δ = find_groundstate(ψ, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end

    @testset "IDMRG1" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy,
                                      IDMRG1(; tol, verbosity=verbosity_full, maxiter=2))

        # compare states
        alg = IDMRG1(; tol, verbosity=verbosity_conv, maxiter=300)
        ψ, envs, δ = find_groundstate(ψ, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end

    @testset "IDMRG2" begin
        ψ₀′ = repeat(ψ₀, 2)
        H_lazy′ = repeat(H_lazy, 2)
        H′ = repeat(H, 2)

        trscheme = truncdim(floor(Int, D * 1.5))
        # test logging passes
        ψ, envs, δ = find_groundstate(ψ₀′, H_lazy′,
                                      IDMRG2(; tol, verbosity=verbosity_full, maxiter=2,
                                             trscheme))

        # compare states
        alg = IDMRG2(; tol, verbosity=verbosity_conv, trscheme)
        ψ, envs, δ = find_groundstate(ψ, H_lazy′, alg)

        @test abs(dot(ψ₀′, ψ)) ≈ 1 atol = atol
    end

    @testset "GradientGrassmann" begin
        # test logging passes
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy,
                                      GradientGrassmann(; tol, verbosity=verbosity_full,
                                                        maxiter=2))

        # compare states
        alg = GradientGrassmann(; tol, verbosity=verbosity_conv)
        ψ, envs, δ = find_groundstate(ψ₀, H_lazy, alg)

        @test abs(dot(ψ₀, ψ)) ≈ 1 atol = atol
    end
end

@testset "timestep" verbose = true begin
    dt = 0.1
    algs = [TDVP(), TDVP2()]
    L = 10

    H = force_planar(heisenberg_XXX(; spin=1 // 2, L))
    ψ₀ = FiniteMPS(L, ℙ^2, ℙ^1)
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E = expectation_value(ψ, H, envs)
        @test E₀ ≈ E atol = 1e-2
    end

    Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])

    @testset "Finite LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, alg)
        E = expectation_value(ψ, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1e-2
    end

    Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

    @testset "Finite TimeDependent LazySum $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in
                                                                                   algs
        ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, alg)
        E = expectation_value(ψ, Ht(1.0), envs)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, alg)
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test E ≈ Et atol = 1e-8
    end

    H = repeat(force_planar(heisenberg_XXX(; spin=1)), 2)
    ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
    E₀ = expectation_value(ψ₀, H)

    @testset "Infinite TDVP" begin
        ψ, envs = timestep(ψ₀, H, 0.0, dt, TDVP())
        E = expectation_value(ψ, H, envs)
        @test E₀ ≈ E atol = 1e-2
    end

    Hlazy = LazySum([3 * deepcopy(H), 1.55 * deepcopy(H), -0.1 * deepcopy(H)])

    @testset "Infinite LazySum TDVP" begin
        ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, TDVP())
        E = expectation_value(ψ, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1e-2
    end

    Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

    @testset "Infinite TimeDependent LazySum" begin
        ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, TDVP())
        E = expectation_value(ψ, Ht(1.0), envs)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, TDVP())
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test E ≈ Et atol = 1e-8
    end
end

@testset "time_evolve" verbose = true begin
    t_span = 0:0.1:0.1
    algs = [TDVP(), TDVP2()]

    L = 10
    H = force_planar(heisenberg_XXX(; spin=1 // 2, L))
    ψ₀ = FiniteMPS(L, ℙ^2, ℙ^1)
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(alg isa TDVP ? "TDVP" : "TDVP2")" for alg in algs
        ψ, envs = time_evolve(ψ₀, H, t_span, alg)
        E = expectation_value(ψ, H, envs)
        @test E₀ ≈ E atol = 1e-2
    end

    H = repeat(force_planar(heisenberg_XXX(; spin=1)), 2)
    ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
    E₀ = expectation_value(ψ₀, H)

    @testset "Infinite TDVP" begin
        ψ, envs = time_evolve(ψ₀, H, t_span, TDVP())
        E = expectation_value(ψ, H, envs)
        @test E₀ ≈ E atol = 1e-2
    end
end

@testset "leading_boundary" verbose = true begin
    tol = 1e-4
    verbosity = verbosity_conv
    algs = [VUMPS(; tol, verbosity), VOMPS(; tol, verbosity),
            GradientGrassmann(; tol, verbosity)]
    mpo = force_planar(classical_ising())

    ψ₀ = InfiniteMPS([ℙ^2], [ℙ^10])
    @testset "Infinite $i" for (i, alg) in enumerate(algs)
        ψ, envs = leading_boundary(ψ₀, mpo, alg)
        ψ, envs = changebonds(ψ, mpo, OptimalExpand(; trscheme=truncdim(3)), envs)
        ψ, envs = leading_boundary(ψ, mpo, alg)

        @test dim(space(ψ.AL[1, 1], 1)) == dim(space(ψ₀.AL[1, 1], 1)) + 3
        @test expectation_value(ψ, mpo, envs) ≈ 2.5337 atol = 1e-3
    end
end

@testset "excitations" verbose = true begin
    @testset "infinite (ham)" begin
        H = repeat(force_planar(heisenberg_XXX()), 2)
        ψ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^48, ℙ^48])
        ψ, envs, _ = find_groundstate(ψ, H; maxiter=400, verbosity=verbosity_conv,
                                      tol=1e-10)
        energies, ϕs = excitations(H, QuasiparticleAnsatz(), Float64(pi), ψ, envs)
        @test energies[1] ≈ 0.41047925 atol = 1e-4
        @test variance(ϕs[1], H) < 1e-8
    end
    @testset "infinite (mpo)" begin
        H = repeat(sixvertex(), 2)
        ψ = InfiniteMPS([ℂ^2, ℂ^2], [ℂ^10, ℂ^10])
        ψ, envs, _ = leading_boundary(ψ, H,
                                      VUMPS(; maxiter=400, verbosity=verbosity_conv,
                                            tol=1e-10))
        energies, ϕs = excitations(H, QuasiparticleAnsatz(), [0.0, Float64(pi / 2)], ψ,
                                   envs; verbosity=0)
        @test abs(energies[1]) > abs(energies[2]) # has a minimum at pi/2
    end

    @testset "finite" begin
        verbosity = verbosity_conv
        H_inf = force_planar(transverse_field_ising())
        ψ_inf = InfiniteMPS([ℙ^2], [ℙ^10])
        ψ_inf, envs, _ = find_groundstate(ψ_inf, H_inf; maxiter=400, verbosity, tol=1e-9)
        energies, ϕs = excitations(H_inf, QuasiparticleAnsatz(), 0.0, ψ_inf, envs)
        inf_en = energies[1]

        fin_en = map([20, 10]) do len
            H = force_planar(transverse_field_ising(; L=len))
            ψ = FiniteMPS(rand, ComplexF64, len, ℙ^2, ℙ^10)
            ψ, envs, = find_groundstate(ψ, H; verbosity)

            # find energy with quasiparticle ansatz
            energies_QP, ϕs = excitations(H, QuasiparticleAnsatz(), ψ, envs)
            @test variance(ϕs[1], H) < 1e-6

            # find energy with normal dmrg
            for gsalg in (DMRG(; verbosity, tol=1e-6),
                          DMRG2(; verbosity, tol=1e-6, trscheme=truncbelow(1e-4)))
                energies_dm, _ = excitations(H, FiniteExcited(; gsalg), ψ)
                @test energies_dm[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1e-4
            end

            # find energy with Chepiga ansatz
            energies_ch, _ = excitations(H, ChepigaAnsatz(), ψ, envs)
            @test energies_ch[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1e-4
            energies_ch2, _ = excitations(H, ChepigaAnsatz2(), ψ, envs)
            @test energies_ch2[1] ≈ energies_QP[1] + expectation_value(ψ, H, envs) atol = 1e-4
            return energies_QP[1]
        end

        @test issorted(abs.(fin_en .- inf_en))
    end
end

@testset "changebonds $((pspace,Dspace))" verbose = true for (pspace, Dspace) in
                                                             [(ℙ^4, ℙ^3),
                                                              (Rep[SU₂](1 => 1),
                                                               Rep[SU₂](0 => 2, 1 => 2,
                                                                        2 => 1))]
    @testset "mpo" begin
        #random nn interaction
        nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'
        H = InfiniteMPOHamiltonian(PeriodicVector(fill(pspace, 1)), (1, 2) => nn)
        Δt = 0.1
        expH = make_time_mpo(H, Δt, WII())

        O = DenseMPO(expH)
        Op = periodic_boundary_conditions(O, 10)
        Op′ = changebonds(Op, SvdCut(; trscheme=truncdim(5)))

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

            state_re = changebonds(state,
                                   RandExpand(;
                                              trscheme=truncdim(dim(Dspace) * dim(Dspace))))
            @test dot(state, state_re) ≈ 1 atol = 1e-8
        end
        # test optimal_expand
        for unit_cell_size in 2:3
            H = repeat(H0, unit_cell_size)
            state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

            state_oe, _ = changebonds(state,
                                      H,
                                      OptimalExpand(;
                                                    trscheme=truncdim(dim(Dspace) *
                                                                      dim(Dspace))))
            @test dot(state, state_oe) ≈ 1 atol = 1e-8
        end
        # test VUMPSSvdCut
        for unit_cell_size in [1, 2, 3, 4]
            H = repeat(H0, unit_cell_size)
            state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

            state_vs, _ = changebonds(state, H,
                                      VUMPSSvdCut(; trscheme=notrunc()))
            @test dim(left_virtualspace(state, 1)) < dim(left_virtualspace(state_vs, 1))

            state_vs_tr = changebonds(state_vs, SvdCut(; trscheme=truncdim(dim(Dspace))))
            @test dim(right_virtualspace(state_vs_tr, 1)) <
                  dim(right_virtualspace(state_vs, 1))
        end
    end

    @testset "finite mps" begin
        #random nn interaction
        L = 10
        nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'
        H = FiniteMPOHamiltonian(fill(pspace, L), (i, i + 1) => nn for i in 1:(L - 1))

        state = FiniteMPS(L, pspace, Dspace)

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        state_oe, _ = changebonds(state, H,
                                  OptimalExpand(;
                                                trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_tr = changebonds(state_oe, SvdCut(; trscheme=truncdim(dim(Dspace))))

        @test dim(left_virtualspace(state_tr, 5)) < dim(left_virtualspace(state_oe, 5))
    end

    @testset "MultilineMPS" begin
        o = rand(ComplexF64, pspace * pspace, pspace * pspace)
        mpo = MultilineMPO(o)

        t = rand(ComplexF64, Dspace * pspace, Dspace)
        state = MultilineMPS(fill(t, 1, 1))

        state_re = changebonds(state,
                               RandExpand(; trscheme=truncdim(dim(Dspace) * dim(Dspace))))
        @test dot(state, state_re) ≈ 1 atol = 1e-8

        state_oe, _ = changebonds(state, mpo,
                                  OptimalExpand(;
                                                trscheme=truncdim(dim(Dspace) *
                                                                  dim(Dspace))))
        @test dot(state, state_oe) ≈ 1 atol = 1e-8

        state_tr = changebonds(state_oe, SvdCut(; trscheme=truncdim(dim(Dspace))))

        @test dim(left_virtualspace(state_tr, 1, 1)) <
              dim(left_virtualspace(state_oe, 1, 1))
    end
end

@testset "Dynamical DMRG" verbose = true begin
    ham = force_planar(-1.0 * transverse_field_ising(; g=-4.0))
    gs, = find_groundstate(InfiniteMPS([ℙ^2], [ℙ^10]), ham, VUMPS(; verbosity=0))
    window = WindowMPS(gs, copy.([gs.AC[1]; [gs.AR[i] for i in 2:10]]), gs)

    szd = force_planar(S_z())
    @test [expectation_value(gs, i => szd) for i in 1:length(window)] ≈
          [expectation_value(window, i => szd) for i in 1:length(window)] atol = 1e-10

    openham = open_boundary_conditions(ham, length(window.window))
    polepos = expectation_value(window.window, openham,
                                environments(window.window, openham))

    vals = (-0.5:0.2:0.5) .+ polepos
    eta = 0.3im

    predicted = [1 / (v + eta - polepos) for v in vals]

    @testset "Flavour $f" for f in (Jeckelmann(), NaiveInvert())
        alg = DynamicalDMRG(; flavour=f, verbosity=0, tol=1e-8)
        data = map(vals) do v
            result, = propagator(window.window, v + eta, openham, alg)
            return result
        end
        @test data ≈ predicted atol = 1e-8
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
        ψ, envs, = find_groundstate(ψ, H, VUMPS(; maxiter=100, verbosity=0))

        numerical_susceptibility = fidelity_susceptibility(ψ, H, [H_X], envs; maxiter=10)
        @test numerical_susceptibility[1, 1] ≈ analytical_susceptibility(λ) atol = 1e-2

        # test if the finite fid sus approximates the analytical one with increasing system size
        fin_en = map([20, 15, 10]) do L
            Hfin = open_boundary_conditions(hamiltonian(λ), L)
            H_Xfin = open_boundary_conditions(H_X, L)
            ψ = FiniteMPS(rand, ComplexF64, L, ℂ^2, ℂ^16)
            ψ, envs, = find_groundstate(ψ, Hfin, DMRG(; verbosity=0))
            numerical_susceptibility = fidelity_susceptibility(ψ, Hfin, [H_Xfin], envs;
                                                               maxiter=10)
            return numerical_susceptibility[1, 1] / L
        end
        @test issorted(abs.(fin_en .- analytical_susceptibility(λ)))
    end
end

# stub tests
@testset "correlation length / entropy" begin
    ψ = InfiniteMPS([ℙ^2], [ℙ^10])
    H = force_planar(transverse_field_ising())
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity=0))
    len_crit = correlation_length(ψ)[1]
    entrop_crit = entropy(ψ)

    H = force_planar(transverse_field_ising(; g=4))
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity=0))
    len_gapped = correlation_length(ψ)[1]
    entrop_gapped = entropy(ψ)

    @test len_crit > len_gapped
    @test real(entrop_crit) > real(entrop_gapped)
end

@testset "expectation value / correlator" begin
    g = 4.0
    ψ = InfiniteMPS(ℂ^2, ℂ^10)
    H = transverse_field_ising(; g)
    ψ, = find_groundstate(ψ, H, VUMPS(; verbosity=0))

    @test expectation_value(ψ, H) ≈
          expectation_value(ψ, 1 => -g * S_x()) + expectation_value(ψ, (1, 2) => -S_zz())
    Z_mpo = MPSKit.add_util_leg(S_z())
    G = correlator(ψ, Z_mpo, Z_mpo, 1, 2:5)
    G2 = correlator(ψ, S_zz(), 1, 3:2:5)
    @test isapprox(G[2], G2[1], atol=1e-2)
    @test isapprox(last(G), last(G2), atol=1e-2)
    @test isapprox(G[1], expectation_value(ψ, (1, 2) => S_zz()), atol=1e-2)
    @test isapprox(G[2], expectation_value(ψ, (1, 3) => S_zz()), atol=1e-2)
end

@testset "approximate" verbose = true begin
    verbosity = 0
    @testset "mpo * infinite ≈ infinite" begin
        ψ = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^10, ℙ^10])
        H = force_planar(repeat(transverse_field_ising(; g=4), 2))

        dt = 1e-3
        sW1 = make_time_mpo(H, dt, TaylorCluster(; N=3))
        sW2 = make_time_mpo(H, dt, WII())
        W1 = DenseMPO(sW1)
        W2 = DenseMPO(sW2)

        ψ1, _ = approximate(ψ, (sW1, ψ), VOMPS(; verbosity))
        ψ2, _ = approximate(ψ, (W2, ψ), VOMPS(; verbosity))
        ψ3, _ = approximate(ψ, (W1, ψ), IDMRG1(; verbosity))
        ψ4, _ = approximate(ψ, (sW2, ψ), IDMRG2(; trscheme=truncdim(20), verbosity))
        ψ5, _ = timestep(ψ, H, 0.0, dt, TDVP())
        ψ6 = changebonds(W1 * ψ, SvdCut(; trscheme=truncdim(10)))

        @test abs(dot(ψ1, ψ5)) ≈ 1.0 atol = dt
        @test abs(dot(ψ3, ψ5)) ≈ 1.0 atol = dt
        @test abs(dot(ψ6, ψ5)) ≈ 1.0 atol = dt
        @test abs(dot(ψ2, ψ4)) ≈ 1.0 atol = dt

        nW1 = changebonds(W1, SvdCut(; trscheme=truncbelow(dt))) # this should be a trivial mpo now
        @test dim(space(nW1[1], 1)) == 1
    end

    finite_algs = [DMRG(; verbosity), DMRG2(; verbosity, trscheme=truncdim(10))]
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

        H = transverse_field_ising(; g=4.0, L)
        τ = 1e-3

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

        O = finite_classical_ising(L)
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
end

end
