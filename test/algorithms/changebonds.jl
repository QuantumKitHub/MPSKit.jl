println("
-----------------------------
|   Changebonds tests       |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ

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
