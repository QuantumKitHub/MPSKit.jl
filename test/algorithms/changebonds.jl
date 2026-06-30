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
using Random

spacelist = [(ℙ^4, ℙ^3), (Rep[SU₂](1 => 1), Rep[SU₂](0 => 2, 1 => 2, 2 => 1))]


@testset "MPO $(spacetype(pspace))" for (pspace, Dspace) in spacelist
    nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    H0 = InfiniteMPOHamiltonian(PeriodicVector(fill(pspace, 1)), (1, 2) => nn)
    Δt = 0.1
    expH = make_time_mpo(H0, Δt, WII())

    O = MPSKit.DenseMPO(expH)
    Op = periodic_boundary_conditions(O, 10)
    Op′ = changebonds(Op, SvdCut(; trscheme = truncrank(5)))

    @test dim(space(Op′[5], 1)) < dim(space(Op[5], 1))
end

@testset "InfiniteMPS $(spacetype(pspace))" for (pspace, Dspace) in spacelist
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

@testset "FiniteMPS $(spacetype(pspace))" for (pspace, Dspace) in spacelist
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

    state_se, _ = changebonds(
        state, H,
        SketchedExpand(; trscheme = truncrank(dim(Dspace) * dim(Dspace)), oversampling = 4)
    )
    @test dot(state, state_se) ≈ 1 atol = 1.0e-8

    # `warmstart` seeds the new directions with the two-site gradient: the expansion is no longer
    # norm-preserving, but the resulting state is still normalized
    for (Exp, kw) in ((OptimalExpand, (;)), (SketchedExpand, (; oversampling = 4)))
        state_ws, _ = changebonds(
            state, H, Exp(; trscheme = truncrank(dim(Dspace) * dim(Dspace)), warmstart = true, kw...)
        )
        @test dot(state_ws, state_ws) ≈ 1 atol = 1.0e-8
        @test !isapprox(abs(dot(state, state_ws)), 1; atol = 1.0e-4)
    end

    state_tr = changebonds(state_oe, SvdCut(; trscheme = truncrank(dim(Dspace))))

    @test dim(left_virtualspace(state_tr, 5)) < dim(left_virtualspace(state_oe, 5))
end

# density-matrix-style MPS: each site carries two physical legs (ket ⊗ bra). The operator-free
# bond-change algorithms (`RandExpand` expansion, `SvdCut` truncation) must handle the extra
# physical leg. Operator-based expanders (`OptimalExpand`/`SketchedExpand`) are not covered here
# because `FiniteMPOHamiltonian`/`FiniteMPO` only support a single physical leg per site.
@testset "Density-matrix FiniteMPS $(spacetype(pcomp))" for (pcomp, Dspace) in [
        (ℙ^2 ⊗ (ℙ^2)', ℙ^6),
        (Rep[SU₂](1 // 2 => 1) ⊗ Rep[SU₂](1 // 2 => 1)', Rep[SU₂](0 => 4, 1 => 3)),
    ]
    Random.seed!(2468)
    L = 8
    maxbond(ψ) = maximum(i -> dim(left_virtualspace(ψ, i)), 2:length(ψ))

    ψ = FiniteMPS(rand, ComplexF64, fill(pcomp, L), Dspace)
    @test numind(ψ.AC[L ÷ 2]) == 4    # two physical legs + two virtual legs

    # RandExpand grows the bond while preserving the state (norm-preserving expansion)
    ψ_re = changebonds(ψ, RandExpand(; trscheme = truncrank(dim(Dspace) * 2)))
    @test numind(ψ_re.AC[L ÷ 2]) == 4
    @test abs(dot(ψ, ψ_re)) ≈ 1 atol = 1.0e-8
    @test maxbond(ψ_re) > maxbond(ψ)

    # SvdCut truncates the enlarged bond back down, leaving a normalized state
    ψ_tr = changebonds(ψ_re, SvdCut(; trscheme = truncrank(dim(Dspace))))
    @test maxbond(ψ_tr) < maxbond(ψ_re)
    @test abs(dot(ψ_tr, ψ_tr)) ≈ 1 atol = 1.0e-8
end

@testset "MultilineMPS $(spacetype(pspace))" for (pspace, Dspace) in spacelist
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
