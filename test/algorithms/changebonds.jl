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
maxbond(ψ) = maximum(i -> dim(left_virtualspace(ψ, i)), 2:length(ψ))


@testset "MPO $(spacetype(pspace))" for (pspace, Dspace) in spacelist
    nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    H0 = InfiniteMPOHamiltonian(PeriodicVector(fill(pspace, 1)), (1, 2) => nn)
    Δt = 0.1
    expH = make_time_mpo(H0, Δt, WII())

    O = MPSKit.DenseMPO(expH)
    Op = periodic_boundary_conditions(O, 10)
    Op′ = changebonds(Op, SvdCut(; trunc = truncrank(5)))

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
            state, RandExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)))
        )
        @test dot(state, state_re) ≈ 1 atol = 1.0e-8
    end
    # test optimal_expand
    for unit_cell_size in 2:3
        H = repeat(H0, unit_cell_size)
        state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

        state_oe, _ = changebonds(
            state, H, OptimalExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)))
        )
        @test dot(state, state_oe) ≈ 1 atol = 1.0e-8
    end
    # test VUMPSSvdCut
    for unit_cell_size in [1, 2, 3, 4]
        H = repeat(H0, unit_cell_size)
        state = InfiniteMPS(fill(pspace, unit_cell_size), fill(Dspace, unit_cell_size))

        state_vs, _ = changebonds(state, H, VUMPSSvdCut(; trunc = notrunc()))
        @test dim(left_virtualspace(state, 1)) < dim(left_virtualspace(state_vs, 1))

        state_vs_tr = changebonds(state_vs, SvdCut(; trunc = truncrank(dim(Dspace))))
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
        state, RandExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)))
    )
    @test maxbond(state_re) > maxbond(state)
    @test dot(state, state_re) ≈ 1 atol = 1.0e-8

    state_oe, _ = changebonds(
        state, H, OptimalExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)))
    )
    @test maxbond(state_oe) > maxbond(state)
    @test dot(state, state_oe) ≈ 1 atol = 1.0e-8

    state_se, _ = changebonds(
        state, H,
        SketchedExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)), oversampling = 4)
    )
    @test maxbond(state_se) > maxbond(state)
    @test dot(state, state_se) ≈ 1 atol = 1.0e-8

    state_tr = changebonds(state_oe, SvdCut(; trunc = truncrank(dim(Dspace))))
    @test dim(left_virtualspace(state_tr, 5)) < dim(left_virtualspace(state_oe, 5))
end

# Regression: CBE expanders must not crash on bonds with no expansion content. A charged
# single-particle fermionic (fℤ₂-graded) state carries only one parity sector on the bonds away
# from the particle, so the projected two-site update `adjoint(NL) * AC2 * adjoint(NR)` is
# *structurally exactly zero* there. `OptimalExpand`/`RandExpand` used to `normalize!` that zero
# tensor, producing NaNs that crashed the SVD (`ArgumentError: invalid argument #4 to LAPACK
# call`); they must instead skip such bonds, leaving the state unchanged.
@testset "CBE zero-content bonds (graded)" begin
    L = 6
    H = kitaev_model(ComplexF64, Trivial; t = 1.0, mu = 0, Delta = 0, L = L)
    P = physicalspace(H)[1]
    S = typeof(P)
    Vtriv = oneunit(P)
    Vodd = S(c => dim(P, c) for c in sectors(P) if !isone(c))
    # single particle created on the last site: interior bonds carry only the even sector, so
    # the expansion content on them is exactly zero
    state = FiniteMPS(
        [isometry(ComplexF64, (k <= L ? Vtriv : Vodd) ⊗ P, (k < L ? Vtriv : Vodd)) for k in 1:L]
    )

    state_oe, _ = changebonds(state, H, OptimalExpand(; trunc = truncrank(2)))
    @test abs(dot(state, state_oe)) ≈ 1 atol = 1.0e-8

    state_re = changebonds(state, RandExpand(; trunc = truncrank(2)))
    @test abs(dot(state, state_re)) ≈ 1 atol = 1.0e-8
end

# density-matrix-style MPS: each site carries two physical legs (ket ⊗ bra). The operator-free
# bond-change algorithms (`RandExpand` expansion, `SvdCut` truncation) must handle the extra
# physical leg. Operator-based expanders (`OptimalExpand`/`SketchedExpand`) make use of a
# one-sided MPO application on the first physical leg.
@testset "Density-matrix FiniteMPS $(spacetype(pcomp))" for (pcomp, Dspace) in [
        (ℙ^2 ⊗ (ℙ^2)', ℙ^6),
        (Rep[SU₂](1 // 2 => 1) ⊗ Rep[SU₂](1 // 2 => 1)', Rep[SU₂](0 => 4, 1 => 3)),
    ]
    Random.seed!(2468)
    L = 8

    ψ = FiniteMPS(rand, ComplexF64, fill(pcomp, L), Dspace)
    @test numind(ψ.AC[L ÷ 2]) == 4    # two physical legs + two virtual legs

    # RandExpand grows the bond while preserving the state (norm-preserving expansion)
    ψ_re = changebonds(ψ, RandExpand(; trunc = truncrank(dim(Dspace) * 2)))
    @test numind(ψ_re.AC[L ÷ 2]) == 4
    @test abs(dot(ψ, ψ_re)) ≈ 1 atol = 1.0e-8
    @test maxbond(ψ_re) > maxbond(ψ)

    # SvdCut truncates the enlarged bond back down, leaving a normalized state
    ψ_tr = changebonds(ψ_re, SvdCut(; trunc = truncrank(dim(Dspace))))
    @test maxbond(ψ_tr) < maxbond(ψ_re)
    @test abs(dot(ψ, ψ_tr)) ≈ 1 atol = 1.0e-8

    # use random time-evolution MPO to test operator-based expanders
    pspace = pcomp[1]
    nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
    nn += nn'
    H = FiniteMPOHamiltonian(fill(pspace, L), (i, i + 1) => nn for i in 1:(L - 1))
    beta = 0.1
    O = make_time_mpo(H, beta, TaylorCluster(; N = 2); imaginary_evolution = true)

    @show typeof(O)
    @show left_virtualspace(O)
    @show physicalspace(O)


    ψ_oe, _ = changebonds(
        ψ, O, OptimalExpand(; trunc = truncrank(dim(Dspace) * 2))
    )
    @test maxbond(ψ_oe) > maxbond(ψ)
    @test dot(ψ, ψ_oe) ≈ 1 atol = 1.0e-8

    ψ_se, _ = changebonds(
        ψ, O,
        SketchedExpand(; trunc = truncrank(dim(Dspace) * 2), oversampling = 4)
    )
    @test maxbond(ψ_se) > maxbond(ψ)
    @test dot(ψ, ψ_se) ≈ 1 atol = 1.0e-8
end

@testset "MultilineMPS $(spacetype(pspace))" for (pspace, Dspace) in spacelist
    o = rand(ComplexF64, pspace * pspace, pspace * pspace)
    mpo = MultilineMPO(o)

    t = rand(ComplexF64, Dspace * pspace, Dspace)
    state = MultilineMPS(fill(t, 1, 1))

    state_re = changebonds(
        state, RandExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)))
    )
    @test dot(state, state_re) ≈ 1 atol = 1.0e-8

    state_oe, _ = changebonds(
        state, mpo, OptimalExpand(; trunc = truncrank(dim(Dspace) * dim(Dspace)))
    )
    @test dot(state, state_oe) ≈ 1 atol = 1.0e-8

    state_tr = changebonds(state_oe, SvdCut(; trunc = truncrank(dim(Dspace))))

    @test dim(left_virtualspace(state_tr, 1, 1)) < dim(left_virtualspace(state_oe, 1, 1))
end
