println("
-----------------------------
|   Approximation tests     |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using Random

verbosity_conv = 1

# fixtures for the `Zipup` testsets
zipup_spacelist = [
    (ℙ^4, ℙ^3, 4),
    (Rep[SU₂](1 => 1), Rep[SU₂](0 => 2, 1 => 2, 2 => 1), 8),
]

function _random_mpo_mps(pspace, Dspace, L; elt = ComplexF64)
    Random.seed!(1357)
    Wspace = Dspace
    Vspaces = [oneunit(Wspace); fill(Wspace, L - 1); oneunit(Wspace)]
    O = FiniteMPO(
        [rand(ComplexF64, Vspaces[i] ⊗ pspace ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
    )
    ψ = FiniteMPS(rand, elt, L, pspace, Dspace)
    return O, ψ
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
        # VOMPS runs the AC and C projections of a site concurrently, so the allocator serving them
        # is shared; the serial arm takes the other branch of that choice
        ψ2, _ = with_scheduler(MPSKit.SerialScheduler()) do
            return approximate(ψ0, (W2, ψ), VOMPS(; verbosity))
        end

        ψ3, _ = approximate(ψ0, (W1, ψ), IDMRG(; verbosity))
        ψ4, _ = approximate(ψ0, (sW2, ψ), IDMRG2(; trunc = truncrank(12), verbosity))
        ψ5, _ = timestep(ψ, H, 0.0, dt, TDVP())
        ψ6 = changebonds(W1 * ψ, SvdCut(; trunc = truncrank(12)))

        @test abs(dot(ψ1, ψ5)) ≈ 1.0 atol = dt
        @test abs(dot(ψ3, ψ5)) ≈ 1.0 atol = dt
        @test abs(dot(ψ6, ψ5)) ≈ 1.0 atol = dt
        @test abs(dot(ψ2, ψ4)) ≈ 1.0 atol = dt

        nW1 = changebonds(W1, SvdCut(; trunc = trunctol(; atol = dt))) # this should be a trivial mpo now
        @test dim(space(nW1[1], 1)) == 1
    end

    finite_algs = [DMRG(; verbosity), DMRG2(; verbosity, trunc = truncrank(10))]
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

    @testset "Finite MPO-MPS zip-up $(spacetype(pspace))" for (pspace, Dspace, _) in zipup_spacelist
        L = 6

        # dense MPO
        O, ψ = _random_mpo_mps(pspace, Dspace, L)
        O_copy = copy(O)
        ψ_copy = copy(ψ)

        trunc = trunctol(; atol = 1.0e-10)
        ref = changebonds(O * ψ, SvdCut(; trunc); normalize = false)

        # both sweep directions, with and without the zip-down pass
        for left_to_right in (true, false), trunc′ in (trunc, (notrunc(), trunc))
            got, info = approximate((O, ψ), Zipup(; trunc = trunc′, left_to_right))
            @test norm(ref - got) / norm(ref) < 1.0e-10
            @test info.ϵ_max < 1.0e-10
            @test isnothing(info.converged) && isnothing(info.normres)
        end

        @test norm(ψ - ψ_copy) < 1.0e-12
        @test all(i -> norm(O[i] - O_copy[i]) < 1.0e-12, 1:length(O))

        # sparse MPO
        nn = rand(ComplexF64, pspace * pspace, pspace * pspace)
        nn += nn'
        H = FiniteMPOHamiltonian(fill(pspace, L), (i, i + 1) => nn for i in 1:(L - 1))
        τ = 1.0e-3
        expH = make_time_mpo(H, τ, WI)

        # reference via TDVP
        ref_s, = timestep(ψ, H, 0.0, τ, TDVP())
        normalize!(ref_s)

        # both sweep directions, with and without the zip-down pass
        for left_to_right in (true, false), trunc′ in (trunc, (notrunc(), trunc))
            got_s, ϵ = approximate((expH, ψ), Zipup(; trunc = trunc′, left_to_right))
            normalize!(got_s)
            @test norm(ref_s - got_s) < 0.002
            @test norm(ψ - got_s) > 0.002
            @test ϵ < 1.0e-10
        end
    end

    @testset "Paeckel two-stage zip-up $(spacetype(pspace)), left_to_right = $left_to_right" for
        (pspace, Dspace, Dcut) in zipup_spacelist, left_to_right in (true, false)
        O, ψ = _random_mpo_mps(pspace, Dspace, 6)
        rtol = 1.0e-8
        final_trunc = truncrank(Dcut) & truncerror(; rtol)
        zipup_trunc = truncrank(2Dcut) & truncerror(; rtol = rtol / 10)

        ref_tr = changebonds(O * ψ, SvdCut(; trunc = final_trunc); normalize = false)
        got_one_sweep, info_one_sweep = approximate((O, ψ), Zipup(; trunc = final_trunc, left_to_right))
        got_two_sweep, _ = approximate(
            (O, ψ), Zipup(; trunc = (zipup_trunc, final_trunc), left_to_right)
        )
        @test info_one_sweep.ϵ_max > 0

        err_one_sweep = norm(ref_tr - got_one_sweep) / norm(ref_tr)
        err_two_sweep = norm(ref_tr - got_two_sweep) / norm(ref_tr)
        @test err_two_sweep < err_one_sweep / 2
        @test maximum(i -> dim(left_virtualspace(got_one_sweep, i)), 2:length(got_one_sweep)) <= Dcut
        @test maximum(i -> dim(left_virtualspace(got_two_sweep, i)), 2:length(got_two_sweep)) <= Dcut
    end

    @testset "In-place zip-up $(spacetype(pspace)), left_to_right = $left_to_right" for
        (pspace, Dspace, Dcut) in zipup_spacelist, left_to_right in (true, false)
        O, ψ = _random_mpo_mps(pspace, Dspace, 6)
        alg = Zipup(; trunc = (truncrank(2Dcut), truncrank(Dcut)), left_to_right)
        ref, info_ref = approximate((O, ψ), alg)

        # empty destination
        dst = similar(ψ, ComplexF64)
        got, info = approximate!(dst, (O, ψ), alg)
        @test got === dst
        @test norm(ref - got) / norm(ref) < 1.0e-12
        @test info.ϵ_max ≈ info_ref.ϵ_max

        # a destination with unrelated contents is overwritten entirely
        dst = FiniteMPS(rand, ComplexF64, length(ψ), pspace, oneunit(Dspace) ⊕ Dspace ⊕ Dspace)
        got, info = approximate!(dst, (O, ψ), alg)
        @test norm(ref - got) / norm(ref) < 1.0e-12
        @test info.ϵ_max ≈ info_ref.ϵ_max

        # the input may serve as its own destination
        got, info = approximate!(ψ, (O, ψ), alg)
        @test got === ψ
        @test norm(ref - got) / norm(ref) < 1.0e-12
        @test info.ϵ_max ≈ info_ref.ϵ_max
    end

    @testset "Zip-up with non-trivial boundary spaces $(spacetype(pspace))" for (pspace, Dspace, _) in zipup_spacelist
        Random.seed!(1357)
        L = 4
        Vspaces = fill(Dspace, L + 1)
        O = FiniteMPO(
            [rand(ComplexF64, Vspaces[i] ⊗ pspace ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
        )
        ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace; left = Dspace, right = Dspace)

        trunc = trunctol(; atol = 1.0e-12)
        got, _ = approximate((O, ψ), Zipup(; trunc))
        got_two_sweep, _ = approximate((O, ψ), Zipup(; trunc = (notrunc(), trunc)))

        # the boundary virtual spaces of the product are the fused ones, in both variants
        for ψ′ in (got, got_two_sweep)
            @test left_virtualspace(ψ′, 1) == fuse(left_virtualspace(ψ, 1) ⊗ left_virtualspace(O, 1))
            @test right_virtualspace(ψ′, L) == fuse(right_virtualspace(ψ, L) ⊗ right_virtualspace(O, L))
        end
        @test norm(got - got_two_sweep) / norm(got) < 1.0e-10
        @test norm(got) ≈ norm(O * ψ)
    end

    @testset "Zip-up with non-square MPO $(spacetype(pspace))" for (pspace, Dspace, _) in zipup_spacelist
        Random.seed!(1357)
        L = 4
        pspace′ = pspace ⊕ oneunit(pspace) # output physical space, different from the input one
        Vspaces = [oneunit(Dspace); fill(Dspace, L - 1); oneunit(Dspace)]
        O = FiniteMPO(
            [rand(ComplexF64, Vspaces[i] ⊗ pspace′ ← pspace ⊗ Vspaces[i + 1]) for i in 1:L]
        )
        ψ = FiniteMPS(rand, ComplexF64, L, pspace, Dspace)

        trunc = trunctol(; atol = 1.0e-12)
        ref = O * ψ
        got, _ = approximate((O, ψ), Zipup(; trunc))
        got_two_sweep, _ = approximate((O, ψ), Zipup(; trunc = (notrunc(), trunc)))

        @test all(i -> physicalspace(got, i) == pspace′, 1:L)
        @test norm(ref - got) / norm(ref) < 1.0e-10
        @test norm(ref - got_two_sweep) / norm(ref) < 1.0e-10

        # the reverse product is not defined
        @test_throws SpaceMismatch approximate((O, got), Zipup(; trunc))
    end

    @testset "Zip-up scalar type promotion $(spacetype(pspace))" for (pspace, Dspace, _) in zipup_spacelist
        O, ψ = _random_mpo_mps(pspace, Dspace, 6; elt = Float64)
        alg = Zipup(; trunc = trunctol(; atol = 1.0e-10))

        got, _ = approximate((O, ψ), alg)
        @test scalartype(got) === ComplexF64
        ref = changebonds(O * ψ, SvdCut(; trunc = trunctol(; atol = 1.0e-10)); normalize = false)
        @test norm(ref - got) / norm(ref) < 1.0e-10

        # a real destination cannot hold the complex result
        @test_throws ArgumentError approximate!(ψ, (O, ψ), alg)
    end
end
