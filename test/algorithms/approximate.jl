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

verbosity_conv = 1

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
