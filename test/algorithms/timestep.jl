println("
-----------------------------
|   Time-stepping tests     |
-----------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using LinearAlgebra: norm, Diagonal
using Random

verbosity_full = 5
verbosity_conv = 1

@testset "timestep" verbose = true begin
    dt = 0.1
    algs = [TDVP(), TDVP2(; trscheme = truncrank(10))]
    L = 10

    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
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

@testset "Finite CBE-TDVP" verbose = true begin
    L = 10
    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    Dstart, Dcap, dt = 2, 16, 0.05

    # controlled bond expansion lets single-site TDVP grow the bond; the evolution should stay
    # unitary (norm-preserving) and energy-conserving while tracking the bond-adaptive TDVP2
    # reference better than fixed-bond single-site TDVP
    @testset "$(nameof(Exp))" for (Exp, kw) in
        ((OptimalExpand, (;)), (SketchedExpand, (; oversampling = 4)))
        Random.seed!(4)
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^Dstart))
        alg = TDVP(; alg_expand = Exp(; trscheme = truncrank(Dstart), kw...), trscheme = truncrank(Dcap))
        E₀ = real(expectation_value(ψ₀, H))

        ref, cbe, plain = ψ₀, ψ₀, ψ₀
        for _ in 1:6
            ref, = timestep(ref, H, 0.0, dt, TDVP2(; trscheme = truncrank(Dcap)))
            cbe, = timestep(cbe, H, 0.0, dt, alg)
            plain, = timestep(plain, H, 0.0, dt, TDVP())   # stuck at Dstart
        end

        @test norm(cbe) ≈ 1 atol = 1.0e-6
        @test real(expectation_value(cbe, H)) ≈ E₀ atol = 1.0e-2
        @test dim(left_virtualspace(cbe, L ÷ 2)) > Dstart
        @test dim(left_virtualspace(plain, L ÷ 2)) == Dstart
        @test abs(dot(ref, cbe)) > abs(dot(ref, plain))
    end

    # by default (`normalize = false`) the bond truncation preserves the norm, so it reflects the
    # discarded weight; `normalize = true` renormalizes each step. This is independent of
    # `imaginary_evolution`.
    @testset "norm handling" begin
        Random.seed!(6)
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^Dstart))
        # a deliberately lossy cap so the truncation discards weight every step
        lossy = TDVP(; alg_expand = OptimalExpand(; trscheme = truncrank(2)), trscheme = truncrank(2))

        ψrt = ψ₀
        for _ in 1:12
            ψrt, = timestep(ψrt, H, 0.0, 0.5, lossy)            # real time, norm preserved by default
        end
        @test norm(ψrt) < 1 - 1.0e-3                            # truncation loss is not renormalized away

        # imaginary-time, norm preserved by default: the weight is *not* pinned to unit norm
        # (imaginary-time evolution rescales the state, so the norm drifts away from 1)
        ψit = ψ₀
        for _ in 1:12
            ψit, = timestep(ψit, H, 0.0, 0.5, lossy; imaginary_evolution = true)
        end
        @test abs(norm(ψit) - 1) > 1.0e-3

        # imaginary-time with `normalize = true`: renormalized to unit norm each step
        ψn = ψ₀
        for _ in 1:12
            ψn, = timestep(ψn, H, 0.0, 0.5, lossy; imaginary_evolution = true, normalize = true)
        end
        @test norm(ψn) ≈ 1 atol = 1.0e-6
    end

    @testset "imaginary-time lowers energy" begin
        Random.seed!(5)
        ψ₀ = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^Dstart))
        alg = TDVP(; alg_expand = OptimalExpand(; trscheme = truncrank(Dstart)), trscheme = truncrank(Dcap))
        E₀ = real(expectation_value(ψ₀, H))
        ψ = ψ₀
        for _ in 1:8
            ψ, = timestep(ψ, H, 0.0, 0.1, alg; imaginary_evolution = true, normalize = true)
        end
        @test real(expectation_value(ψ, H)) < E₀
        @test dim(left_virtualspace(ψ, L ÷ 2)) > Dstart
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

# A single spin flip on the fully polarized vacuum of the XX chain (`XY_model` with `g = 0`)
# propagates as a free particle hopping on the open L-site chain, so the real-time correlator
# has a closed form:
#     C_nm(t) = ⟨vac| S^x_n e^{-iHt} S^x_m |vac⟩ = ¼ [exp(-i h t)]_{nm},
# where the single-particle hopping matrix `h` is tridiagonal (`h_{i,i±1} = -1`) and diagonalizes
# analytically on the open chain: eigenvalues `ε_k = -2 cos(kπ/(L+1))` with eigenvectors
# `φ_k(n) = √(2/(L+1)) sin(nkπ/(L+1))`. Being a genuine two-time correlator, this also exercises
# the norm-preserving `normalize = false` default: the evolving state must NOT be renormalized.
@testset "dynamical correlator vs free-particle analytics" begin
    L = 16
    m = 8                            # source site (chain centre)
    ns = 5:11                        # measured sink sites
    dt = 0.05
    ts = 0.0:dt:1.5

    H = force_planar(XY_model(ComplexF64, Trivial; g = 0, L = L))

    # fully polarized product state |↑…↑⟩ (basis index 1 on every site): a zero-energy eigenstate
    Vp = ℙ^2
    Vl = oneunit(Vp)
    ψvac = FiniteMPS([isometry(ComplexF64, Vl ⊗ Vp, Vl) for _ in 1:L])
    @test norm(H * ψvac) < 1.0e-10   # H annihilates the polarized vacuum ⇒ E₀ = 0, no bra phase

    # apply S^x at a single site (S^x = ½σ^x flips one spin ⇒ a single free particle)
    Sx = force_planar(S_x())
    apply_Sx(i) = FiniteMPOHamiltonian(physicalspace(ψvac), i => Sx) * ψvac
    ϕ_bra = Dict(n => apply_Sx(n) for n in ns)   # static ½|n⟩
    ϕ = apply_Sx(m)                               # ½|m⟩, evolved below

    # analytic single-particle propagator exp(-i h t) on the open chain
    ks = 1:L
    ε = [-2 * cos(k * π / (L + 1)) for k in ks]
    φ = [sqrt(2 / (L + 1)) * sin(n * k * π / (L + 1)) for n in 1:L, k in ks]
    Uref(t) = φ * Diagonal(cis.(-ε .* t)) * transpose(φ)
    Cref(n, t) = 0.25 * Uref(t)[n, m]

    # single-particle physics stays rank ≤ 2, so TDVP2 with a generous cap is essentially exact
    alg = TDVP2(; trscheme = truncrank(6))
    t_prev = 0.0
    maxerr = 0.0
    for t in ts
        t > 0 && ((ϕ, _) = timestep(ϕ, H, t_prev, t - t_prev, alg); t_prev = t)
        for n in ns
            maxerr = max(maxerr, abs(dot(ϕ_bra[n], ϕ) - Cref(n, t)))
        end
    end
    @test maxerr < 1.0e-3

    # the correlator retains the un-normalized amplitude: real-time evolution preserves the norm
    @test norm(ϕ) ≈ norm(ϕ_bra[m]) atol = 1.0e-6
end
