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
