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
using LinearAlgebra: dot, norm
using Random

verbosity_full = 5
verbosity_conv = 1

# name a time-evolution algorithm for @testset labels ("TDVP", "TDVP2", "BUG")
algname(alg) = string(nameof(typeof(alg)))

maxbond(ψ) = maximum(i -> dim(left_virtualspace(ψ, i)), 1:length(ψ))

@testset "timestep" verbose = true begin
    dt = 0.1
    # every rank-adaptive algorithm gets the same cap: without one, BUG's augmentation doubles the
    # bond dimension on every half-sweep with nothing ever cutting it back
    algs = [TDVP(), TDVP2(; trunc = truncrank(10)), BUG(; trunc = truncrank(10))]
    L = 10

    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    ψ = FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4)
    E = expectation_value(ψ, H)
    ψ₀, = find_groundstate(ψ, H)
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(algname(alg))" for alg in algs
        ψ1, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4

        # the integrator carries no scratch space of its own - the sweep's allocator is obtained
        # per `timestep` - so re-using one across steps has to reproduce the answer
        ψ2, = timestep(ψ₀, H, 0.0, dt, alg)
        @test dot(ψ1, ψ2) ≈ norm(ψ1) * norm(ψ2) atol = 1.0e-10
    end

    Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])

    @testset "Finite LazySum $(algname(alg))" for alg in algs
        ψ, envs = timestep(ψ₀, Hlazy, 0.0, dt, alg)
        E = expectation_value(ψ, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1.0e-2
    end

    Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)

    @testset "Finite TimeDependent LazySum $(algname(alg))" for alg in algs
        ψ, envs = timestep(ψ₀, Ht(1.0), 0.0, dt, alg)
        E = expectation_value(ψ, Ht(1.0), envs)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, alg)
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test E ≈ Et atol = 1.0e-8
    end

    Ht2 = MultipliedOperator(H, t -> t < 0 ? error("t < 0!") : 4) +
        MultipliedOperator(H, 1.45)
    @testset "Finite TimeDependent LazySum (fix negative t issue) $(algname(alg))" for alg in algs
        ψ, envs = timestep(ψ₀, Ht2, 0.0, dt, alg)
        E = expectation_value(ψ, Ht2(0.0), envs)

        ψt, envst = timestep(ψ₀, Ht2, 0.0, dt, alg)
        Et = expectation_value(ψt, Ht2(0.0), envst)
        @test E ≈ Et atol = 1.0e-8
    end

    H = repeat(force_planar(heisenberg_XXX(; spin = 1)), 2)
    ψ₀ = InfiniteMPS([ℙ^3, ℙ^3], [ℙ^50, ℙ^50])
    E₀ = expectation_value(ψ₀, H)

    # the AC and C sweeps of the infinite integrator spawn over the unit cell, and additionally run
    # concurrently with each other, so they share one allocator: check both schedulers agree
    @testset "Infinite TDVP ($schedname)" for (schedname, scheduler) in SCHEDULERS
        ψ, envs = with_scheduler(scheduler) do
            return timestep(ψ₀, H, 0.0, dt, TDVP())
        end
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

# BUG-specific: unlike `TDVP` there is no backward-in-time substep, which makes it a natural
# imaginary-time integrator (cf. its docstring). `TDVP`'s imaginary-time behaviour is covered by the
# CBE block below.
@testset "Finite imaginary-time BUG" begin
    L = 10
    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))

    alg = BUG(; trunc = truncrank(4))

    Random.seed!(5)
    ψi = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
    E_start = real(expectation_value(ψi, H))
    E_prev = E_start
    for _ in 1:8
        ψi, = timestep(ψi, H, 0.0, 0.1, alg; imaginary_evolution = true, normalize = true)
        E_now = real(expectation_value(ψi, H))
        @test E_now ≤ E_prev + 1.0e-6   # monotone (non-increasing) energy
        @test maxbond(ψi) ≤ 8           # the cap bounds the bond at `2D` throughout
        E_prev = E_now
    end
    @test E_prev < E_start - 0.5        # substantial lowering toward the ground state
    @test norm(ψi) ≈ 1 atol = 1.0e-6    # `normalize = true` renormalizes each step
end

# The truncating path, which is BUG's distinguishing feature. `BUG(; trunc = …)` cuts the bond
# *ahead of* every local update — the bond carrying the previous half-sweep's augmentation — and
# augments without truncating, so the bond dimension oscillates between the requested rank and (at
# most) twice it.
@testset "BUG with truncation" begin
    L = 8
    H = force_planar(transverse_field_ising(ComplexF64, Trivial; L, g = 1.5))
    δt, nsteps, D = 0.05, 20, 4

    Random.seed!(11)
    ψ₀ = FiniteMPS(rand, ComplexF64, L, ℙ^2, ℙ^2)   # low rank to start with, entanglement grows
    normalize!(ψ₀)

    tovec(ψ) = (v = convert(TensorMap, ψ); v / norm(v))
    ref = exp(-im * convert(TensorMap, H) * (nsteps * δt)) * tovec(ψ₀)

    # the cap actually bites here: the exact state needs bond 16
    ψ = ψ₀
    for k in 0:(nsteps - 1)
        ψ, = timestep(ψ, H, k * δt, δt, BUG(; trunc = truncrank(D)))
        @test maxbond(ψ) ≤ 2D   # `D` right after the cut, at most `2D` after the augment
    end
    @test 1 - abs(dot(tovec(ψ), ref)) < 1.0e-3
end

# Genuine symmetric-tensor coverage (no `force_planar`) for the single-site finite integrators.
# BUG's augment step stacks bases per sector, so it can add or drop sectors: both `TDVP` and `BUG`
# must conserve the energy, accrue only an eigenstate phase, and preserve the total boundary charge
# (the fixed `right` virtual space at site L). TDVP2 is excluded here: it requires a `trunc` and is
# the two-site variant; these are single-site conservation properties.
@testset "Finite symmetric-tensor time evolution" begin
    dt = 0.1
    L = 6

    Random.seed!(2718)
    H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
    maxV = MPSKit.max_virtualspaces(physicalspace(H))
    ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
    ψ₀, = find_groundstate(ψ, H; verbosity = 0)
    E₀ = expectation_value(ψ₀, H)

    # the exact state has Schmidt rank ≤ 8 here, so this cap bounds the bond without discarding
    # anything — it only stops BUG's augmentation from doubling unchecked
    algs = [TDVP(), BUG(; trunc = truncrank(8))]

    @testset "U(1) Heisenberg ($(algname(alg)))" for alg in algs
        ψ1, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E1 = expectation_value(ψ1, H, envs)

        @test E₀ ≈ E1 atol = 1.0e-2
        @test imag(E1) ≈ 0 atol = 1.0e-8
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test right_virtualspace(ψ1, L) == right_virtualspace(ψ₀, L)
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
        alg = TDVP(; alg_expand = Exp(; trunc = truncrank(Dstart), kw...), trunc = truncrank(Dcap))
        E₀ = real(expectation_value(ψ₀, H))

        ref, cbe, plain = ψ₀, ψ₀, ψ₀
        for _ in 1:6
            ref, = timestep(ref, H, 0.0, dt, TDVP2(; trunc = truncrank(Dcap)))
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
        lossy = TDVP(; alg_expand = OptimalExpand(; trunc = truncrank(2)), trunc = truncrank(2))

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
        alg = TDVP(; alg_expand = OptimalExpand(; trunc = truncrank(Dstart)), trunc = truncrank(Dcap))
        E₀ = real(expectation_value(ψ₀, H))
        ψ = ψ₀
        for _ in 1:8
            ψ, = timestep(ψ, H, 0.0, 0.1, alg; imaginary_evolution = true, normalize = true)  # gauge renormalizes
        end
        @test real(expectation_value(ψ, H)) < E₀
        @test dim(left_virtualspace(ψ, L ÷ 2)) > Dstart
    end
end

@testset "Truncation error" verbose = true begin
    L = 10
    dt = 0.1
    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    Random.seed!(7)
    ψ₀ = normalize!(complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^16)))

    # fixed bond dimension sweep never discards anything, so the reported error is exactly zero
    @testset "no truncation" begin
        for alg in (TDVP(), BUG())
            info = last(timestep(ψ₀, H, 0.0, dt, alg))
            @test info isa MPSKit.AlgorithmInfo
            @test info.ϵ_max == 0
            @test info.ϵ_total == 0
            @test info.numtrunc == 0
            @test info.numsteps == 1
        end
    end

    @testset "$(nameof(alg))" for alg in (TDVP2, BUG)
        _, _, loose = timestep(ψ₀, H, 0.0, dt, alg(; trunc = truncrank(32)))
        ψ, _, tight = timestep(ψ₀, H, 0.0, dt, alg(; trunc = truncrank(2)))

        @test tight.ϵ_total > 1.0e-3 # throw away real weight
        @test tight.ϵ_max > 1.0e-4
        @test loose.ϵ_max < tight.ϵ_max # throw away less weight with a more forgiving truncation
        @test tight.numtrunc > 0
        @test tight.ϵ_total >= tight.ϵ_max
        @test tight.ϵ_total <= sqrt(tight.numtrunc) * tight.ϵ_max
        # `ϵ_total` = norm loss in real time with `normalize = false`
        @test norm(ψ)^2 ≈ norm(ψ₀)^2 - tight.ϵ_total^2 atol = 1.0e-12
    end

    @testset "aggregation over an evolution" begin
        alg = TDVP2(; trunc = truncrank(2))
        nsteps = 4
        _, _, step = timestep(ψ₀, H, 0.0, dt, alg)
        ψ, _, total = time_evolve(ψ₀, H, 0:dt:(nsteps * dt), alg)

        @test total.numsteps == nsteps
        @test total.numtrunc >= step.numtrunc
        @test total.ϵ_total >= step.ϵ_total
        @test norm(ψ)^2 ≈ norm(ψ₀)^2 - total.ϵ_total^2 atol = 1.0e-12
        @test total.ϵ_max >= step.ϵ_max
        @test total.ϵ_max <= total.ϵ_total
    end
end

@testset "time_evolve" verbose = true begin
    t_span = 0:0.1:0.1
    algs = [TDVP(), TDVP2(; trunc = truncrank(10)), BUG(; trunc = truncrank(10))]

    L = 10
    H = force_planar(heisenberg_XXX(; spin = 1 // 2, L))
    ψ₀ = FiniteMPS(L, ℙ^2, ℙ^1)
    E₀ = expectation_value(ψ₀, H)

    @testset "Finite $(algname(alg))" for alg in algs
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
