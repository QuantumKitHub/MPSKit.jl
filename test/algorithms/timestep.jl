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

@testset "timestep" verbose = true begin
    dt = 0.1
    algs = [TDVP(), TDVP2(; trunc = truncrank(10)), BUG()]
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

    # imaginary-time evolution lowers the energy toward the ground state. This is generic to
    # symmetric integrators; BUG in particular has no backward substep (cf. its docstring), so it is
    # a natural imaginary-time integrator.
    @testset "Finite imaginary-time lowers energy $(algname(alg))" for alg in [TDVP(), BUG()]
        Random.seed!(5)
        ψi = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:20
            ψi, = timestep(ψi, H, 0.0, 0.1, alg; imaginary_evolution = true, normalize = true)
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6   # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 1.0        # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6    # imaginary-time renormalizes each step
    end

    # second-order (temporal) convergence on a small full-rank system, which isolates the temporal
    # order. This is a BUG-only test: at full rank single-site TDVP / two-site TDVP2 integrate the
    # *exact* dynamics (the tangent-space projector is the identity, cf. Lubich/Haegeman), so their
    # overlap error sits at the numerical floor and a convergence slope is undefined. BUG's
    # augment-and-Galerkin step has a genuine temporal error, so it shows a clean ≥ 2nd-order slope
    # (often ≈ 4 on full-rank systems); we assert the order floor.
    @testset "second-order convergence (BUG)" begin
        Random.seed!(2)
        Lc = 4
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψ_full = FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^4)   # full-rank: 1,2,4,2,1

        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψ_full)
        ψvec /= norm(ψvec)

        T = 0.5
        dts = [0.1, 0.05, 0.025]
        errs = map(dts) do δt
            n = round(Int, T / δt)
            ref = exp(-im * Hmat * (n * δt)) * ψvec
            ψ = copy(ψ_full)
            envs = environments(ψ, Hc, ψ)
            for k in 0:(n - 1)
                timestep!(ψ, Hc, k * δt, δt, BUG(), envs)
            end
            ψout = convert(TensorMap, ψ)
            ψout /= norm(ψout)
            return 1 - abs(dot(ψout, ref))
        end

        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "BUG convergence" errs slopes
        for s in slopes
            @test s ≥ 1.7
        end
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

# BUG is a drop-in for TDVP on the finite interface: over a few real-time steps of a generic
# (random) state the two integrators must agree, up to their shared temporal-discretization error.
# This is a cross-check between integrators, so it has no meaning as a TDVP-vs-TDVP test.
@testset "BUG agrees with TDVP" begin
    L = 10
    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    Random.seed!(1234)
    ψr = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
    δt = 0.01
    ψ_bug, ψ_tdvp = ψr, ψr
    for k in 0:4
        ψ_bug, = timestep(ψ_bug, H, k * δt, δt, BUG())
        ψ_tdvp, = timestep(ψ_tdvp, H, k * δt, δt, TDVP())
    end
    @test expectation_value(ψ_bug, H) ≈ expectation_value(ψ_tdvp, H) atol = 1.0e-3
    @test abs(dot(ψ_bug, ψ_tdvp)) ≈ 1 atol = 1.0e-3
end

# Genuine symmetric-tensor coverage (no `force_planar`) for the single-site finite integrators.
# Both TDVP and BUG must conserve the energy and accrue only an eigenstate phase, and must preserve
# the total boundary charge (the fixed `right` virtual space at site L). TDVP2 is excluded here: it
# requires a `trscheme` and is the two-site variant; these are single-site conservation properties.
@testset "Finite symmetric-tensor time evolution" verbose = true begin
    dt = 0.1
    L = 6
    algs = [TDVP(), BUG()]

    # U(1)-symmetric Heisenberg, both in the natural total-Sz = 0 sector and in a fixed nonzero
    # total-charge (Sz = 1) sector.
    @testset "U(1) Heisenberg (total Sz = $label, $(algname(alg)))" for (label, right) in
            (("0", U1Space(0 => 1)), ("1", U1Space(1 => 1))), alg in algs
        Random.seed!(2718)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)]; right)
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E1 = expectation_value(ψ1, H, envs)

        @test E₀ ≈ E1 atol = 1.0e-2
        @test imag(E1) ≈ 0 atol = 1.0e-8
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test right_virtualspace(ψ1, L) == right_virtualspace(ψ₀, L)
    end

    # A second symmetry group. Z2 (transverse-field Ising) and SU2 (Heisenberg); same assertions.
    @testset "Z2 transverse-field Ising ($(algname(alg)))" for alg in algs
        Random.seed!(161803)
        H = transverse_field_ising(ComplexF64, Z2Irrep; g = 1.0, L)
        ψ = FiniteMPS(physicalspace(H), Z2Space(0 => 4, 1 => 4))
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test right_virtualspace(ψ1, L) == right_virtualspace(ψ₀, L)
    end

    @testset "SU(2) Heisenberg ($(algname(alg)))" for alg in algs
        Random.seed!(577215)
        H = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L)
        # SU(2) spin-1/2 bonds alternate between integer / half-integer spins, so use the
        # model's own full-rank virtual spaces rather than a hand-picked (integer-only) space.
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, alg)
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test right_virtualspace(ψ1, L) == right_virtualspace(ψ₀, L)
    end

    # Imaginary-time symmetric evolution lowers the energy while preserving the sector + norm.
    @testset "imaginary-time (U(1), $(algname(alg)))" for alg in algs
        Random.seed!(141421)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψi = FiniteMPS(physicalspace(H), maxV[2:(end - 1)])
        Rtot = right_virtualspace(ψi, L)   # total boundary charge, fixed throughout
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:15
            ψi, = timestep(ψi, H, 0.0, 0.1, alg; imaginary_evolution = true, normalize = true)
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6            # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 0.5                 # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6             # imaginary-time renormalizes each step
        @test right_virtualspace(ψi, L) == Rtot      # total boundary charge conserved throughout
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

@testset "time_evolve" verbose = true begin
    t_span = 0:0.1:0.1
    algs = [TDVP(), TDVP2(; trunc = truncrank(10)), BUG()]

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
