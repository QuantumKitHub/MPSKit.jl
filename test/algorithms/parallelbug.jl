println("
-------------------------------------
|   ParallelBUG time-stepping tests |
-------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using LinearAlgebra: dot, norm
using Random

# NOTE (experimental). `ParallelBUG` is the parallel Basis-Update & Galerkin integrator
# (Ceruti et al. 2024) specialized to the caterpillar `FiniteMPS`. It reproduces the exact matrix
# parallel-BUG step for two sites, conserves energy / the eigenstate phase exactly (amplitude carried
# once, at the root), grows bonds adaptively, agrees with `TDVP` over short times, and converges at
# (at least) the documented first order in `dt` toward the dense reference.
@testset "ParallelBUG time evolution" verbose = true begin
    dt = 0.1
    L = 6

    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    ψ = FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4)
    ψ₀, = find_groundstate(ψ, H; verbosity = 0)
    E₀ = expectation_value(ψ₀, H)

    # 1. energy conservation + eigenstate phase (amplitude is carried exactly once, at the root)
    @testset "energy conservation + eigenstate phase" begin
        ψ1, envs = timestep(ψ₀, H, 0.0, dt, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12)))
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
    end

    # 2. two sites: exact reproduction of the dense exp(-iH·dt) step (locks the block conventions)
    @testset "two-site exactness vs dense reference" begin
        Random.seed!(2)
        Lc = 2
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψc = FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^4)
        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψc); ψvec /= norm(ψvec)
        ref = exp(-im * Hmat * 0.05) * ψvec
        ψ1, = timestep(ψc, Hc, 0.0, 0.05, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12)))
        out = convert(TensorMap, ψ1); out /= norm(out)
        @test 1 - abs(dot(out, ref)) < 1.0e-10
    end

    # 3. agreement with TDVP over a few short real-time steps of a random MPS
    @testset "agreement with TDVP" begin
        Random.seed!(1234)
        ψr = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        δt = 0.01
        ψ_p, ψ_tdvp = ψr, ψr
        for k in 0:4
            ψ_p, = timestep(ψ_p, H, k * δt, δt, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12)))
            ψ_tdvp, = timestep(ψ_tdvp, H, k * δt, δt, TDVP())
        end
        @test expectation_value(ψ_p, H) ≈ expectation_value(ψ_tdvp, H) atol = 1.0e-3
        @test abs(dot(ψ_p, ψ_tdvp)) ≈ 1 atol = 1.0e-3
    end

    # 4. imaginary-time evolution lowers the energy monotonically and stays normalized
    @testset "imaginary-time lowers energy" begin
        Random.seed!(5)
        ψi = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:20
            ψi, = timestep(
                ψi, H, 0.0, 0.1, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-10));
                imaginary_evolution = true, normalize = true
            )
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6      # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 1.0           # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6       # imaginary-time renormalizes each step
    end

    # 5. rank adaptivity: a low-bond-dim start grows under a tight tolerance and stays small under a
    #    loose one (the augment-then-SvdCut mechanism injects the new directions the evolution finds).
    @testset "bond growth" begin
        Random.seed!(3)
        Lg = 6
        Hg = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lg))
        ψg = normalize!(complex(FiniteMPS(rand, Float64, Lg, ℙ^2, ℙ^2)))   # bond dim 2
        Dstart = maximum(dim(right_virtualspace(ψg, b)) for b in 1:(Lg - 1))

        ψtight = ψg
        for k in 0:4
            ψtight, = timestep(ψtight, Hg, k * 0.05, 0.05, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-8)))
        end
        Dtight = maximum(dim(right_virtualspace(ψtight, b)) for b in 1:(Lg - 1))

        ψloose = ψg
        for k in 0:4
            ψloose, = timestep(ψloose, Hg, k * 0.05, 0.05, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-2)))
        end
        Dloose = maximum(dim(right_virtualspace(ψloose, b)) for b in 1:(Lg - 1))

        @info "ParallelBUG bond growth" Dstart Dloose Dtight
        @test Dtight > Dstart          # rank-adaptivity grows the bond
        @test Dloose < Dtight          # a looser tolerance keeps a smaller bond
    end

    # 6. LazySum smoke test
    @testset "LazySum" begin
        Hlazy = LazySum([3 * H, 1.55 * H, -0.1 * H])
        ψl, envs = timestep(ψ₀, Hlazy, 0.0, dt, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12)))
        E = expectation_value(ψl, Hlazy, envs)
        @test (3 + 1.55 - 0.1) * E₀ ≈ E atol = 1.0e-2
    end

    # 6b. TimeDependent operator smoke test: exercises the `TimedOperator` coupling path (which
    #     has no one-arg apply). A constant-in-`t` coefficient makes midpoint-freezing exact, so
    #     evolving the time-dependent `Ht` at `t=1.0` must match evolving the pre-evaluated `Ht(1.0)`.
    @testset "TimeDependent LazySum" begin
        Ht = MultipliedOperator(H, t -> 4) + MultipliedOperator(H, 1.45)
        alg = ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12))
        ψa, envsa = timestep(ψ₀, Ht(1.0), 0.0, dt, alg)
        Ea = expectation_value(ψa, Ht(1.0), envsa)

        ψt, envst = timestep(ψ₀, Ht, 1.0, dt, alg)
        Et = expectation_value(ψt, Ht(1.0), envst)
        @test Ea ≈ Et atol = 1.0e-8
    end

    # 7. convergence order: the integrator is documented as (globally) first order in dt; assert at
    #    least that. (At these bond dimensions the augmented spans are near-exact and the measured
    #    slope is ≈ 2, so only a lower bound is imposed.)
    @testset "first-order accuracy" begin
        Random.seed!(2)
        Lc = 4
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψf = normalize!(FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^4))
        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψf); ψvec /= norm(ψvec)

        Tfin = 0.2
        dts = [0.05, 0.025, 0.0125]
        errs = map(dts) do δt
            n = round(Int, Tfin / δt)
            ref = exp(-im * Hmat * (n * δt)) * ψvec
            ψc = copy(ψf)
            for k in 0:(n - 1)
                ψc, = timestep(ψc, Hc, k * δt, δt, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12)))
            end
            out = convert(TensorMap, ψc); out /= norm(out)
            return 1 - abs(dot(out, ref))
        end
        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "ParallelBUG convergence" errs slopes
        # documented rate: first order (slope ≈ 1); assert at least that
        @test all(s -> s > 0.8, slopes)
    end

    # 7b. THE decisive local-order gate (Ceruti et al. 2024; MPSKit design §8): the single-step local
    #     error ‖step(dt) − exp(−iH·dt)ψ₀‖ against the dense propagator, log–log slope vs dt, at
    #     L = 3 and L = 4 (2-site is exact and does NOT exercise the dropped "new–new" corner, so
    #     L ≥ 3 is required). A first-order method has local error O(dt²) ⇒ slope ≈ 2; this is the gate
    #     that separates genuine second order (slope ≈ 3, see `ParallelBUG2`) from a better-constant
    #     first-order scheme. Full bonds (ℙ^8) isolate the time-discretization error.
    @testset "local order slope L=$Lc" for Lc in (3, 4)
        Random.seed!(2)
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψf = normalize!(FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^8))
        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψf); ψvec /= norm(ψvec)
        dts = [0.04, 0.02, 0.01, 0.005]
        errs = map(dts) do δt
            ref = exp(-im * Hmat * δt) * ψvec
            ϕ, = timestep(ψf, Hc, 0.0, δt, ParallelBUG(; trscheme = truncerror(; atol = 1.0e-12)))
            out = convert(TensorMap, ϕ)
            return norm(out - ref)
        end
        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "ParallelBUG local order" Lc errs slopes
        for s in slopes
            @test s ≈ 2 atol = 0.3
        end
    end

    # 8. the truncation tolerance ϑ maps onto the accumulating `c·n·ϑ` error term: tightening ϑ
    #    improves the overlap with a ϑ → 0 run of the same integrator (which shares the time
    #    discretization error, so the comparison isolates the truncation term).
    @testset "accuracy improves as ϑ decreases" begin
        Random.seed!(202)
        Lc = 6
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψ0 = normalize!(complex(FiniteMPS(rand, Float64, Lc, ℙ^2, ℙ^2)))
        Tfin = 0.2; δt = 0.05; n = round(Int, Tfin / δt)
        evolve(ϑ) = foldl(0:(n - 1); init = ψ0) do ψc, k
            first(timestep(ψc, Hc, k * δt, δt, ParallelBUG(; trscheme = truncerror(; atol = ϑ))))
        end
        ref = evolve(1.0e-12)
        ϑs = [1.0e-2, 1.0e-4, 1.0e-6]
        errs = map(ϑs) do ϑ
            ψc = evolve(ϑ)
            return 1 - abs(dot(ψc, ref)) / (norm(ψc) * norm(ref))
        end
        @info "ParallelBUG accuracy vs ϑ" ϑs errs
        @test issorted(errs; rev = true)
        @test errs[end] < errs[1] / 10
    end

    # 9. step rejection: a small-bond start under a tight tolerance + a large `dt` can saturate the
    #    doubling cap; `maxiter_rejection > 0` then recomputes the step as half-steps. The recompute
    #    path must run cleanly, keep a normalized state, and never worsen the overlap with the dense
    #    reference (sub-stepping only refines it).
    @testset "step rejection" begin
        Random.seed!(7)
        Lr = 4
        Hr = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lr))
        Hmat = convert(TensorMap, Hr)
        ψr = normalize!(complex(FiniteMPS(rand, Float64, Lr, ℙ^2, ℙ^2)))   # bond dim 2 (small)
        ψvec = convert(TensorMap, ψr); ψvec /= norm(ψvec)
        dtbig = 0.3
        ref = exp(-im * Hmat * dtbig) * ψvec

        tol = truncerror(; atol = 1.0e-6)
        ψno, = timestep(ψr, Hr, 0.0, dtbig, ParallelBUG(; trscheme = tol, maxiter_rejection = 0))
        ψrej, = timestep(ψr, Hr, 0.0, dtbig, ParallelBUG(; trscheme = tol, maxiter_rejection = 4))
        outno = convert(TensorMap, ψno); outno /= norm(outno)
        outrej = convert(TensorMap, ψrej); outrej /= norm(outrej)
        errno = 1 - abs(dot(outno, ref))
        errrej = 1 - abs(dot(outrej, ref))
        @info "ParallelBUG step rejection" errno errrej
        @test errrej ≤ errno + 1.0e-12     # sub-stepping never worsens the accuracy
        @test isfinite(errrej)
    end

    # 10. first-order error accumulation (Ceruti et al. 2024, Thm 4.5: `‖error‖ ≲ c·n·ϑ`). At fixed
    #     `dt` and `ϑ` the global error grows ~linearly in the number of steps `n`. (A `dt`-refinement
    #     slope is ill-posed here: at fixed `ϑ`, halving `dt` doubles `n` and thus the accumulated
    #     truncation error, so refining `dt` need not converge — hence this per-`n` test instead.)
    @testset "linear error accumulation" begin
        Random.seed!(13)
        Le = 6
        He = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Le))
        Hmat = convert(TensorMap, He)
        ψe = normalize!(complex(FiniteMPS(rand, Float64, Le, ℙ^2, ℙ^2)))
        ψvec = convert(TensorMap, ψe); ψvec /= norm(ψvec)
        δt = 0.02
        alg = ParallelBUG(; trscheme = truncerror(; atol = 1.0e-4))
        ns = [5, 10, 20]
        # state 2-norm error (phase-aligned), the quantity the `c·n·ϑ` bound controls — the overlap
        # *infidelity* `1-|⟨·⟩|` is its square, so it would grow ~n² and must not be used here.
        errs = map(ns) do n
            ref = exp(-im * Hmat * (n * δt)) * ψvec
            ψc = copy(ψe)
            for k in 0:(n - 1)
                ψc, = timestep(ψc, He, k * δt, δt, alg)
            end
            out = convert(TensorMap, ψc); out /= norm(out)
            return sqrt(2 * (1 - abs(dot(out, ref))))
        end
        @info "ParallelBUG error accumulation" ns errs
        @test issorted(errs)                        # error grows monotonically with the step count
        @test 2.5 * errs[1] < errs[3] < 6 * errs[1]   # ~linear in n (5→20 = 4×), not saturating/quadratic
    end
end

# Charge-sector coverage: a fixed-rank symmetric step must preserve the total charge and the graded
# structure of every bond (energy conservation + eigenstate phase carry over from the trivial case).
@testset "ParallelBUG symmetric tensors" verbose = true begin
    dt = 0.1
    L = 6

    @testset "U(1) Heisenberg" begin
        Random.seed!(2718)
        H = heisenberg_XXX(ComplexF64, U1Irrep; spin = 1 // 2, L)
        maxV = MPSKit.max_virtualspaces(physicalspace(H))
        ψ = FiniteMPS(physicalspace(H), maxV[2:(end - 1)]; right = U1Space(0 => 1))
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)
        Rtot = right_virtualspace(ψ₀, L)
        Vl₀ = left_virtualspace.(Ref(ψ₀), 1:L)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, ParallelBUG())   # fixed-rank (notrunc)
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test imag(E1) ≈ 0 atol = 1.0e-8
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
        @test right_virtualspace(ψ1, L) == Rtot            # total boundary charge preserved
        @test left_virtualspace.(Ref(ψ1), 1:L) == Vl₀      # graded structure preserved (fixed rank)
    end

    @testset "Z2 transverse-field Ising" begin
        Random.seed!(161803)
        H = transverse_field_ising(ComplexF64, Z2Irrep; g = 1.0, L)
        ψ = FiniteMPS(physicalspace(H), Z2Space(0 => 4, 1 => 4))
        ψ₀, = find_groundstate(ψ, H; verbosity = 0)
        E₀ = expectation_value(ψ₀, H)

        ψ1, envs = timestep(ψ₀, H, 0.0, dt, ParallelBUG())
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
    end
end

# `ParallelBUG2` (second-order parallel BUG, Kusch 2024 Variant 2 / `4r`, EXPERIMENTAL). The genuine
# second-order assembly enriches every bond basis with one `H·ψ₀` application (`Û0`/`V̂0`), runs the
# `2r` Galerkin K-steps freezing the enriched right basis, and transports the EVOLVED-amplitude
# coupling to the root through the frozen right isometries, keeping the "new–new" corner zero (O(dt³)).
# It reproduces the exact two-site step, conserves energy / the eigenstate phase, agrees with `TDVP`,
# lowers the energy monotonically in imaginary time, preserves the charge sectors, and — the decisive
# gate — has a local error one order higher than the first-order `ParallelBUG`: slope ≥ 3 (vs ≈ 2).
@testset "ParallelBUG2 (second order)" verbose = true begin
    dt = 0.1
    L = 6
    H = force_planar(heisenberg_XXX(Float64, Trivial; spin = 1 // 2, L))
    ψ = FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4)
    ψ₀, = find_groundstate(ψ, H; verbosity = 0)
    E₀ = expectation_value(ψ₀, H)

    # 1. two sites: exact reproduction of the dense exp(-iH·dt) step
    @testset "two-site exactness vs dense reference" begin
        Random.seed!(2)
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = 2))
        ψc = FiniteMPS(rand, ComplexF64, 2, ℙ^2, ℙ^4)
        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψc); ψvec /= norm(ψvec)
        ref = exp(-im * Hmat * 0.05) * ψvec
        ψ1, = timestep(ψc, Hc, 0.0, 0.05, ParallelBUG2(; trscheme = truncerror(; atol = 1.0e-12)))
        out = convert(TensorMap, ψ1); out /= norm(out)
        @test 1 - abs(dot(out, ref)) < 1.0e-10
    end

    # 2. energy conservation + eigenstate phase (amplitude carried exactly once, at the root)
    @testset "energy conservation + eigenstate phase" begin
        ψ1, envs = timestep(ψ₀, H, 0.0, dt, ParallelBUG2(; trscheme = truncerror(; atol = 1.0e-12)))
        E1 = expectation_value(ψ1, H, envs)
        @test E₀ ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψ₀) ≈ exp(im * dt * E₀) atol = 1.0e-4
    end

    # 3. THE decisive local-order gate (mirror of the first-order testset above). The single-step
    #    local error ‖step(dt) − exp(−iH·dt)ψ₀‖ has slope ≈ 2 for first order and slope ≈ 3 for genuine
    #    second order (local `O(dt³)`). Full bonds (ℙ^8) isolate the time-discretization error.
    @testset "local order slope ≈ 3 (full, L=$Lc)" for Lc in (4, 5)
        Random.seed!(2)
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψf = normalize!(FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^8))
        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψf); ψvec /= norm(ψvec)
        dts = [0.04, 0.02, 0.01, 0.005]
        errs = map(dts) do δt
            ref = exp(-im * Hmat * δt) * ψvec
            ϕ, = timestep(ψf, Hc, 0.0, δt, ParallelBUG2(; trscheme = truncerror(; atol = 1.0e-12)))
            return norm(convert(TensorMap, ϕ) - ref)
        end
        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "ParallelBUG2 local order (full)" Lc errs slopes
        for s in slopes
            @test s ≈ 3 atol = 0.4      # genuine second order (vs the first-order value 2)
        end
    end

    # the second-order slope must persist as the chain grows past a single interior bond; L = 6 with
    # a reduced bond dimension ℙ^4 has an unsaturated interior bond (a genuine dropped corner).
    @testset "local order slope ≈ 3 (reduced bond, L=6)" begin
        Random.seed!(2)
        Lc = 6
        Hc = force_planar(transverse_field_ising(ComplexF64, Trivial; L = Lc))
        ψf = normalize!(FiniteMPS(rand, ComplexF64, Lc, ℙ^2, ℙ^4))
        Hmat = convert(TensorMap, Hc)
        ψvec = convert(TensorMap, ψf); ψvec /= norm(ψvec)
        dts = [0.04, 0.02, 0.01, 0.005]
        errs = map(dts) do δt
            ref = exp(-im * Hmat * δt) * ψvec
            ϕ, = timestep(ψf, Hc, 0.0, δt, ParallelBUG2(; trscheme = truncerror(; atol = 1.0e-12)))
            return norm(convert(TensorMap, ϕ) - ref)
        end
        slopes = [
            (log(errs[i + 1]) - log(errs[i])) / (log(dts[i + 1]) - log(dts[i]))
                for i in 1:(length(dts) - 1)
        ]
        @info "ParallelBUG2 local order (reduced)" errs slopes
        for s in slopes
            @test s ≈ 3 atol = 0.4
        end
    end

    # 4. agreement with TDVP over a few short real-time steps of a random MPS
    @testset "agreement with TDVP" begin
        Random.seed!(1234)
        ψr = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        δt = 0.01
        ψ_p, ψ_tdvp = ψr, ψr
        for k in 0:4
            ψ_p, = timestep(ψ_p, H, k * δt, δt, ParallelBUG2(; trscheme = truncerror(; atol = 1.0e-12)))
            ψ_tdvp, = timestep(ψ_tdvp, H, k * δt, δt, TDVP())
        end
        @test expectation_value(ψ_p, H) ≈ expectation_value(ψ_tdvp, H) atol = 1.0e-3
        @test abs(dot(ψ_p, ψ_tdvp)) ≈ 1 atol = 1.0e-3
    end

    # 5. imaginary-time evolution lowers the energy monotonically and stays normalized
    @testset "imaginary-time lowers energy" begin
        Random.seed!(5)
        ψi = complex(FiniteMPS(rand, Float64, L, ℙ^2, ℙ^4))
        E_start = real(expectation_value(ψi, H))
        E_prev = E_start
        for _ in 1:20
            ψi, = timestep(
                ψi, H, 0.0, 0.1, ParallelBUG2(; trscheme = truncerror(; atol = 1.0e-10));
                imaginary_evolution = true, normalize = true
            )
            E_now = real(expectation_value(ψi, H))
            @test E_now ≤ E_prev + 1.0e-6      # monotone (non-increasing) energy
            E_prev = E_now
        end
        @test E_prev < E_start - 1.0           # substantial lowering toward the ground state
        @test norm(ψi) ≈ 1 atol = 1.0e-6       # imaginary-time renormalizes each step
    end

    # 6. charge-sector coverage: a fixed-rank (notrunc) Z2 step preserves the total boundary charge
    @testset "Z2 symmetry preservation" begin
        Random.seed!(161803)
        Hz = transverse_field_ising(ComplexF64, Z2Irrep; g = 1.0, L)
        ψz = FiniteMPS(physicalspace(Hz), Z2Space(0 => 4, 1 => 4))
        ψz0, = find_groundstate(ψz, Hz; verbosity = 0)
        Ez = expectation_value(ψz0, Hz)
        Rtot = right_virtualspace(ψz0, L)

        ψ1, envs = timestep(ψz0, Hz, 0.0, dt, ParallelBUG2())   # fixed-rank (notrunc)
        E1 = expectation_value(ψ1, Hz, envs)
        @test Ez ≈ E1 atol = 1.0e-2
        @test dot(ψ1, ψz0) ≈ exp(im * dt * Ez) atol = 1.0e-4
        @test right_virtualspace(ψ1, L) == Rtot                # total boundary charge preserved
    end
end
