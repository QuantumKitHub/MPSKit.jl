println("
------------------------------------
|    Allocators in algorithms      |
------------------------------------
")

using .TestSetup
using Test, TestExtras
using MPSKit
using TensorKit
using TensorKit: ℙ
using MPSKit: SerialScheduler, DynamicScheduler
using Random

verbosity = 0

# The scheduler is process-global: always restore it.
function with_scheduler(f, scheduler)
    old_sched = MPSKit.Defaults.scheduler[]
    try
        MPSKit.Defaults.scheduler[] = scheduler
        return f()
    finally
        MPSKit.Defaults.scheduler[] = old_sched
    end
end

# The scheduler decides which allocator the local updates get - a buffer when a single task owns it,
# a shared manual allocator otherwise - so it must not change any number.
const SCHEDULERS = ("serial" => SerialScheduler(), "dynamic" => DynamicScheduler())

@testset "finite algorithms" verbose = true begin
    L = 8
    H = force_planar(transverse_field_ising(; g = 4.0, L))
    Random.seed!(0x1234)
    ψ₀ = FiniteMPS(randn, ComplexF64, L, ℙ^2, ℙ^8)

    @testset "DMRG" begin
        alg = DMRG(; verbosity, maxiter = 10)
        ψ, _, _ = find_groundstate(ψ₀, H, alg)
        E = expectation_value(ψ, H)

        # the algorithm object holds no scratch space, so a second solve reproduces the first
        ψ2, _, _ = find_groundstate(ψ₀, H, alg)
        @test expectation_value(ψ2, H) ≈ E atol = 1.0e-8
    end

    @testset "DMRG2" begin
        reference = expectation_value(first(find_groundstate(ψ₀, H, DMRG(; verbosity, maxiter = 10))), H)
        alg = DMRG2(; verbosity, maxiter = 10, trunc = truncrank(8))
        ψ, _, _ = find_groundstate(ψ₀, H, alg)
        @test expectation_value(ψ, H) ≈ reference atol = 1.0e-6
    end

    @testset "TDVP" begin
        ψ, _ = timestep(complex(ψ₀), H, 0.0, 0.01, TDVP())
        @test norm(ψ) ≈ norm(ψ₀) atol = 1.0e-6
    end

    @testset "BUG" begin
        # the augmentation step doubles the bond, so compare against the reference at a bond the
        # buffer-served eigensolves and the plain ones both reach
        ψ, _ = timestep(complex(ψ₀), H, 0.0, 0.01, BUG())
        @test norm(ψ) ≈ norm(ψ₀) atol = 1.0e-6

        ψ_tdvp, _ = timestep(complex(ψ₀), H, 0.0, 0.01, TDVP())
        @test abs(dot(ψ, ψ_tdvp)) / (norm(ψ) * norm(ψ_tdvp)) ≈ 1.0 atol = 1.0e-4
    end

    @testset "nested algorithms" begin
        # `alg_expand` is driven through `changebond!`, which is not an entry point and therefore
        # selects an allocator of its own while the outer sweep still holds one
        alg = DMRG(;
            verbosity, maxiter = 3, alg_expand = SketchedExpand(; trunc = truncrank(2)),
            trunc = truncrank(8),
        )
        ψ, _, _ = find_groundstate(ψ₀, H, alg)
        @test expectation_value(ψ, H) isa Number
    end

    @testset "time_evolve reuses nothing across steps" begin
        nsteps = Ref(0)
        finalize = function (t, ψ, H, envs)
            nsteps[] += 1
            return ψ, envs
        end
        ψ, _ = time_evolve(complex(ψ₀), H, 0.0:0.01:0.03, TDVP(; finalize))
        @test nsteps[] == 3
        @test norm(ψ) ≈ norm(ψ₀) atol = 1.0e-6
    end
end

@testset "infinite algorithms" verbose = true begin
    H = force_planar(repeat(transverse_field_ising(; g = 4.0), 2))
    Random.seed!(0x1234)
    ψ₀ = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^8, ℙ^8])

    reference = nothing
    for (schedname, scheduler) in SCHEDULERS
        @testset "VUMPS ($schedname)" begin
            alg = VUMPS(; verbosity, maxiter = 50)
            ψ, _, _ = with_scheduler(scheduler) do
                find_groundstate(ψ₀, H, alg)
            end
            E = sum(expectation_value(ψ, H))
            isnothing(reference) ? (reference = E) : (@test E ≈ reference atol = 1.0e-6)
        end

        @testset "GradientGrassmann ($schedname)" begin
            alg = GradientGrassmann(; verbosity, maxiter = 50)
            ψ, _, _ = with_scheduler(scheduler) do
                find_groundstate(ψ₀, H, alg)
            end
            @test sum(expectation_value(ψ, H)) ≈ reference atol = 1.0e-4
        end

        @testset "TDVP ($schedname)" begin
            ψ, _ = with_scheduler(scheduler) do
                timestep(ψ₀, H, 0.0, 0.01, TDVP())
            end
            @test sum(expectation_value(ψ, H)) isa Number
        end

        @testset "VOMPS ($schedname)" begin
            # leading_boundary and approximate take separate code paths, and the latter is the one
            # that runs the AC and C projections of a site concurrently with each other
            β = 0.5
            O = classical_ising(; β)
            ψ, envs = with_scheduler(scheduler) do
                leading_boundary(InfiniteMPS(ℂ^2, ℂ^10), O, VOMPS(; verbosity, tol = 1.0e-8))
            end
            @test -log(expectation_value(ψ, O, envs)) / β ≈ -2.0515856253898357 atol = 1.0e-8

            dt = 1.0e-3
            W = make_time_mpo(H, dt, WII())
            ϕ = InfiniteMPS([ℙ^2, ℙ^2], [ℙ^10, ℙ^10])
            ϕ′, _ = with_scheduler(scheduler) do
                approximate(InfiniteMPS([ℙ^2, ℙ^2], [ℙ^12, ℙ^12]), (W, ϕ), VOMPS(; verbosity))
            end
            ϕ_ref, _ = timestep(ϕ, H, 0.0, dt, TDVP())
            @test abs(dot(ϕ′, ϕ_ref)) ≈ 1.0 atol = dt
        end
    end

    @testset "IDMRG" begin
        alg = IDMRG(; verbosity, maxiter = 100)
        ψ, _, _ = find_groundstate(ψ₀, H, alg)
        @test sum(expectation_value(ψ, H)) ≈ reference atol = 1.0e-4
    end

    @testset "changebonds" begin
        alg = OptimalExpand(; trunc = truncrank(4))
        ψ, _ = changebonds(ψ₀, H, alg)
        @test dim(left_virtualspace(ψ, 1)) > dim(left_virtualspace(ψ₀, 1))
    end
end
