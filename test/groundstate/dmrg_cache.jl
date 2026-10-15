using .TestSetup
using Test, MPSKit, TensorKit, Random, Logging

sweep_allocator(ψ) = MPSKit.default_allocator(ψ, MPSKit.SerialScheduler())

# Run `sweeps` complete sweeps of local updates on the solve-owned cache, testing at every
# local window that the cached effective Hamiltonian acts like one built from the
# environments `reference(ψ)` of the current state.
function test_cached_windows(alg, ψ, H, envs = environments(ψ, H, ψ); reference = ψ -> environments(ψ, H, ψ), sweeps = 1)
    nsites = alg isa DMRG ? 1 : 2
    allocator = sweep_allocator(ψ)
    state = MPSKit.DMRGState(ψ, H, alg, envs, allocator, MPSKit.NoTimerOutput())
    cache = state.envs
    @test MPSKit.uses_sweep_cache(cache)
    for iter in 1:sweeps, (direction, sites) in zip((Val(:right), Val(:left)), MPSKit.sweep_ranges(alg, ψ))
        for pos in sites
            MPSKit.center_window!(cache, ψ, pos, pos + nsites - 1)
            fresh = reference(ψ)
            if nsites == 1
                x = ψ.AC[pos]
                @test MPSKit.AC_hamiltonian(pos, ψ, H, ψ, cache) * x ≈ MPSKit.AC_hamiltonian(pos, ψ, H, ψ, fresh) * x
            else
                x = MPSKit.AC2(ψ, pos)
                @test MPSKit.AC2_hamiltonian(pos, ψ, H, ψ, cache) * x ≈ MPSKit.AC2_hamiltonian(pos, ψ, H, ψ, fresh) * x
            end
            ψ, = MPSKit.local_update!(
                pos, direction, ψ, H, alg, cache, 1.0, 0.0, 0.0, iter, MPSKit.NoTimerOutput(), allocator
            )
        end
    end
    return cache
end

const X = S_x(ComplexF64, Trivial; spin = 1 // 2)
function cache_models(L)
    nearest = transverse_field_ising(; g = 2.0, L)
    longrange = long_range_ising(Float64; g = 2.0, L)
    # three-site terms give non-trivial A·A, C·A and A·B channels
    mixed = nearest + FiniteMPOHamiltonian(
        fill(ℂ^2, L),
        (1, 2, 5) => X ⊗ X ⊗ X, (2, 3, 6) => X ⊗ X ⊗ X,
        (1, 4) => X ⊗ X, (3, 6) => X ⊗ X
    )
    return (; nearest, longrange, mixed)
end

cache_algorithms(; kwargs...) = (
    DMRG(; verbosity = 0, kwargs...),
    DMRG2(; verbosity = 0, trunc = truncrank(4), kwargs...),
    DMRG(; verbosity = 0, alg_expand = OptimalExpand(; trunc = truncrank(1)), trunc = truncrank(4), kwargs...),
    DMRG(; verbosity = 0, alg_expand = SketchedExpand(; trunc = truncrank(1)), trunc = truncrank(4), kwargs...),
    DMRG(; verbosity = 0, alg_gauge = DMRG3S(0.1, ExponentialDecay(0.7)), trunc = truncrank(4), kwargs...),
)

@testset "Cached effective Hamiltonians at every local window" begin
    L = 6
    models = cache_models(L)
    @testset "$name / $(nameof(typeof(H))) / $k" for (name, H0) in pairs(models), H in (H0, FiniteMPO(H0)),
            (k, alg) in enumerate(cache_algorithms())
        Random.seed!(42)
        test_cached_windows(alg, FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^2), H)
    end
    @testset "planar $(nameof(typeof(alg)))" for alg in cache_algorithms()[1:2]
        H = force_planar(models.mixed)
        test_cached_windows(alg, FiniteMPS(randn, ComplexF64, L, TensorKit.ℙ^2, TensorKit.ℙ^3), H)
    end
    @testset "real $(nameof(typeof(alg))), L = $L" for L in (2, 4), alg in cache_algorithms()[1:2]
        H = transverse_field_ising(Float64; g = 2.0, L)
        test_cached_windows(alg, FiniteMPS(randn, Float64, L, ℂ^2, ℂ^3), H)
    end
    H1 = transverse_field_ising(; g = 2.0, L = 4)
    H2 = long_range_ising(Float64; g = 1.0, L = 4)
    @testset "$(nameof(typeof(H))) $(nameof(typeof(alg)))" for H in (
                LazySum([H1, H2]), LazySum([MultipliedOperator(H1, 0.7), MultipliedOperator(H2, 1.3)]),
            ), alg in cache_algorithms()[1:2]
        test_cached_windows(alg, FiniteMPS(randn, ComplexF64, 4, ℂ^2, ℂ^3), H)
    end
    @testset "SU2 $(nameof(typeof(alg)))" for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(8)))
        H = heisenberg_XXX(ComplexF64, SU2Irrep; spin = 1 // 2, L = 4)
        exchange = TestSetup.S_exchange(ComplexF64, SU2Irrep; spin = 1 // 2)
        H += FiniteMPOHamiltonian(fill(physicalspace(H, 1), 4), (1, 3) => exchange, (2, 4) => exchange)
        V = Rep[SU₂](0 => 2, 1 // 2 => 2, 1 => 1)
        test_cached_windows(alg, FiniteMPS(randn, ComplexF64, 4, physicalspace(H, 1), V), H)
    end
    @testset "WindowMPS with $(nameof(typeof(H)))" for H in (
            force_planar(transverse_field_ising(; g = 2.0)),
            WindowMPOHamiltonian(force_planar(transverse_field_ising(; g = 2.0)), 1:4),
        )
        Hinf = H isa WindowMPOHamiltonian ? H.left_ham : H
        gs = InfiniteMPS(randn, ComplexF64, [TensorKit.ℙ^2], [TensorKit.ℙ^3])
        ψ = WindowMPS(gs, 4)
        boundaries = environments(ψ, Hinf, ψ)
        op = H isa WindowMPOHamiltonian ? H.finite_ham : H
        initialize(ψ) = MPSKit.initialize_environments(ψ, op, ψ, copy(boundaries.GLs[1]), copy(boundaries.GRs[end]))
        cache = test_cached_windows(DMRG(; verbosity = 0), ψ, H, initialize(ψ); reference = initialize)
        @test cache.left[1].environment ≈ boundaries.GLs[1]
        @test cache.right[end].environment ≈ boundaries.GRs[end]
    end
end

@testset "Ground states agree with exact diagonalization" begin
    L = 6
    models = cache_models(L)
    @testset "$name / $(nameof(typeof(H))) / $k" for (name, H0) in pairs(models), H in (H0, FiniteMPO(H0)),
            (k, alg) in enumerate(cache_algorithms(; maxiter = 20, trunc = truncrank(8)))
        E0 = first(first(exact_diagonalization(H0)))
        Random.seed!(7)
        ψ0 = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^8)
        ψ, envs, = find_groundstate(ψ0, H, alg)
        @test real(expectation_value(ψ, H0)) ≈ E0 atol = 1.0e-5
        @test envs isa MPSKit.FiniteEnvironments
    end
end

@testset "Cached effective operators are independent snapshots" begin
    for H in (transverse_field_ising(; g = 2.0, L = 5), long_range_ising(Float64; g = 2.0, L = 5))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        allocator = sweep_allocator(ψ)
        cache = MPSKit.initialize_sweep_cache(
            ψ, H, environments(ψ, H, ψ); nsites = 2, backend = MPSKit.DefaultBackend(), allocator
        )
        pairs = cache.pairs
        retained = []
        for (constructor, x) in ((MPSKit.AC_hamiltonian, ψ.AC[1]), (MPSKit.AC2_hamiltonian, MPSKit.AC2(ψ, 1)))
            fresh = constructor(1, ψ, H, ψ, environments(ψ, H, ψ); allocator)
            first = constructor(1, ψ, H, ψ, cache; allocator)
            y = first * x
            push!(retained, (first, copy(x), copy(y)))
            for _ in 1:3
                @test constructor(1, ψ, H, ψ, cache; allocator) * x ≈ fresh * x
                # The requested matvec allocator is independent of the cache's allocator.
                @test constructor(1, ψ, H, ψ, cache; allocator = MPSKit.DefaultAllocator()) * x ≈ fresh * x
                raw = constructor(1, ψ, H, ψ, cache; prepare = false, allocator)
                @test raw * x ≈ fresh * x
                @test MPSKit.prepare_operator!!(raw) * x ≈ fresh * x
                @test first * x ≈ y
            end
        end
        alg = DMRG2(; verbosity = 0, trunc = truncrank(4))
        state = MPSKit.DMRGState(ψ, H, cache, 0, 1.0, ones(4), zeros(4), zeros(4), MPSKit.NoTimerOutput(), allocator)
        it = MPSKit.IterativeSolver(alg, state)
        state = MPSKit.sweep!(it, state, Val(:right), 1)
        left = copy(cache.left)
        state = MPSKit.sweep!(it, state, Val(:left), 1)
        @test all(a === b for (a, b) in zip(left, cache.left))
        @test cache.pairs === pairs
        @test all(H * x ≈ y for (H, x, y) in retained)
        right = copy(cache.right)
        MPSKit.sweep!(it, state, Val(:right), 2)
        @test all(a === b for (a, b) in zip(right[1:(end - 2)], cache.right[1:(end - 2)]))
        # The boundary record is never replaced.
        @test cache.right[end] === right[end]
    end
end

@testset "Effective operators reuse the prepared records" begin
    H0 = cache_models(6).mixed
    ψ = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
    @testset "$(nameof(typeof(H))), $nsites sites" for H in (H0, FiniteMPO(H0)), nsites in (1, 2)
        cache = MPSKit.initialize_sweep_cache(
            ψ, H, environments(ψ, H, ψ); nsites, backend = MPSKit.DefaultBackend(), allocator = sweep_allocator(ψ)
        )
        MPSKit.absorb_site!(cache, ψ, 1, Val(:right))
        l, r = MPSKit.left_contribution(cache, 2), MPSKit.right_contribution(cache, 1 + nsites)
        Heff = nsites == 1 ? MPSKit.AC_hamiltonian(2, ψ, H, ψ, cache) : MPSKit.AC2_hamiltonian(2, ψ, H, ψ, cache)
        if H isa FiniteMPO
            @test Heff.leftenv === l
            @test Heff.rightenv === r
            continue
        end
        continuing = nsites == 1 ? :A : :AA
        @test getfield(Heff, continuing).leftenv === l.continuing
        @test getfield(Heff, continuing).rightenv === r.continuing
        for f in fieldnames(typeof(Heff))
            f in (continuing, :backend, :allocator) && continue
            v = getfield(Heff, f)
            ismissing(v) || @test v === getfield(l.prepared, f) || v === getfield(r.prepared, f)
        end
    end
end

@testset "Transfers are independent of prepared operator contributions" begin
    L = 6
    H0 = cache_models(L).mixed
    for H in (H0, FiniteMPO(H0))
        ψ = FiniteMPS(randn, ComplexF64, L, ℂ^2, ℂ^3)
        cache = MPSKit.initialize_sweep_cache(
            ψ, H, environments(ψ, H, ψ); nsites = 1,
            backend = MPSKit.DefaultBackend(), allocator = MPSKit.DefaultAllocator(),
        )
        i = 3
        for j in 1:(i - 1)
            MPSKit.absorb_site!(cache, ψ, j, Val(:right))
        end
        # Corrupt the prepared contributions: transfers must only read the environments.
        poison(side) = H isa MPOHamiltonian ? typeof(side)(side.prepared, zero(side.continuing)) : zero(side)
        left, right = cache.left[i], cache.right[i]
        cache.left[i] = typeof(left)(left.environment, poison(left.contribution))
        cache.right[i] = typeof(right)(right.environment, poison(right.contribution))
        MPSKit.absorb_site!(cache, ψ, i, Val(:right))
        MPSKit.absorb_site!(cache, ψ, i, Val(:left))
        reference = environments(ψ, H, ψ)
        @test cache.left[i + 1].environment ≈ leftenv(reference, i + 1, ψ)
        @test cache.right[i - 1].environment ≈ rightenv(reference, i - 1, ψ)
        @test cache.environments.GLs[i + 1] === cache.left[i + 1].environment
        @test cache.environments.GRs[i] === cache.right[i - 1].environment
    end
end

@testset "Finalizers" begin
    H = long_range_ising(Float64; g = 2.0, L = 6)
    E0 = first(first(exact_diagonalization(H)))
    ψ0 = FiniteMPS(randn, ComplexF64, 6, ℂ^2, ℂ^3)
    seen = Int[]
    readonly = function (iter, ψ, H, envs)
        push!(seen, iter)
        # Ordinary read access can materialize environments beyond the sweep center.
        for i in 1:length(ψ)
            leftenv(envs, i, ψ)
            rightenv(envs, i, ψ)
        end
        return ψ, envs
    end
    alg = DMRG2(; verbosity = 0, maxiter = 2, trunc = truncrank(4), finalize = readonly)
    ψ, envs, info = @test_logs min_level = Logging.Warn find_groundstate(ψ0, H, alg)
    ref, = find_groundstate(ψ0, H, DMRG2(; verbosity = 0, maxiter = 2, trunc = truncrank(4)))
    @test seen == collect(1:info.numiter)
    @test abs(dot(ψ, ref)) ≈ norm(ψ) * norm(ref) atol = 1.0e-10
    @test envs isa MPSKit.FiniteEnvironments
    @test all(leftenv(envs, i, ψ) ≈ leftenv(environments(ψ, H, ψ), i, ψ) for i in 1:length(ψ))
    @test all(rightenv(envs, i, ψ) ≈ rightenv(environments(ψ, H, ψ), i, ψ) for i in 1:length(ψ))

    # Returning a new state and environments rebuilds the cache from them.
    replacing = function (iter, ψ, H, envs)
        ψ′ = copy(ψ)
        return ψ′, environments(ψ′, H, ψ′)
    end
    @testset "$(nameof(typeof(alg)))" for alg in (
            DMRG(; verbosity = 0, maxiter = 20, finalize = replacing),
            DMRG2(; verbosity = 0, maxiter = 20, trunc = truncrank(8), finalize = replacing),
        )
        ψ, envs, = @test_logs (:warn, r"rebuilding the sweep cache") match_mode = :any find_groundstate(ψ0, H, alg)
        @test real(expectation_value(ψ, H, envs)) ≈ E0 atol = 1.0e-6
    end
end

@testset "Environment records advance at the next local window" begin
    H = transverse_field_ising(; g = 2.0, L = 5)
    for alg in (DMRG(; verbosity = 0), DMRG2(; verbosity = 0, trunc = truncrank(4)))
        ψ = FiniteMPS(randn, ComplexF64, 5, ℂ^2, ℂ^3)
        state = MPSKit.DMRGState(ψ, H, alg, environments(ψ, H, ψ), sweep_allocator(ψ), MPSKit.NoTimerOutput())
        cache = state.envs
        @test isnothing(cache.left[2])
        for site in 1:2
            MPSKit.local_update!(
                site, Val(:right), ψ, H, alg, cache,
                state.ϵ, 0.0, 0.0, 1, state.timeroutput, state.allocator
            )
            if site == 1
                # Solving the first window leaves advancement for the next call.
                @test isnothing(cache.left[2])
            else
                @test !isnothing(cache.left[2])
                @test leftenv(cache, 2, ψ) ≈ leftenv(environments(ψ, H, ψ), 2, ψ)
            end
        end
        # Nearest-neighbor MPOs have no path through two continuing A blocks.
        alg isa DMRG2 && @test all(all(isempty, pair.channels) for pair in cache.pairs)
    end
end
